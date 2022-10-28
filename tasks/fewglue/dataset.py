from ast import Raise
from datasets import load_dataset, load_metric, concatenate_datasets
import random
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
}

logger = logging.getLogger(__name__)


class FewGlueDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})
        self.prompt = self.tokenizer.additional_special_tokens[0]
        self.mask = self.tokenizer.mask_token
        self.pad = self.tokenizer.pad_token

        if data_args.dataset_name in ["cb", "rte"]:
            self.verbalizer_dict = {
                "2": {
                    "0": {"0": "yes", "1": "no", "-1": "no"},
                    "1": {"0": "true", "1": "false", "-1": "a"},
                },
                "3": {
                    "0": {"0": "yes", "1": "no", "2": "maybe", "-1": "a"},
                    "1": {"0": "true", "1": "false", "2": "maybe", "-1": "a"},
                },
            }
        elif data_args.dataset_name in ["boolq", "wic", "wsc", "copa", "multirc"]:
            self.verbalizer_dict = {
                "2": {
                    "0": {"0": "no", "1": "yes", "-1": "no"},
                    "1": {"0": "false", "1": "true", "-1": "a"},
                },
            }


        self.pre_seq_len = self.model_args.pre_seq_len
        self.raw_datasets = load_dataset("../../../tasks/fewglue/fewglue_dataset.py", data_args.dataset_name, ignore_verifications=True) 

        self.label_list = self.raw_datasets["train"].features["label"].names
        self.num_labels = len(self.label_list)

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        self.label2token = self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id]
        self.token2label = {v: k for k, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()}
        self.label_token_list = [v for _, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()]
        self.label_token_ids_list = [self.tokenizer.encode(l)[1:-1] for l in self.label_token_list]
        self.label2id = {label: id for id, label in enumerate(self.label_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}
        print(f"{self.label2token}")
        print(f"{self.token2label}")
        print(f"{self.label_token_list}")
        print(f"{self.label_token_ids_list}")
        print(f"{self.label2id}")
        print(f"{self.id2label}")

        if self.data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        self.preprocess_function_one_list = {
            "boolq": self.preprocess_function_one_boolq,
            "rte": self.preprocess_function_one_nli,
            "cb": self.preprocess_function_one_nli,
            "wic": self.preprocess_function_one_wic,
            "wsc": self.preprocess_function_one_wsc,
            "copa": self.preprocess_function_one_copa,
            "multirc": self.preprocess_function_one_multirc
        }

        self.preprocess_function_two_list = {
            "boolq": self.preprocess_function_two,
            "rte": self.preprocess_function_two,
            "cb": self.preprocess_function_two,
            "wic": self.preprocess_function_two,
            "wsc": self.preprocess_function_two_wsc,
            "copa": self.preprocess_function_two_copa,
            "multirc": self.preprocess_function_two
        }

        if self.data_args.dataset_name == "wsc":
            self.raw_datasets["train"] = self.raw_datasets["train"].map(lambda example: {"set_type": "train"}).filter(lambda example: example['label'] == 1)
            self.raw_datasets["validation"] = self.raw_datasets["validation"].map(lambda example: {"set_type": "validation"})
            self.raw_datasets["test"] = self.raw_datasets["validation"].map(lambda example: {"set_type": "test"})
        elif self.data_args.dataset_name == "copa":
            train_len = len(self.raw_datasets["train"])
            updated_dataset =  self.raw_datasets["train"].map(lambda example: {'idx': example['idx'] + train_len, 'choice1': example['choice2'], 'choice2': example["choice1"], 'label': 1 - example['label']})
            self.raw_datasets["train"] = concatenate_datasets([self.raw_datasets["train"], updated_dataset])

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function_one_list[self.data_args.dataset_name],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Runing one tokenization"
        )

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function_two_list[self.data_args.dataset_name],
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Runing two tokenization"
        )

        if 1 == 0: # Only works in multi-task setting
            self.data_collator = DataCollatorWithPadding
            self.raw_datasets = self.raw_datasets.map(
                self.preprocess_function_three,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Runing three tokenization"
            )

        if training_args.do_train:
            self.train_dataset = self.raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        if training_args.do_eval:
            self.eval_dataset = self.raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = self.raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = load_metric("../../../tasks/superglue/superglue_metric.py", data_args.dataset_name)
        self.test_key = "accuracy" if data_args.dataset_name not in ["record", "multirc"] else "f1"


    def data_collator(self, features):
        first = features[0]
        batch = {}

        # labels work
        # labels, label_token_ids_list -> batch
        for f in features:
            if self.config.model_type in ["gpt2", "t5"]: # t5和gpt2
                f["labels"] = f["label_token_ids"]
                f["label_token_ids_list"] = self.label_token_ids_list
            elif self.config.model_type in ["bert", "roberta", "albert", "deberta-v2"]: # 普通的模型
                if self.data_args.dataset_name in ["boolq", "cb", "rte", "wic", "multirc"]:
                    label_token_ids = self.label_token_ids_list[f["label"]]
                    label_ids = [-100 for _ in range(len(f["input_ids"]))]
                    mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                    label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
                    f["labels"] = label_ids
                    f["label_token_ids_list"] = self.label_token_ids_list
                elif self.data_args.dataset_name in ["wsc"]: # 不需要 label_token_ids_list，需要label_token_ids
                    label_ids = [-100 for _ in range(len(f["input_ids"]))]
                    mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                    # label_ids[mask_start + len(f["label_token_ids"][1:-1]): mask_start + len(f["label_token_ids"][1:-1]) + f["num_pad"]] = [self.tokenizer.pad_token_id] * f["num_pad"]
                    label_ids[mask_start: mask_start + len(f["label_token_ids"][1:-1])] = f["label_token_ids"][1:-1]
                    f["labels"] = label_ids
                elif self.data_args.dataset_name in ["copa"]: # 不需要 label_token_ids_list和label_token_ids 换成了choice1和choice2
                    label_ids = [-100 for _ in range(len(f["input_ids"]))]
                    mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
                    label_ids[mask_start: mask_start + len(f["label_token_ids"][1:-1])] = f["label_token_ids"][1:-1]
                    f["labels"] = label_ids
                    for choice in ["choice1", "choice2"]:
                        mask_end = mask_start + len(f[f'{choice}_ids'][1:-1])
                        label_ids = [-100 for _ in range(len(f["input_ids"]))]
                        label_ids[mask_start: mask_end] = f[f'{choice}_ids'][1:-1]
                        f[f'{choice}_ids'] = label_ids

        # Padding work
        #input_ids, sentence_ids, labels -> batch
        pad_key_list = ["input_ids", "sentence_ids", "labels"]
        if self.data_args.dataset_name == "copa":
            pad_key_list.extend(["choice1_ids", "choice2_ids"])
        for key in pad_key_list:
            result = self.tokenizer.pad(
                {"input_ids": [f[key] for f in features]},
                padding=self.padding,
                max_length=self.max_seq_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch[key] = result["input_ids"]
            if key == "input_ids" and "attention_mask" not in batch.keys():
                batch["attention_mask"] = result["attention_mask"]

        reduced_column = []
        reduced_column.extend(["input_ids", "sentence_ids", "attention_mask", "label_token_ids", "labels"]) # data_collator pad
        reduced_column.extend(["idx", "input_tokens", "sentence_tokens", "label_tokens"]) # preprocess_function_pre
        reduced_column.extend(["choice1_ids", "choice2_ids"]) # copa
        
        for k, v in first.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])

        # WSC thing
        if self.data_args.dataset_name == "wsc":
            batch["label_token_ids"] = [f["label_token_ids"] for f in features]

        return batch

    def preprocess_function_three(self, example):
        if self.data_args.dataset_name in ["boolq", "cb," "rte", "wic", "multirc"]:
            label_token_ids = self.label2token[str(example["label"])]
            label_ids = [-100 for _ in range(len(example["input_ids"]))]
            mask_start = example["input_ids"].index(self.tokenizer.mask_token_id)
            label_ids[mask_start: mask_start + len(label_token_ids)] = label_token_ids
        elif self.data_args.dataset_name in ["wsc", "copa"]: # 不需要 label_token_ids_list，需要label_token_ids
            label_ids = [-100 for _ in range(len(example["input_ids"]))]
            mask_start = example["input_ids"].index(self.tokenizer.mask_token_id)
            label_ids[mask_start: mask_start + len(example["label_token_ids"][1:-1])] = example["label_token_ids"][1:-1]

        example["labels"] = label_ids
        return example


    def preprocess_function_two(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        return result


    def preprocess_function_two_wsc(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "label_token_ids": self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        return result


    def preprocess_function_two_copa(self, examples):
        result = {
            "input_ids": self.tokenizer(examples["input_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "sentence_ids": self.tokenizer(examples["sentence_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "label_token_ids": self.tokenizer(examples["label_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "choice1_ids": self.tokenizer(examples["choice1_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
            "choice2_ids": self.tokenizer(examples["choice2_tokens"], padding=False, max_length=512, truncation=True)["input_ids"],
        }
        return result


    def preprocess_function_one_boolq(self, examples):
        passage = examples["passage"][:450]
        question = examples["question"]
        result = {}

        # input_tokens
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([passage, question, self.prompt * self.pre_seq_len, self.mask])
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([passage, '. Question: ', question, self.prompt * self.pre_seq_len, '? Answer: ', self.mask, '.'])
        elif self.model_args.template_id == "2":
            result["input_tokens"] = ''.join([passage, '. Based on the previous passage, ', question, self.prompt * self.pre_seq_len, '?', self.mask, '.'])
        elif self.model_args.template_id == "3":
            result["input_tokens"] = ''.join(['Based on the following passage, ', question, self.prompt * self.pre_seq_len, '?', self.mask, '.', passage])
        elif self.model_args.template_id == "4": # No mask, for t5/gpt-2
            result["input_tokens"] =  ''.join([passage, question, self.prompt * self.pre_seq_len, "? Answer: "])
        else:
            raise NotImplementedError(
                "The template id {} has not been defined.".format(self.model_args.template_id)
            )

        # sentence_tokens
        result["sentence_tokens"] = ''.join([question])

        # label_tokens
        result["label"] = examples["label"]
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        elif self.config.model_type == "gpt2":
            result["label_tokens"] = ''.join([result["input_tokens"], result["label_tokens"]])

        return result          


    def preprocess_function_one_nli(self, examples):
        premise = examples["premise"]
        hypothesis = examples["hypothesis"]
        result = {}

        # input_tokens
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([premise, self.prompt * self.pre_seq_len, self.mask, hypothesis])
        if self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([premise, '. Question: ', hypothesis, self.prompt * self.pre_seq_len, '? Answer: ', self.mask, '.'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([premise, hypothesis])

        # label_tokens
        result["label"] = examples["label"]
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        elif self.config.model_type == "gpt2":
            result["label_tokens"] = ''.join([result["input_tokens"], result["label_tokens"]])

        return result    


    def preprocess_function_one_wic(self, examples):
        sentence1 = examples["sentence1"]
        sentence2 = examples["sentence2"]
        word = examples["word"]
        result = {}

        # input_tokens
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([sentence1, word, self.prompt * self.pre_seq_len, self.mask, sentence2])
        elif self.model_args.template_id == "2":
            result["input_tokens"] =  ''.join(['"', sentence1, '" / "', sentence2, self.prompt * self.pre_seq_len, '" Similar sense of "', word, '"?', self.mask, '.'])
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([sentence1, sentence2, 'Does ' + word + ' have the same meaning in both sentences?', self.prompt * self.pre_seq_len, self.mask])
        elif self.model_args.template_id == "3":
            result["input_tokens"] =  ''.join([word, ' . Sense (1) (a) "', sentence1, '" (', self.mask, ') "', sentence2, '"'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([sentence1, sentence2])

        # label_tokens
        result["label"] = examples["label"]
        if self.config.model_type == "t5":
            result["label_tokens"] = ''.join([result["label_tokens"]])
        elif self.config.model_type == "gpt2":
            result["label_tokens"] = ''.join([result["input_tokens"], result["label_tokens"]])

        return result     


    def preprocess_function_one_wsc(self, examples):
        text = examples["text"]
        span1_text = examples["span1_text"]
        span2_text = examples["span2_text"]
        num_pad = self.rng.randint(0, 3) if examples["set_type"] == "train" else 1
        masks = self.mask * (len(self.tokenizer(span1_text, padding=self.padding, max_length=512, truncation=True)["input_ids"][1: -1]) + num_pad)
        result = {}
        result["num_pad"] = num_pad

        # input_tokens
        if self.model_args.template_id == "0":
            pass
        elif self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([text, self.prompt * self.pre_seq_len, "The pronoun '*", span2_text + "*' refers to", masks, '.'])
        elif self.model_args.template_id == "2":
            result["input_tokens"] =  ''.join([text, self.prompt * self.pre_seq_len, "In the previous sentence, the pronoun '*", span2_text, "*' refers to", masks, '.'])
        elif self.model_args.template_id == "3":
            result["input_tokens"] =  ''.join([text, self.prompt * self.pre_seq_len, "Question: In the passage above, what does the pronoun '*", span2_text, "*' refer to? Answer: ", self.masks + '.'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([text])

        # label_tokens
        result["label"] = examples["label"]
        result["label_tokens"] = span1_text

        return result   


    def preprocess_function_one_copa(self, examples):
        premise = examples["premise"]
        question = examples["question"]
        choice1 = examples["choice1"]
        choice2 = examples["choice2"]
        num_masks = max(len(self.tokenizer(choice, padding=self.padding, max_length=512, truncation=True)["input_ids"][1: -1]) for choice in [choice1, choice2])
        result = {}

        # input_tokens

        if question == 'cause':
            joiner = "because"
            if self.model_args.template_id == "0":
                result["input_tokens"] =  ''.join(['"', choice1, '" or "', choice2, '"?', premise, joiner, self.prompt * self.pre_seq_len, self.mask * num_masks, '.'])
            elif self.model_args.template_id == "1":
                result["input_tokens"] =  ''.join([choice1, 'or', choice2, '?', premise, self.prompt * self.pre_seq_len, joiner, self.mask * num_masks, '.'])
        else:
            joiner = "so"
            if self.model_args.template_id == "0":
                result["input_tokens"] =  ''.join(['"', choice1, '" or "', choice2, '"?', premise, ', ', joiner, self.prompt * self.pre_seq_len, self.mask * num_masks, '.'])
            elif self.model_args.template_id == "1":
                result["input_tokens"] =  ''.join([choice1, 'or', choice2, '?', premise, ', ', joiner, self.prompt * self.pre_seq_len, self.mask * num_masks, '.'])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([joiner])

        # label_tokens
        result["label"] = examples["label"]
        result["choice1_tokens"] = choice1
        result["choice2_tokens"] = choice2
        result["label_tokens"] = [choice1, choice2][result["label"]] # Only used in multi-task setting

        return result   


    def preprocess_function_one_multirc(self, examples):
        paragraph = examples["paragraph"][:450]
        question = examples["question"]
        answer = examples["answer"]
        result = {}

        # input_tokens
        if self.model_args.template_id == "1":
            result["input_tokens"] =  ''.join([paragraph, '. Question: ', question, self.prompt * self.pre_seq_len, '? Is it ', answer, '?', self.mask, '.'])
        if self.model_args.template_id == "0":
            result["input_tokens"] =  ''.join([paragraph, '. Question: ', question, self.prompt * self.pre_seq_len, '? Is the correct answer "', answer, '"?', self.mask, '.'])
        if self.model_args.template_id == "2":
            result["input_tokens"] =  ''.join([paragraph, '. Based on the previous passage, ', question, self.prompt * self.pre_seq_len, '? Is "', answer, '" a correct answer?', self.mask, '.'])
        if self.model_args.template_id == "3":
            result["input_tokens"] =  ''.join([paragraph, question, self.prompt * self.pre_seq_len, '- [', self.mask, ']', answer])

        # sentence_tokens
        result["sentence_tokens"] = ''.join([question])

        # label_tokens
        result["label"] = examples["label"]
        result["label_tokens"] = self.label2token[str(examples["label"])]

        return result   


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        if self.data_args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.data_args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

