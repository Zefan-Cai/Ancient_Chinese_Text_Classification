# Ancient_Chinese_Text_Classification

- 本工作基于HuggingFace的Transformer库加载NLP语言模型，利用PyTorch进行深度学习训练，并且继承了Transformer库中的Trainer类来进行训练。

## 如何添加新的数据集
- 先在tasks中编辑数据集载入、训练、测试的代码。
- 在脚本中修改数据集名称。

## 如何语言模型
- 如果要更改新的语言模型，需要先下载要使用的语言模型。
- 接着在run_script文件夹创建新的运行脚本，并且在脚本下写入语言模型的路径。

## 如何修改实验参数
- 具体实验参数设置可以通过修改脚本来实现。其中训练集样本个数的修改也可以通过修改脚本实现。
- 具体每个参数的意义请参考文档https://huggingface.co/docs/transformers/index
