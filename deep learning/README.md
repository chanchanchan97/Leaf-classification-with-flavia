Leaf Classification with Flavia
===

# 使用说明
## 1. Leaf_classification_Alexnet.py
### 1.1 介绍说明
使用Pytorch搭建AlexNet模型结构（非预训练模型），并训练该模型用于Flavia数据集的叶片分类。
### 1.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数img_dir为数据集文件路径，train_dir为划分后的训练集文件保存路径，val_dir为划分后的验证集文件保存路径，split_rate为训练集和验证集的划分比例（默认为0.8）。
### 1.3 数据处理和图像增强
train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。
train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中参数batach_size为每个batch中包含的图片数量，shuffle为是否将数据随机打乱，num_workers为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
### 1.4 模型训练
train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数model为定义的网络模型，traindataloader为训练集数据，valdataloader为验证集数据，criterion为使用的损失函数，device为使用的运算平台（'cuda'或'cpu'），optimizer为超参数的优化算法，scheduler为学习率的调整方式，num_epochs为训练的epoch数。
test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数model为训练好的网络模型，testdataloader为测试集数据，label为测试集图片的标签，device为使用的运算平台（'cuda'或'cpu'）。
train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。
