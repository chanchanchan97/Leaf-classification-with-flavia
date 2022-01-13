Leaf Classification with Flavia
===

# 1. Leaf_classification_Selfnet.py
## 1.1 介绍说明
使用Pytorch搭建**AlexNet**模型结构（非预训练模型），并训练该模型用于Flavia数据集的图片分类。
## 1.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数**img_dir**为数据集文件路径，**train_dir**为划分后的训练集文件保存路径，**val_dir**为划分后的验证集文件保存路径，**split_rate**为训练集和验证集的划分比例（默认为0.8）。
## 1.3 数据处理和图像增强
1. train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。  
2. train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中的参数**batach_size**为每个batch中包含的图片数量，**shuffle**为是否将数据随机打乱，**num_workers**为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
## 1.4 模型搭建
ConvNet(nn.Module)类定义了自己搭建的卷积神经网络模型结构，以及数据的前向传播过程。
## 1.5 模型训练
1. train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数**model**为定义的网络模型，**traindataloader**为训练集数据，**valdataloader**为验证集数据，**criterion**为使用的损失函数，**device**为使用的运算平台（'cuda'或'cpu'），**optimizer**为超参数的优化算法，**scheduler**为学习率的调整方式，**num_epochs**为训练的epoch数。  
2. test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数**model**为训练好的网络模型，**testdataloader**为测试集数据，**label**为测试集图片的标签，**device**为使用的运算平台（'cuda'或'cpu'）。  
3. train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。

# 2. Leaf_classification_Alexnet.py
## 2.1 介绍说明
使用Pytorch搭建**AlexNet**模型结构（非预训练模型），并训练该模型用于Flavia数据集的图片分类。
## 2.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数**img_dir**为数据集文件路径，**train_dir**为划分后的训练集文件保存路径，**val_dir**为划分后的验证集文件保存路径，**split_rate**为训练集和验证集的划分比例（默认为0.8）。
## 2.3 数据处理和图像增强
1. train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。  
2. train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中的参数**batach_size**为每个batch中包含的图片数量，**shuffle**为是否将数据随机打乱，**num_workers**为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
## 2.4 模型搭建
AlexNet(nn.Module)类定义了AlexNet的模型结构，以及数据的前向传播过程。
## 2.5 模型训练
1. train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数**model**为定义的网络模型，**traindataloader**为训练集数据，**valdataloader**为验证集数据，**criterion**为使用的损失函数，**device**为使用的运算平台（'cuda'或'cpu'），**optimizer**为超参数的优化算法，**scheduler**为学习率的调整方式，**num_epochs**为训练的epoch数。  
2. test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数**model**为训练好的网络模型，**testdataloader**为测试集数据，**label**为测试集图片的标签，**device**为使用的运算平台（'cuda'或'cpu'）。  
3. train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。

# 3. Leaf_classification_GoogLeNet.py
## 3.1 介绍说明
使用Pytorch搭建**GoogLeNet**模型结构（非预训练模型），并训练该模型用于Flavia数据集的图片分类。
## 3.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数**img_dir**为数据集文件路径，**train_dir**为划分后的训练集文件保存路径，**val_dir**为划分后的验证集文件保存路径，**split_rate**为训练集和验证集的划分比例（默认为0.8）。
## 3.3 数据处理和图像增强
1. train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。  
2. train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中的参数**batach_size**为每个batch中包含的图片数量，**shuffle**为是否将数据随机打乱，**num_workers**为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
## 3.4 模型搭建
1. Inception(nn.Module)类定义了Inception模块结构。
2. GoogLeNet()函数定义了GoogLeNet的网络结构，其中由多个Inception模块构成。
## 3.5 权重初始化
weights_initialize(model)函数的作用是对网络模型中卷积层的权重使用Kaiming初始化。
## 3.6 模型训练
1. train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数**model**为定义的网络模型，**traindataloader**为训练集数据，**valdataloader**为验证集数据，**criterion**为使用的损失函数，**device**为使用的运算平台（'cuda'或'cpu'），**optimizer**为超参数的优化算法，**scheduler**为学习率的调整方式，**num_epochs**为训练的epoch数。  
2. test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数**model**为训练好的网络模型，**testdataloader**为测试集数据，**label**为测试集图片的标签，**device**为使用的运算平台（'cuda'或'cpu'）。  
3. train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。

# 4. Leaf_classification_VGG11.py
## 4.1 介绍说明
使用Pytorch搭建**VGG11**模型结构（非预训练模型），并训练该模型用于Flavia数据集的图片分类。
## 4.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数**img_dir**为数据集文件路径，**train_dir**为划分后的训练集文件保存路径，**val_dir**为划分后的验证集文件保存路径，**split_rate**为训练集和验证集的划分比例（默认为0.8）。
## 4.3 数据处理和图像增强
1. train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。  
2. train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中的参数**batach_size**为每个batch中包含的图片数量，**shuffle**为是否将数据随机打乱，**num_workers**为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
## 4.4 模型搭建
1. vgg_block(num_convs, in_channels, out_channels)函数定义了vgg模块结构，其中参数**num_convs**为卷积层数量，**in_channels**为输入图像的通道数，**out_channels**为输出图像的通道数。
2. VGG_Net(nn.Module)类定义了VGG11的网络结构，其中由多个vgg模块构成。
## 4.5 权重初始化
weights_initialize(model)函数的作用是对网络模型中卷积层的权重使用Kaiming初始化。
## 4.6 模型训练
1. train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数**model**为定义的网络模型，**traindataloader**为训练集数据，**valdataloader**为验证集数据，**criterion**为使用的损失函数，**device**为使用的运算平台（'cuda'或'cpu'），**optimizer**为超参数的优化算法，**scheduler**为学习率的调整方式，**num_epochs**为训练的epoch数。  
2. test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数**model**为训练好的网络模型，**testdataloader**为测试集数据，**label**为测试集图片的标签，**device**为使用的运算平台（'cuda'或'cpu'）。  
3. train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。


# 5. Leaf_classification_Resnet18.py
## 5.1 介绍说明
使用Pytorch搭建**Resnet18**模型结构（非预训练模型），并训练该模型用于Flavia数据集的图片分类。
## 5.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数**img_dir**为数据集文件路径，**train_dir**为划分后的训练集文件保存路径，**val_dir**为划分后的验证集文件保存路径，**split_rate**为训练集和验证集的划分比例（默认为0.8）。
## 5.3 数据处理和图像增强
1. train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。  
2. train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中的参数**batach_size**为每个batch中包含的图片数量，**shuffle**为是否将数据随机打乱，**num_workers**为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
## 5.4 模型搭建
1. resnet_block(in_channels, out_channels, num_residuals, first_block=False)函数定义了残差块结构，其中参数**in_channels**为输入图像的通道数，**out_channels**为输出图像的通道数，**num_residuals**为残差块的数量，**first_block**为是否是第一个残差块。
2. ResNet()函数定义了ResNet18的网络结构，其中由多个残差块构成。
## 4.5 权重初始化
weights_initialize(model)函数的作用是对网络模型中卷积层的权重使用Kaiming初始化。
## 5.6 模型训练
1. train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数**model**为定义的网络模型，**traindataloader**为训练集数据，**valdataloader**为验证集数据，**criterion**为使用的损失函数，**device**为使用的运算平台（'cuda'或'cpu'），**optimizer**为超参数的优化算法，**scheduler**为学习率的调整方式，**num_epochs**为训练的epoch数。  
2. test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数**model**为训练好的网络模型，**testdataloader**为测试集数据，**label**为测试集图片的标签，**device**为使用的运算平台（'cuda'或'cpu'）。  
3. train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。

# 6. Leaf_classification_HRnet.py
## 6.1 介绍说明
使用Pytorch搭建**HRNet**模型结构（非预训练模型），并训练该模型用于Flavia数据集的图片分类。
## 6.2 训练集和验证集文件的划分
train_val_split(imgdir, traindir, valdir, split_rate=0.8)函数的作用是将数据集文件按一定比例划分为训练集和验证集，其中参数**img_dir**为数据集文件路径，**train_dir**为划分后的训练集文件保存路径，**val_dir**为划分后的验证集文件保存路径，**split_rate**为训练集和验证集的划分比例（默认为0.8）。
## 6.3 数据处理和图像增强
1. train_data_transforms、val_data_transforms和test_data_transforms分别定义了训练集、验证集和测试集图片的预处理操作，其中包含了图像压缩、图像增强和数据归一化等操作。  
2. train_data_process(train_data_path)、val_data_process(val_data_path)和test_data_process(test_data_path)函数的作用分别是对训练集、验证集和测试集图片进行预处理，并打包成多个batch用于后面训练、验证和测试。其中Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)函数中的参数**batach_size**为每个batch中包含的图片数量，**shuffle**为是否将数据随机打乱，**num_workers**为加载batch数据的进程数（Windows中num_workers=0，否则会出错）。
## 6.4 模型搭建
1. Bottleneck(nn.Module)类和BasicBlock(nn.Module)类分别定义了瓶颈块和残差块结构。
2. HighResolutionModule(nn.Module)类主要定义了不同分辨率的特征图之间的转换和融合，以及分支的构建。
3. HRnet(nn.Module)类定义了HRNet的网络结构，以及数据的前向传播过程。
## 6.5 权重初始化
weights_initialize(model)函数的作用是对网络模型中卷积层的权重使用Kaiming初始化。
## 6.6 模型训练
1. train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25)函数定义了模型的训练和验证过程，其中参数**model**为定义的网络模型，**traindataloader**为训练集数据，**valdataloader**为验证集数据，**criterion**为使用的损失函数，**device**为使用的运算平台（'cuda'或'cpu'），**optimizer**为超参数的优化算法，**scheduler**为学习率的调整方式，**num_epochs**为训练的epoch数。  
2. test_model(model, testdataloader, label, device)函数定义了模型的测试过程，其中参数**model**为训练好的网络模型，**testdataloader**为测试集数据，**label**为测试集图片的标签，**device**为使用的运算平台（'cuda'或'cpu'）。  
3. train_model_process(myconvnet)函数的作用是获取上述定义的参数，并分别调用train_model()函数和test_model()函数对模型进行训练和测试。

# 7. CAM_Visualization.py
