import numpy as np
import pandas as pd
import torchvision.models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import shutil as st
import copy
import time
from torchviz import make_dot
import hiddenlayer as hl
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder

img_dir = 'C:/Users/HASEE/Desktop/Pycharm_Project/data/Flavia/flavia/'
train_dir = 'C:/Users/HASEE/Desktop/Pycharm_Project/data/Flavia/train/'
val_dir = 'C:/Users/HASEE/Desktop/Pycharm_Project/data/Flavia/val/'

train_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5],
                                                                 [0.5])
                                            ]
                                           )

val_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],
                                                               [0.5])
                                          ]
                                         )

test_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],
                                                               [0.5])
                                          ]
                                         )


# 分离训练集和验证集文件
def train_val_split(imgdir, traindir, valdir, split_rate=0.8):
    # 清空验证集文件
    classlist = os.listdir(valdir)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        if os.listdir(valdir + leaf_class):  # 如果文件夹不为空，则清空文件夹
            imglist = os.listdir(valdir + leaf_class)
            for img_class in imglist:  # 读取每个类别文件夹中的图片
                os.remove(valdir + leaf_class + '/' + img_class)  # 删除图片

    # 清空训练集文件
    classlist = os.listdir(traindir)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        if os.listdir(traindir + leaf_class):  # 如果文件夹不为空，则清空文件夹
            imglist = os.listdir(traindir + leaf_class)
            for img_class in imglist:  # 读取每个类别文件夹中的图片
                os.remove(traindir + leaf_class + '/' + img_class)  # 删除图片

    classlist = os.listdir(imgdir)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        all_imgs = []
        imglist = os.listdir(imgdir + leaf_class)
        for img_class in imglist:  # 读取每个类别文件夹中的图片
            all_imgs.append(img_class)
        random.shuffle(all_imgs)  # 随机打乱图片顺序
        train_size = int(len(all_imgs) * split_rate)  # 按比例分割训练集和验证集
        val_size = len(all_imgs) - train_size
        assert (train_size > 0)
        assert (val_size > 0)

        train_imgs = all_imgs[:train_size]
        val_imgs = all_imgs[train_size:]

        for idx, imgs in enumerate(train_imgs):  # 移动图片到训练集文件夹中
            st.copy(imgdir + leaf_class + '/' + imgs, traindir + leaf_class)

        for idx, imgs in enumerate(val_imgs):  # 移动图片到验证集文件夹中
            st.copy(imgdir + leaf_class + '/' + imgs, valdir + leaf_class)

    print('dataset has been split.')


def train_data_process(train_data_path):
    train_data = ImageFolder(train_data_path, transform=train_data_transforms)
    train_data_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    class_label = train_data.classes  # 训练集的标签

    # 加载及可视化一个Batch的图像
    '''for step, (b_x, b_y) in enumerate(train_data_loader):
        if step > 0:
            break

    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组

    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii+1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()'''

    return train_data_loader, class_label


def val_data_process(val_data_path):
    val_data = ImageFolder(val_data_path, transform=val_data_transforms)
    val_data_loader = Data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)

    return val_data_loader


# 处理测试集数据
def test_data_process(test_data_path):
    test_data = ImageFolder(test_data_path, transform=test_data_transforms)
    test_data_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    return test_data_loader


def model_strcture_visual(model, img_depth, img_size, graph_type='viz'):
    if graph_type == 'viz':
        x = torch.randn(1, img_depth, img_size, img_size).requires_grad_(True)
        y = model(x)
        net = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
        net.format = 'png'
        net.directory = "C:/Users/HASEE/Desktop/Pycharm_Project"
        net.view()
    else:
        graph = hl.build_graph(model, torch.zeros([1, img_depth, img_size, img_size]))
        graph.theme = hl.graph.THEMES["blue"].copy()
        graph.save("C:/Users/HASEE/Desktop/Pycharm_Project/net.png", format="png")


# 定义一个VGG模块
def vgg_block(num_convs, in_channels, out_channels):
    '''
    :param num_convs: 卷积层数量
    :param in_channels: 输入图像的通道数
    :param out_channels: 输出图像的通道数
    :return:
    '''

    blk = []
    for num in range(num_convs):
        if num == 0:  # 定义第一个卷积块的输入输出通道数
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:  # 定义后面几个卷积块的输入输出通道数
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 定义一个最大池化层

    return nn.Sequential(*blk)


# 定义VGG-11网络结构
class VGG_Net(nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096):
        super(VGG_Net, self).__init__()

        self.conv = nn.Sequential()
        self.fc = nn.Sequential()
        # 卷积层部分
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            # 根据配置增加VGG模块
            self.conv.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))

        # 全连接层部分
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(fc_features, fc_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_hidden_units, fc_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_hidden_units, 32)
                                )

    # 前向传播路径
    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        output = self.fc(x.view(x.size(0), -1))
        print(output.shape)

        return output


# 使用VGG预训练模型中的特征提取层+全连接层
class VGG_PretrainedModel(nn.Module):
    def __init__(self, pretrained_model):
        super(VGG_PretrainedModel, self).__init__()
        self.vgg = pretrained_model
        self.fc = nn.Sequential(nn.Linear(25088, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256, 32)
                                )

    # 前向传播路径
    def forward(self, x):
        x = self.vgg(x)
        output = self.fc(x.view(x.size(0), -1))

        return output


# 定义网络权重初始化
def weights_initialize(model):
    for m in model.modules():
        # 初始化卷积层权值
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


# 定义网络的训练过程
def train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25):
    '''
    :param model: 网络模型
    :param traindataloader: 训练数据集
    :param valdataloader: 验证数据集
    :param criterion: 损失函数
    :param device: 运行设备
    :param optimizer: 优化器
    :param scheduler: 学习率调整方法
    :param num_epochs: 训练的轮数
    '''

    batch_num = len(traindataloader)  # batch数量
    best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型的参数
    # 初始化参数
    best_acc = 0.0  # 最高准确度
    train_loss_all = []  # 训练集损失函数列表
    train_acc_all = []  # 训练集准确度列表
    val_loss_all = []  # 验证集损失函数列表
    val_acc_all = []  # 验证集准确度列表
    since = time.time()  # 当前时间
    # 进行迭代训练模型
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('{} Learning Rate: {:.8f}'.format(epoch, optimizer.param_groups[0]['lr']))

        # 初始化参数
        train_loss = 0.0  # 训练集损失函数
        train_corrects = 0  # 训练集准确度
        train_num = 0  # 训练集样本数量
        val_loss = 0.0  # 验证集损失函数
        val_corrects = 0  # 验证集准确度
        val_num = 0  # 验证集样本数量

        # 对每一个mini-batch训练和计算
        for step_train, (b_x_train, b_y_train) in enumerate(traindataloader):  # 遍历训练集数据进行训练
            b_x_train = b_x_train.to(device)
            b_y_train = b_y_train.to(device)

            model.train()  # 设置模型为训练模式，启用Batch Normalization和Dropout
            output = model(b_x_train)  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            loss = criterion(output, b_y_train)  # 计算每一个batch的损失函数
            optimizer.zero_grad()  # 将梯度初始化为0
            loss.backward()  # 反向传播计算
            optimizer.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            train_loss += loss.item() * b_x_train.size(0)  # 对损失函数进行累加
            train_corrects += torch.sum(pre_lab == b_y_train.data)  # 如果预测正确，则准确度train_corrects加1
            train_num += b_x_train.size(0)  # 当前用于训练的样本数量

        for step_val, (b_x_val, b_y_val) in enumerate(valdataloader):  # 遍历验证集数据进行测试
            b_x_val = b_x_val.to(device)
            b_y_val = b_y_val.to(device)

            model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
            output = model(b_x_val)  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            loss = criterion(output, b_y_val)  # 计算每一个batch中64个样本的平均损失函数
            val_loss += loss.item() * b_x_val.size(0)  # 将验证集中每一个batch的损失函数进行累加
            val_corrects += torch.sum(pre_lab == b_y_val.data)  # 如果预测正确，则准确度val_corrects加1
            val_num += b_x_val.size(0)  # 当前用于验证的样本数量

        scheduler.step()  # 调整学习率

        # 计算并保存每一次迭代的成本函数和准确率
        train_loss_all.append(train_loss / train_num)  # 计算并保存训练集的成本函数
        train_acc_all.append(train_corrects.double().item() / train_num)  # 计算并保存训练集的准确率
        val_loss_all.append(val_loss / val_num)  # 计算并保存验证集的成本函数
        val_acc_all.append(val_corrects.double().item() / val_num)  # 计算并保存验证集的准确率
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存当前最高准确度下的模型参数
        time_use = time.time() - since  # 计算耗费时间
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    model.load_state_dict(best_model_wts)  # 加载最高准确度下的模型参数
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all}
                                 )  # 将每一代的损失函数和准确度保存为DataFrame格式

    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    return model, train_process


# 测试模型
def test_model(model, testdataloader, label, device):
    '''
    :param model: 网络模型
    :param testdataloader: 测试数据集
    :param label: 数据集标签
    :param device:
    '''

    test_corrects = 0.0
    test_num = 0
    test_acc = 0.0
    test_true = []
    test_pre = []

    with torch.no_grad():
        for test_data_x, test_data_y in testdataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
            output = model(test_data_x)  # 前向传播过程，输入为测试数据集，输出为对每个样本的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            test_corrects += torch.sum(pre_lab == test_data_y.data)  # 如果预测正确，则准确度val_corrects加1
            test_num += test_data_x.size(0)  # 当前用于训练的样本数量
            test_true.append(test_data_y.cpu().item())
            test_pre.append(pre_lab.cpu().item())

    test_acc = test_corrects.double().item() / test_num
    print("test accuracy:", test_acc)

    # 计算混淆矩阵并可视化
    plt.figure(figsize=(7, 6))
    conf_mat = confusion_matrix(test_true, test_pre)
    df_cm = pd.DataFrame(conf_mat, index=label, columns=label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.35)
    plt.show()


# 训练模型
def train_model_process(myconvnet):
    optimizer = torch.optim.SGD(myconvnet.parameters(), lr=0.02, weight_decay=0.01)  # 使用Adam优化器，学习率为0.0003
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵函数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU加速
    train_loader, class_label = train_data_process(train_dir)  # 加载训练集
    val_loader = val_data_process(val_dir)  # 加载验证集
    test_loader = test_data_process(val_dir)  # 加载测试集

    myconvnet = myconvnet.to(device)
    myconvnet, train_process = train_model(myconvnet, train_loader, val_loader, criterion, device, optimizer, scheduler, num_epochs=150)  # 进行模型训练
    test_model(myconvnet, test_loader, class_label, device)  # 使用测试集进行评估


if __name__ == '__main__':
    #train_val_split(img_dir, train_dir, val_dir)

    small_conv_arch = [(1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)]
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096

    vggnet = VGG_Net(small_conv_arch, fc_features, fc_hidden_units)
    weights_initialize(vggnet)
    '''vgg11 = models.vgg11(pretrained=True)
    model = vgg11.features
    for param in model.parameters():
        param.requires_grad_(False)
    vggnet = VGG_PretrainedModel(model)'''
    #model_strcture_visual(vggnet, img_depth=3, img_size=224, graph_type='hl')
    train_model_process(vggnet)
