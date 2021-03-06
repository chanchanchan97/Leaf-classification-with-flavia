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


# ?????????????????????????????????
def train_val_split(imgdir, traindir, valdir, split_rate=0.8):
    # ?????????????????????
    classlist = os.listdir(valdir)
    for leaf_class in classlist:  # ??????????????????????????????
        if os.listdir(valdir + leaf_class):  # ?????????????????????????????????????????????
            imglist = os.listdir(valdir + leaf_class)
            for img_class in imglist:  # ???????????????????????????????????????
                os.remove(valdir + leaf_class + '/' + img_class)  # ????????????

    # ?????????????????????
    classlist = os.listdir(traindir)
    for leaf_class in classlist:  # ??????????????????????????????
        if os.listdir(traindir + leaf_class):  # ?????????????????????????????????????????????
            imglist = os.listdir(traindir + leaf_class)
            for img_class in imglist:  # ???????????????????????????????????????
                os.remove(traindir + leaf_class + '/' + img_class)  # ????????????

    classlist = os.listdir(imgdir)
    for leaf_class in classlist:  # ??????????????????????????????
        all_imgs = []
        imglist = os.listdir(imgdir + leaf_class)
        for img_class in imglist:  # ???????????????????????????????????????
            all_imgs.append(img_class)
        random.shuffle(all_imgs)  # ????????????????????????
        train_size = int(len(all_imgs) * split_rate)  # ????????????????????????????????????
        val_size = len(all_imgs) - train_size
        assert (train_size > 0)
        assert (val_size > 0)

        train_imgs = all_imgs[:train_size]
        val_imgs = all_imgs[train_size:]

        for idx, imgs in enumerate(train_imgs):  # ????????????????????????????????????
            st.copy(imgdir + leaf_class + '/' + imgs, traindir + leaf_class)

        for idx, imgs in enumerate(val_imgs):  # ????????????????????????????????????
            st.copy(imgdir + leaf_class + '/' + imgs, valdir + leaf_class)

    print('dataset has been split.')


def train_data_process(train_data_path):
    train_data = ImageFolder(train_data_path, transform=train_data_transforms)
    train_data_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    class_label = train_data.classes  # ??????????????????

    # ????????????????????????Batch?????????
    '''for step, (b_x, b_y) in enumerate(train_data_loader):
        if step > 0:
            break

    batch_x = b_x.squeeze().numpy()  # ????????????????????????1??????????????????Numpy??????
    batch_y = b_y.numpy()  # ??????????????????Numpy??????

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


# ?????????????????????
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


# ????????????VGG??????
def vgg_block(num_convs, in_channels, out_channels):
    '''
    :param num_convs: ???????????????
    :param in_channels: ????????????????????????
    :param out_channels: ????????????????????????
    :return:
    '''

    blk = []
    for num in range(num_convs):
        if num == 0:  # ????????????????????????????????????????????????
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:  # ???????????????????????????????????????????????????
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # ???????????????????????????

    return nn.Sequential(*blk)


# ??????VGG-11????????????
class VGG_Net(nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096):
        super(VGG_Net, self).__init__()

        self.conv = nn.Sequential()
        self.fc = nn.Sequential()
        # ???????????????
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            # ??????????????????VGG??????
            self.conv.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))

        # ??????????????????
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(fc_features, fc_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_hidden_units, fc_hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_hidden_units, 32)
                                )

    # ??????????????????
    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        output = self.fc(x.view(x.size(0), -1))
        print(output.shape)

        return output


# ???????????????????????????
def weights_initialize(model):
    for m in model.modules():
        # ????????????????????????
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


# ???????????????????????????
def train_model(model, traindataloader, valdataloader, criterion, device, optimizer, scheduler, num_epochs=25):
    '''
    :param model: ????????????
    :param traindataloader: ???????????????
    :param valdataloader: ???????????????
    :param criterion: ????????????
    :param device: ????????????
    :param optimizer: ?????????
    :param scheduler: ?????????????????????
    :param num_epochs: ???????????????
    '''

    batch_num = len(traindataloader)  # batch??????
    best_model_wts = copy.deepcopy(model.state_dict())  # ???????????????????????????
    # ???????????????
    best_acc = 0.0  # ???????????????
    train_loss_all = []  # ???????????????????????????
    train_acc_all = []  # ????????????????????????
    val_loss_all = []  # ???????????????????????????
    val_acc_all = []  # ????????????????????????
    since = time.time()  # ????????????
    # ????????????????????????
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('{} Learning Rate: {:.8f}'.format(epoch, optimizer.param_groups[0]['lr']))

        # ???????????????
        train_loss = 0.0  # ?????????????????????
        train_corrects = 0  # ??????????????????
        train_num = 0  # ?????????????????????
        val_loss = 0.0  # ?????????????????????
        val_corrects = 0  # ??????????????????
        val_num = 0  # ?????????????????????

        # ????????????mini-batch???????????????
        for step_train, (b_x_train, b_y_train) in enumerate(traindataloader):  # ?????????????????????????????????
            b_x_train = b_x_train.to(device)
            b_y_train = b_y_train.to(device)

            model.train()  # ????????????????????????????????????Batch Normalization???Dropout
            output = model(b_x_train)  # ????????????????????????????????????batch??????????????????batch??????????????????
            pre_lab = torch.argmax(output, 1)  # ??????????????????????????????????????????
            loss = criterion(output, b_y_train)  # ???????????????batch???????????????
            optimizer.zero_grad()  # ?????????????????????0
            loss.backward()  # ??????????????????
            optimizer.step()  # ?????????????????????????????????????????????????????????????????????????????????loss????????????????????????
            train_loss += loss.item() * b_x_train.size(0)  # ???????????????????????????
            train_corrects += torch.sum(pre_lab == b_y_train.data)  # ?????????????????????????????????train_corrects???1
            train_num += b_x_train.size(0)  # ?????????????????????????????????

        for step_val, (b_x_val, b_y_val) in enumerate(valdataloader):  # ?????????????????????????????????
            b_x_val = b_x_val.to(device)
            b_y_val = b_y_val.to(device)

            model.eval()  # ???????????????????????????????????????Batch Normalization???Dropout
            output = model(b_x_val)  # ????????????????????????????????????batch??????????????????batch??????????????????
            pre_lab = torch.argmax(output, 1)  # ??????????????????????????????????????????
            loss = criterion(output, b_y_val)  # ???????????????batch???64??????????????????????????????
            val_loss += loss.item() * b_x_val.size(0)  # ????????????????????????batch???????????????????????????
            val_corrects += torch.sum(pre_lab == b_y_val.data)  # ?????????????????????????????????val_corrects???1
            val_num += b_x_val.size(0)  # ?????????????????????????????????

        scheduler.step()  # ???????????????

        # ?????????????????????????????????????????????????????????
        train_loss_all.append(train_loss / train_num)  # ???????????????????????????????????????
        train_acc_all.append(train_corrects.double().item() / train_num)  # ????????????????????????????????????
        val_loss_all.append(val_loss / val_num)  # ???????????????????????????????????????
        val_acc_all.append(val_corrects.double().item() / val_num)  # ????????????????????????????????????
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # ?????????????????????
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # ??????????????????????????????
            best_model_wts = copy.deepcopy(model.state_dict())  # ?????????????????????????????????????????????
        time_use = time.time() - since  # ??????????????????
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # ??????????????????
    model.load_state_dict(best_model_wts)  # ???????????????????????????????????????
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all}
                                 )  # ????????????????????????????????????????????????DataFrame??????

    # ???????????????????????????????????????????????????????????????????????????
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


# ????????????
def test_model(model, testdataloader, label, device):
    '''
    :param model: ????????????
    :param testdataloader: ???????????????
    :param label: ???????????????
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
            model.eval()  # ???????????????????????????????????????Batch Normalization???Dropout
            output = model(test_data_x)  # ?????????????????????????????????????????????????????????????????????????????????
            pre_lab = torch.argmax(output, 1)  # ??????????????????????????????????????????
            test_corrects += torch.sum(pre_lab == test_data_y.data)  # ?????????????????????????????????val_corrects???1
            test_num += test_data_x.size(0)  # ?????????????????????????????????
            test_true.append(test_data_y.cpu().item())
            test_pre.append(pre_lab.cpu().item())

    test_acc = test_corrects.double().item() / test_num
    print("test accuracy:", test_acc)

    # ??????????????????????????????
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


# ????????????
def train_model_process(myconvnet):
    optimizer = torch.optim.SGD(myconvnet.parameters(), lr=0.02, weight_decay=0.01)  # ??????Adam????????????????????????0.0003
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()  # ??????????????????????????????
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU??????
    train_loader, class_label = train_data_process(train_dir)  # ???????????????
    val_loader = val_data_process(val_dir)  # ???????????????
    test_loader = test_data_process(val_dir)  # ???????????????

    myconvnet = myconvnet.to(device)
    myconvnet, train_process = train_model(myconvnet, train_loader, val_loader, criterion, device, optimizer, scheduler, num_epochs=150)  # ??????????????????
    test_model(myconvnet, test_loader, class_label, device)  # ???????????????????????????


if __name__ == '__main__':
    #train_val_split(img_dir, train_dir, val_dir)

    small_conv_arch = [(1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)]  # vgg????????????
    fc_features = 512 * 7 * 7  # ??????????????????????????????
    fc_hidden_units = 4096  # ??????????????????????????????

    vggnet = VGG_Net(small_conv_arch, fc_features, fc_hidden_units)
    weights_initialize(vggnet)
    train_model_process(vggnet)
