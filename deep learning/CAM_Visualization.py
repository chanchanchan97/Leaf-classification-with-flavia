import cv2
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import logging
import matplotlib.pyplot as plt
import torchsummary as summary
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_path = "C:/Users/HASEE/Desktop/Pycharm_Project/images/flavia"
save_path = "C:/Users/HASEE/Desktop/Pycharm_Project/images/CAM"
img_path = "C:/Users/HASEE/Desktop/Pycharm_Project/images/flavia/castor aralia/1396.jpg"

class_name = ['Anhui Barberry', 'Beale\'s barberry', 'Big-fruited Holly', 'Canadian poplar', 'Chinese Toon', 'Chinese cinnamon',
              'Chinese horse chestnut', 'Chinese redbud', 'Chinese tulip tree', 'Crape myrtle, Crepe myrtle', 'Ford Woodlotus',
              'Glossy Privet', 'Japan Arrowwood', 'Japanese Flowering Cherry', 'Japanese cheesewood', 'Japanese maple', 'Nanmu',
              'camphortree', 'castor aralia', 'deodar', 'ginkgo, maidenhair tree', 'goldenrain tree', 'oleander', 'peach',
              'pubescent bamboo', 'southern magnolia', 'sweet osmanthus', 'tangerine', 'trident maple', 'true indigo',
              'wintersweet', 'yew plum pine']
fixed_angle = [45, 90, 135, 180, 225, 315]
rotate = True
cbam = True

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.769235, 0.869587, 0.733468],
                                                     [0.339912, 0.204988, 0.388254])
                                ]
                               )

logger = logging.getLogger(__name__)


# 瓶颈块结构
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(out_channels, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=stride,
                                             padding=0, bias=False),
                                   nn.BatchNorm2d(out_channels * self.expansion, momentum=0.1)
                                   )
        self.downsample = downsample
        self.stride = stride
        '''self.use_cbam = use_cbam

        if self.use_cbam:
            self.cbam = CBAM(in_channels=out_channels)'''

    def forward(self, x):
        residual = x
        y = self.conv1(x)

        '''if self.use_cbam:
            y = self.cbam(y)'''

        if self.downsample is not None:
            residual = self.downsample(x)

        return nn.functional.relu(y + residual)


# 残差块结构
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=True):
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels, momentum=0.1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels, momentum=0.1)
                                  )
        self.downsample = downsample
        self.stride = stride
        '''self.use_cbam = use_cbam

        if self.use_cbam:
            self.cbam = CBAM(in_channels=out_channels)'''

    def forward(self, x):
        residual = x
        y = self.conv(x)

        '''if self.use_cbam:
            y = self.cbam(y)'''

        if self.downsample is not None:
            residual = self.downsample(x)

        return nn.functional.relu(y + residual)


# 定义一个全局平均池化层
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])  # 池化窗口形状等于输入图像的形状


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, in_channels, out_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, in_channels, out_channels)

        self.in_channels = in_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, out_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    # 检查参数是否符合要求
    def _check_branches(self, num_branches, blocks, num_blocks, in_channels, out_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(out_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(out_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(in_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 构建一个分支branch
    def _make_one_branch(self, branch_index, block, num_blocks, out_channels, stride=1):
        downsample = None
        # 如果通道变大(分辨率变小)，则使用1x1卷积进行下采样
        if stride != 1 or self.in_channels[branch_index] != out_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], out_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels[branch_index] * block.expansion, momentum=0.1),
                                       )

        layers = []
        # 每个分支重复num_blocks[branch_index]个block，其中第一个block进行通道数转换
        layers.append(block(self.in_channels[branch_index], out_channels[branch_index], stride, downsample))
        self.in_channels[branch_index] = out_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], out_channels[branch_index]))

        return nn.Sequential(*layers)

    # 创建多个分支branches
    def _make_branches(self, num_branches, block, num_blocks, out_channels):
        branches = []

        # 通过循环构建多分支，每个分支属于不同的分辨率
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, out_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:  # 使用1x1卷积进行(j-i)次2倍上采样，使用最近邻插值，从而能够与另一分支的feature map进行相加
                    fuse_layer.append(nn.Sequential(nn.Conv2d(in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                                                    nn.BatchNorm2d(in_channels[i], momentum=0.1),
                                                    nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                                                    )
                                      )
                elif j == i:  # 同一分支不做任何操作
                    fuse_layer.append(None)
                else:  # 使用strided 3x3卷积进行下采样。如果跨两层，则使用两倍的strided 3x3卷积。通过学习的方式，降低信息损失。
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            out_channels_conv3x3 = in_channels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(in_channels[j], out_channels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False),
                                                          nn.BatchNorm2d(out_channels_conv3x3, momentum=0.1)
                                                          )
                                            )
                        else:
                            out_channels_conv3x3 = in_channels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(in_channels[j], out_channels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False),
                                                          nn.BatchNorm2d(out_channels_conv3x3, momentum=0.1),
                                                          nn.ReLU(False)
                                                          )
                                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.in_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])  # 每一个分支branch中包含了num_blocks个block

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRnet(nn.Module):
    def __init__(self):
        super(HRnet, self).__init__()
        self.start = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64, momentum=0.1),
                                   nn.ReLU(inplace=True),
                                   )
        self.stage1 = self._make_layer(Bottleneck, 64, 64, 4)
        self.transition1 = self._make_transition_layer([256], [32, 64])

        self.stage2, pre_stage2_channels = self._make_stage(BasicBlock, num_modules=1, num_branches=2, num_blocks=[4, 4], in_channels=[32, 64], out_channels=[32, 64], fuse_method='SUM')
        self.transition2 = self._make_transition_layer(pre_stage2_channels, [32, 64, 128])

        self.stage3, pre_stage3_channels = self._make_stage(BasicBlock, num_modules=1, num_branches=3, num_blocks=[4, 4, 4], in_channels=[32, 64, 128], out_channels=[32, 64, 128], fuse_method='SUM')
        self.transition3 = self._make_transition_layer(pre_stage3_channels, [32, 64, 128, 256])

        self.stage4, pre_stage4_channels = self._make_stage(BasicBlock, num_modules=1, num_branches=4, num_blocks=[4, 4, 4, 4], in_channels=[32, 64, 128, 256], out_channels=[32, 64, 128, 256], fuse_method='SUM', multi_scale_output=True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage4_channels)

        self.classifier = nn.Linear(2048, 32)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(block=head_block, in_channels=channels, out_channels=head_channels[i], num_blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                                            nn.BatchNorm2d(out_channels, momentum=0.1),
                                            nn.ReLU(inplace=True)
                                            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(nn.Conv2d(in_channels=head_channels[3] * head_block.expansion, out_channels=2048, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(2048, momentum=0.1),
                                    nn.ReLU(inplace=True)
                                    )

        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:  # 如果输入输出的通道数不匹配，则将输入通道数进行转换，图像尺寸保持不变
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels * block.expansion, momentum=0.1)
                                       )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))  # 第一个block增加通道数
        in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):  # 其余block通道数保持不变
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        '''
        :param num_channels_pre_layer: 各个分支的输入通道数（以列表形式存储）
        :param num_channels_cur_layer: 各个分支的输出通道数（以列表形式存储）
        :return:
        '''

        num_branches_pre = len(num_channels_pre_layer)  # 输入分支数
        num_branches_cur = len(num_channels_cur_layer)  # 输出分支数

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 如果输入和输出通道数不同，则进行通道数转换，否则不做任何操作
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False),
                                                           nn.BatchNorm2d(num_channels_cur_layer[i], momentum=0.1),
                                                           nn.ReLU(inplace=True)
                                                           )
                                             )
                else:
                    transition_layers.append(None)
            else:  # i >= num_branches_pre
                conv3x3s = []
                for j in range(i+1-num_branches_pre):  # 使用stride 3x3卷积生成下采样2倍的分支
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i-num_branches_pre else in_channels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                                  nn.BatchNorm2d(out_channels, momentum=0.1),
                                                  nn.ReLU(inplace=True)
                                                  )
                                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, block, num_modules, num_branches, num_blocks, in_channels, out_channels, fuse_method, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(HighResolutionModule(num_branches, block, num_blocks, in_channels, out_channels, fuse_method, reset_multi_scale_output))
            in_channels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.start(x)

        # Stage 1
        x = self.stage1(x)

        x_list = []
        for i in range(2):  # 遍历每个分支branch
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # Stage 2
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):  # 遍历每个分支branch
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage 3
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(4):  # 遍历每个分支branch
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        y = nn.functional.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y


# 通道注意力模块
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':  # 平均池化，池化窗口大小与输入图像大小相同
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':  # 最大池化，池化窗口大小与输入图像大小相同
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)  # 将平均池化和最大池化的输出分别输入到MLP中，得到的结果进行相加
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output  # 将输入F和通道注意力模块的输出Mc相乘，得到F'


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
                                               nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
                                               )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out


def model_feature(model, x):
    x = model.start(x)

    # Stage 1
    x = model.stage1(x)

    x_list = []
    for i in range(2):  # 遍历每个分支branch
        if model.transition1[i] is not None:
            x_list.append(model.transition1[i](x))
        else:
            x_list.append(x)

    # Stage 2
    y_list = model.stage2(x_list)

    x_list = []
    for i in range(3):  # 遍历每个分支branch
        if model.transition2[i] is not None:
            x_list.append(model.transition2[i](y_list[-1]))
        else:
            x_list.append(y_list[i])

    # Stage 3
    y_list = model.stage3(x_list)

    x_list = []
    for i in range(4):  # 遍历每个分支branch
        if model.transition3[i] is not None:
            x_list.append(model.transition3[i](y_list[-1]))
        else:
            x_list.append(y_list[i])
    y_list = model.stage4(x_list)

    # Classification Head
    y = model.incre_modules[0](y_list[0])
    for i in range(len(model.downsamp_modules)):
        y = model.incre_modules[i + 1](y_list[i + 1]) + model.downsamp_modules[i](y)

    y = model.final_layer(y)

    return y


def test_data_process(test_data_path):
    test_data = ImageFolder(test_data_path, transform=transform)
    test_data_loader = Data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
    class_label = test_data.classes  # 训练集的标签

    return test_data_loader, class_label


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
            print(test_data_y, pre_lab)
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


if __name__ == '__main__':
    '''model = torch.load("HRnet.pkl")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader, class_label = test_data_process(load_path)  # 加载测试集
    test_model(model, test_loader, class_label, device)'''

    classlist = os.listdir(load_path)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        all_imgs = []
        if not os.path.exists(os.path.join(save_path, leaf_class)):
            os.mkdir(os.path.join(save_path, leaf_class))
        
        imglist = os.listdir(os.path.join(load_path, leaf_class))
        for img_class in imglist:  # 读取每个类别文件夹中的图片
            path = os.path.join(load_path, leaf_class, img_class)
            '''img = Image.open(path).convert('RGB')
            img = img.rotate(50, expand=True)'''
            img = cv2.imread(path)  # 读取图片
            height, width, _ = img.shape  # 读取输入图片的尺寸
            if rotate is True:
                center = (width // 2, height // 2)
                angle = random.choice(fixed_angle)
                M = cv2.getRotationMatrix2D(center, angle, 1)  # 旋转图像
                img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.INTER_LINEAR, borderValue=cv2.BORDER_REPLICATE)  # 根据边缘填充图像
            img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Opencv转PIL
            img_tensor = transform(img_PIL).unsqueeze(dim=0)

            # 加载模型并进行预测
            if cbam is True:
                model = torch.load("HRnet_CBAM.pkl")
            else:
                model = torch.load("HRnet.pkl")
            device = torch.device("cuda")
            img_tensor = img_tensor.to(device)
            model = model.to(device)
            model.eval()
            pred = model(img_tensor)
            prob_pred = nn.functional.softmax(pred, dim=1).data.squeeze()
            pred_class = torch.argmax(prob_pred).cpu().item()

            # 提取特征图和权重并进行CAM计算
            features = model_feature(model, img_tensor).detach().cpu().numpy()
            class_weights = model.state_dict()['classifier.weight']
            batch_size, features_channel, features_height, features_width = features.shape
            features = features.reshape((features_channel, features_height * features_width))
            cam = np.dot(class_weights[pred_class].cpu().numpy(), features)  # 特征图与权重进行点积运算
            cam = cam.reshape(features_height, features_width)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # 将cam的值缩减到0-1之间
            cam_img = np.uint8(255 * cam_img)  # 将cam的值放大到0-255

            # CAM类激活热力图与原图像融合并保存和显示
            '''img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))'''
            heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
            result = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)  # CAM热力图与原图像叠加
            text = '%s: %.2f%%' % (class_name[pred_class], prob_pred[pred_class] * 100)
            cv2.putText(result, text, (25, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.8, color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
            if cbam is True:  # 使用CBAM模块
                if rotate is True:  # 对输入图像进行旋转
                    file_name = os.path.splitext(img_class)[0]
                    file_extension = os.path.splitext(img_class)[1]
                    file_rename = file_name + str('_') + str(angle) + str('_CBAM') + file_extension
                else:  # 不对输入图像进行旋转
                    file_name = os.path.splitext(img_class)[0]
                    file_extension = os.path.splitext(img_class)[1]
                    file_rename = file_name + str('_CBAM')+ file_extension
            else:  # 不使用CBAM模块
                if rotate is True:  # 对输入图像进行旋转
                    file_name = os.path.splitext(img_class)[0]
                    file_extension = os.path.splitext(img_class)[1]
                    file_rename = file_name + str('_') + str(angle) + file_extension
                else:  # 不对输入图像进行旋转
                    file_rename = img_class
            cv2.imwrite(os.path.join(save_path, leaf_class, file_rename), result)  # 保存图片
            '''cv2.namedWindow('image', 0)
            cv2.resizeWindow('image', (width, height))
            cv2.moveWindow("image", 450, 100)
            cv2.imshow("image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''



