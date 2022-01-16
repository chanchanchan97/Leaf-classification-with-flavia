import numpy as np
import cv2
import math
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

classdir = 'C:/Users/HASEE/Desktop/Pycharm_Project/images/Leaves/'
imgdir = 'C:/Users/HASEE/Desktop/Pycharm_Project/images/Leaves/0/1001.jpg'


def histogram(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(1, 3, i+1)
        plt.plot(histr, color=col)
        plt.title(col, fontsize=12)
        plt.xlim([0, 256])
        print(col, histr.shape)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.42, hspace=None)
    plt.show()


def binarization(imgray):
    imgblur5 = cv2.blur(imgray, (5, 5))
    imgblur3 = cv2.blur(imgray, (3, 3))
    imgblur2 = cv2.blur(imgray, (2, 2))
    ret_5x5, thresh_5x5 = cv2.threshold(imgblur5, 242, 255, 1)
    ret_3x3, thresh_3x3 = cv2.threshold(imgblur3, 242, 255, 1)
    ret_2x2, thresh_2x2 = cv2.threshold(imgblur2, 242, 255, 1)

    '''plt.subplot(1, 2, 1)
    plt.imshow(imgray, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(thresh_3x3, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()'''


    return thresh_5x5, thresh_3x3, thresh_2x2


def margin_detection(imgbi):
    laplacian = cv2.Laplacian(imgbi, ddepth=-1, ksize=3)
    margin = cv2.bitwise_not(laplacian)

    return margin


def feature5_extraction(imgray, thresh_5x5, thresh_3x3, thresh_2x2):
    feature5 = []
    color = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
    # 计算图像的矩
    contours_3x3, hierarchy_3x3 = cv2.findContours(thresh_3x3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_3x3 = contours_3x3[0]
    contours_5x5, hierarchy_5x5 = cv2.findContours(thresh_5x5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_5x5 = contours_5x5[0]
    contours_2x2, hierarchy_2x2 = cv2.findContours(thresh_2x2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_2x2 = contours_2x2[0]

    # 计算最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt_3x3)
    center = (int(x), int(y))
    radius = int(radius)
    diameter = radius << 1
    img = cv2.circle(color, center, radius, (0, 255, 0), 2)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(cnt_3x3)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = rect[1][0]
    height = rect[1][1]
    img = cv2.drawContours(color, [box], 0, (255, 0, 0), 1)
    perimeter = cv2.arcLength(cnt_3x3, True)
    area_3x3 = cv2.contourArea(cnt_3x3)
    area_5x5 = cv2.contourArea(cnt_5x5)
    area_2x2 = cv2.contourArea(cnt_2x2)

    feature5 = [diameter, height, width, area_2x2, area_3x3, area_5x5, perimeter]

    return color, feature5


def feature12_extraction(thresh, feature):
    feature12 = []

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening3 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opening4 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel4)

    contours_op1, hierarchy_op1 = cv2.findContours(opening1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_op1 = contours_op1[0]
    Av1 = cv2.contourArea(cnt_op1)
    contours_op2, hierarchy_op2 = cv2.findContours(opening2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_op2 = contours_op2[0]
    Av2 = cv2.contourArea(cnt_op2)
    contours_op3, hierarchy_op3 = cv2.findContours(opening3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_op3 = contours_op3[0]
    Av3 = cv2.contourArea(cnt_op3)
    contours_op4, hierarchy_op4 = cv2.findContours(opening4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_op4 = contours_op4[0]
    Av4 = cv2.contourArea(cnt_op4)

    smooth_factor = feature[5] / feature[3]
    aspect_ratio = feature[1] / feature[2]
    form_factor = 4 * math.pi * feature[5] / pow(feature[6], 2)
    rectangularity = feature[1] * feature[2] / feature[5]
    narrow_factor = feature[0] / feature[1]
    Pr_diameter = feature[6] / feature[0]
    Pr_width_length = feature[6] / (feature[0] + feature[1])
    vein_feature1 = Av1 / feature[4]
    vein_feature2 = Av2 / feature[4]
    vein_feature3 = Av3 / feature[4]
    vein_feature4 = Av4 / feature[4]
    vein_feature5 = Av4 / Av1
    feature12 = [smooth_factor, aspect_ratio, form_factor, rectangularity, narrow_factor, Pr_diameter,
                Pr_width_length, vein_feature1, vein_feature2, vein_feature3, vein_feature4, vein_feature5]

    return feature12



def data_PCA(data):
    pca = PCA(n_components=5)
    pca.fit(data)
    data_reduction = pca.transform(data)

    return data_reduction


def img_process(img_path):
    im = cv2.imread(img_path)  # 获取图片
    #histogram(im)
    im_resize = cv2.resize(im, (600, 450))
    imgray = cv2.cvtColor(im_resize, cv2.COLOR_BGR2GRAY)  # RGB转灰度
    thresh_5x5, thresh_3x3, thresh_2x2 = binarization(imgray)  # 图像二值化
    laplacian = margin_detection(thresh_3x3)  # 拉普拉斯边缘检测
    color, f5 = feature5_extraction(imgray, thresh_5x5, thresh_3x3, thresh_2x2)  # 提取5种特征

    return thresh_3x3, f5


if __name__ == '__main__':
    img_data = []
    name = ['smooth_factor', 'aspect_ratio', 'form_factor', 'rectangularity', 'narrow_factor', 'Pr_diameter',
             'Pr_width_length', 'vein_feature1', 'vein_feature2', 'vein_feature3', 'vein_feature4', 'vein_feature5',
            'image id', 'class id']
    name_reduction = ['r1', 'r2', 'r3', 'r4', 'r5']
    classlist = os.listdir(classdir)
    for leaf_class in classlist:
        imglist = os.listdir(classdir+leaf_class)
        for img_class in imglist:
            img_id = os.path.splitext(img_class)[0]
            class_id = leaf_class
            thresh, f = img_process(classdir+leaf_class+'/'+img_class)  # 获取5种几何特征
            if f[3] == 0 or f[1] == 0 or f[0] == 0 or f[5] == 0:  # 将特征值提取有问题的叶片图像过滤掉
                continue
            f12 = feature12_extraction(thresh, f)  # 获取12种数字形态特征
            if abs(f12[0]) > 2:  # 将特征值提取有问题的叶片图像过滤掉
                continue
            f12.append(img_id)
            f12.append(class_id)
            img_data.append(f12)

    '''data_reduction = data_PCA(img_data)
    # 数据归一化
    data_reduction[:, 0] = (data_reduction[:, 0] - np.mean(data_reduction[:, 0])) / np.std(data_reduction[:, 0])
    data_reduction[:, 1] = (data_reduction[:, 1] - np.mean(data_reduction[:, 1])) / np.std(data_reduction[:, 1])
    data_reduction[:, 2] = (data_reduction[:, 2] - np.mean(data_reduction[:, 2])) / np.std(data_reduction[:, 2])
    data_reduction[:, 3] = (data_reduction[:, 3] - np.mean(data_reduction[:, 3])) / np.std(data_reduction[:, 3])
    data_reduction[:, 4] = (data_reduction[:, 4] - np.mean(data_reduction[:, 4])) / np.std(data_reduction[:, 4])
    print(np.array(data_reduction).shape)'''

    datalist = pd.DataFrame(columns=name, data=img_data)
    datalist.to_csv('C:/Users/HASEE/Desktop/Pycharm_Project/feature_r5_2.csv')
