Machine Learning
===

# 1. Leaf_data_acquisition.py
## 1.1 介绍说明
提取叶片特征，并保存为csv文件，为后续使用机器学习算法进行分类做准备。
## 1.2 图像预处理
1. histogram(image)函数的作用是获取输入图片R、G、B三通道的像素值分布情况。
2. binarization(imgray)函数的作用是分别使用2x2、3x3、5x5的卷积核对输入的灰度图进行平均滤波和二值化处理。
3. margin_detection(imgbi)函数的作用是使用拉普拉斯算子提取图像边缘特征。
## 1.3 特征提取
1. feature5_extraction(imgray, thresh_5x5, thresh_3x3, thresh_2x2)函数的作用是提取5种几何特征，即最小外接圆直径、最小外接矩形的宽度和高度、不同卷积核平均滤波后的叶片面积（2x2、3x3、5x5）、3x3卷积核平均滤波后的叶片周长。
2. feature12_extraction(thresh, feature)函数的作用是根据上面提取到的5种几何特征，获取12种数字形态特征，即平滑因子、纵横比、形状因子、矩形程度、狭窄因子、直径周长比、周长与长宽比、5种静脉特征。
## 1.4 数据降维
data_PCA(img_data)函数的作用是将12种数字形态特征降维到5维。


# 2. Leaf_classification_ML.py
## 2.1 介绍说明
根据叶片特征，使用不同的机器学习算法对叶片进行分类。
## 2.2 数据预处理
1. encode(train, test)函数的作用是对训练集和测试集中的数据进行编码，以及其他预处理操作。
2. deta_acquisition()函数的作用是对训练集按4：1的比例划分为训练集和测试集（验证集）。
## 2.3 叶片分类
ML_classifier(X_train, X_test, y_train, y_test)函数的作用是使用不同的机器学习算法对叶片进行分类，并显示分类准确率和损失函数，其中列表**classifiers**中包含了所使用的机器学习算法。
## 2.4 实验结果
<center>
![image](https://user-images.githubusercontent.com/39607836/149734571-bf3b4373-5c93-4648-b11d-94cad8decc5b.png)  
</center>
<center>
![image](https://user-images.githubusercontent.com/39607836/149734641-ddd226c4-2ee4-402a-9ed8-bfd5d9b7226f.png)
</center>
