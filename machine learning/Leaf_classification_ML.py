import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


train_data = pd.read_csv('C:/Users/HASEE/Desktop/Pycharm_Project/train.csv')
test_data = pd.read_csv('C:/Users/HASEE/Desktop/Pycharm_Project/test.csv')

def encode(train, test):
    le = LabelEncoder().fit(train.class_id)  # 对数据进行标签编码
    labels = le.transform(train.class_id)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.image_id  # 保存测试集id

    train = train.drop(['image_id', 'class_id'], axis=1)  # 删除列
    test = test.drop(['image_id', 'class_id'], axis=1)  # 删除列

    return train, labels, test, test_ids, classes


def deta_acquisition():
    train, labels, test, test_ids, classes = encode(train_data, test_data)

    ss = StratifiedShuffleSplit(n_splits=2, test_size=0.25,
                                random_state=42)  # train/test对的组数，train/test对中train和test所占的比例，随机数种子
    for train_index, test_index in ss.split(train, labels):
        X_train, X_test = train.values[train_index], train.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    return X_train, X_test, y_train, y_test


def ML_classifier(X_train, X_test, y_train, y_test):
    classifiers = [KNeighborsClassifier(3),  # K近邻算法
                   SVC(kernel="rbf", C=0.025, probability=True),  # 支持向量机分类
                   NuSVC(probability=True),  # 核支持向量机分类
                   DecisionTreeClassifier(),  # 决策树
                   RandomForestClassifier(),  # 随机森林
                   AdaBoostClassifier(),  # AdaBoost分类
                   GradientBoostingClassifier(),  # 梯度提升决策树
                   GaussianNB(),  # 朴素贝叶斯
                   LinearDiscriminantAnalysis(),  # 线性判别分析
                   QuadraticDiscriminantAnalysis()  # 二次判别分析
                   ]

    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

    print("=" * 30)
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    sns.set_color_codes("muted")
    sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = deta_acquisition()
    ML_classifier(X_train, X_test, y_train, y_test)
