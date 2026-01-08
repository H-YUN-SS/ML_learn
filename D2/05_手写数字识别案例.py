"""
案例: 演示 KNN算法 识别图片, 即: 手写数字识别案例。
介绍:
每张图片都是由 28 * 28 像素组成的, 即: 我们的csv文件中每一行都有 784 个像素点, 表示图片(每个像素)的 颜色。
最终构成图像。
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter


# 1.定义函数，接收用户传入的索引，展示该索引对应的图片
def show_digit(idx):
    # 1.读取数据集，获取到源数据
    df = pd.read_csv('./data/手写数字识别.csv')
    # print(df)#[42000 rows x 785 columns]

    # 2.判断传入的索引是否越界.
    if idx < 0 or idx > len(df) - 1:
        print("索引越界！")
        return

    # 3.说明没有越界，正常获取数据
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # 4.查看用户传入的索引对应的图片是几
    print(f'该图片对应的数字是:{y.iloc[idx]}')
    print(f'查看所有的标签的分布情况{Counter(y)}')

    # 5.查看下用户传入的索引对应的图片的形状
    print(x.iloc[idx].shape)  # (784,)  要想办法把(784,)转换成(28,28)
    # print(x.iloc[idx].values)#具体的784个像素点的数据
    # 6 把(784,)转换成(28,28)
    x = x.iloc[idx].values.reshape(28, 28)
    # print(x) #28*28像素点
    # 7.具体的绘制灰度图的动作
    plt.imshow(x, cmap='grey')  # 灰度图
    plt.axis('off')
    plt.show()


# 2.


# 3.


# 4.测试
if __name__ == '__main__':
    show_digit(23)
