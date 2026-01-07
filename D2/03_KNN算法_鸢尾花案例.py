"""
案例：通过KNN算法实现 鸢尾花的 分类操作。
回顾：机器学习项目的研发流程
1．加载数据．
2．数据的预处理．
3．特征工程(提取，预处理…)
4．模型训练．
5．模型评估．
6．模型预测．
"""
# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花测试集的．
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估的，计算模型预测的准确率


# 1.定义函数，加载鸢尾花数据集，并查看数据集
def dm01_load_iris():
    # 1.加载鸢尾花数据集.
    iris_data = load_iris()
    # 2.查看数据集
    # print(f'数据集：{iris_data}')  # 字典形态
    # print(f'数据集的类型：{type(iris_data)}')  # <class 'sklearn.utils._bunch.Bunch'>
    # 3.查看数据集所有的键
    print(f'数据集所有的键:{iris_data.keys()}')
    # 4.查看数据集的键对应的值
    # print(f'具体的数据:{iris_data.data[:5]}')  # 有150条数据，每条四个特征，我们只看前五条
    print(f'具体的数据:{iris_data.data}')
    # print(f'具体的标签:{iris_data.target[:5]}')  # 有150条数据，每条一个标签，我们只看前五条
    print(f'具体的标签:{iris_data.target}')
    print(f'标签对应的名称:{iris_data.target_names}')  # ['setosa' 'versicolor' 'virginica']
    print(
        f'特征对应的名称:{iris_data.feature_names}')  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # print(f'数据集的描述:{iris_data.DESCR}')
    # print(f'数据集的框架:{iris_data.frame}')  # 数据集的框架:None
    print(f'数据集的文件名:{iris_data.filename}')  # 数据集的文件名:iris.csv
    print(f'数据集的形状:{iris_data.data.shape}')  # (150, 4)
    print(f'数据集的模型（在哪个包下）:{iris_data.data_module}')  # sklearn.datasets.data

    # 2.定义函数，绘制数据集的散点图.


def dm02_show_iris():
    # 1.加载数据集
    iris_data = load_iris()
    # 2.把鸢尾花数据集封装成 DataFrame 对象. （因为cable可视化需要dataframe对象才能可视化
    # iris_df=pd.DataFrame(iris_data.data)
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # print(f'具体的数据:{iris_data.data}')
    # 3.给df对象新增一列->标签列
    iris_df['label']=iris_data.target
    print(iris_df)


# 3.

# 4.
# 5.测试
if __name__ == '__main__':
    dm01_load_iris()
    dm02_show_iris()
