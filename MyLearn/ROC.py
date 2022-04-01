#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import Edge as ed
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plotROC(predStrengths,inY,fit=False):  # predStrengths：分类器预测强度，m*1矩阵，inY：真实类别,m*1矩阵
    numPosClas = np.sum(inY == 1.0)  # 统计正类的数量
    numNegClas = len(inY) - numPosClas  # 统计负类的数量
    sortedIndex = (-predStrengths.T).argsort()  # 预测强度排序,从高到低
    numP, numN, AUCH, AUCW = 0, 0, 0, 1 / numNegClas
    coordinate = []
    for index in sortedIndex.tolist()[0]:
        if inY[index] == 1.0:  # 向上位移
            numP += 1
            TPR = numP / numPosClas
            FPR = numN / numNegClas
        else:
            numN += 1
            FPR = numN / numNegClas
            TPR = numP / numPosClas
            AUCH += TPR
        coordinate.append([FPR, TPR])
    AUC = AUCH * AUCW
    coordinate = np.mat(coordinate)
    # 凸包拟合
    plt.figure(10)
    if fit:
        # 凸包拟合锯齿边界
        xx = ed.Edge(coordinate)  # 实例化
        xx.margin()  # 获取凸边界点
        xx.boundaryPoint[1].append(xx.boundaryPoint[1].pop(0))
        plt.plot(np.mat(xx.boundaryPoint[1])[:, 0], np.mat(xx.boundaryPoint[1])[:, 1], 'r')
    else:
        plt.plot(coordinate[:, 0], coordinate[:, 1], 'r')


    plt.plot([0, 1], [0, 1], 'b--')
    plt.title('ROC曲线')
    plt.xlabel('灵敏度')
    plt.ylabel('特异度')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    print('AUC面积为:', AUC)  # 计算AUC
    plt.show()
