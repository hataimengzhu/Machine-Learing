# -*- coding: utf-8 -*-
import numpy as np
import operator

import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def textToMatrix(path):
    file = open(path)
    datas = file.readlines() #全部读取
    dataMatrix = np.zeros((len(datas), 3)) #初始化数据为m行3列（飞行里程，游戏时间，冰淇淋数）
    #解析文件数据到列表
    labelsList = []
    i = 0
    for da in datas:
        da = da.strip() #去除字符串尾回车符
        da = da.split('\t') #按制表符分割字符串为列表  制表符作用：垂直对齐
        dataMatrix[i] = da[0:3]  #特征矩阵
        labelsList.append(int(da[-1])) #类列表
        i += 1
    return dataMatrix, labelsList

def dataNorm(dataSet):
    minValues = dataSet.min(0) #获取每列的最小值
    maxValues = dataSet.max(0)  # 获取每列的最大值
    normDataSet = np.zeros(dataSet.shape)
    h = dataSet.shape[0]
    scope = maxValues - minValues
    normDataSet = dataSet - np.tile(minValues,(h, 1))
    normDataSet = normDataSet/ scope
    return normDataSet, scope, minValues

def draw(dataMatrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMatrix[:,0],dataMatrix[:,1],15*np.array(labelsList), labelsList)

    plt.title('样本散点图')
    plt.xlabel('玩视频游戏所耗时间百分比')
    plt.ylabel('每周消费的冰激凌公升数')
    labels=['不喜欢', '魅力一般', '极具魅力']
    plt.legend(labels, loc=0, ncol=2)
    plt.show()

def classifyKNN(inData, dataSet, labels, k):
    rows = dataSet.shape[0]
    distance = np.sum((np.tile(inData, (rows, 1)) - dataSet)**2, axis=1)**0.5
    disIndex = distance.argsort() #获得升序 索引
    classCount = {}
    # 记录距离最小的前K个类，并存放入列表。KEY对应标签，VALUE对应计数
    for d in range(k):
        label = labels[disIndex[d]]
        classCount[label] = classCount.get(label, 0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #降序，返回降序序列[(key1,num1),(key2,num2)]
    return sortedClassCount[0][0]

def classifyTest(path, radio, k):
    dataMatrix, labelsList = textToMatrix(path)
    normDataSet, scope, minValues = dataNorm(dataMatrix)
    row = normDataSet.shape[0]
    testNum = int(row*radio)
    errorCount = 0
    for i in range(testNum):
        result = classifyKNN(normDataSet[i], normDataSet[testNum:row,:], labelsList[testNum:row], k)
        if result != labelsList[i]:
            errorCount += 1
        errorRadio = errorCount/testNum
    return errorRadio

def classify():
    # 样本数据
    lablesMapping = ['不喜欢', '魅力一般', '极具魅力']
    filePath = r'.\Data\ch01-KNN\datingTestSet2.txt'
    dataMatrix, labelsList= textToMatrix(filePath)
    normData, scope, minValues = dataNorm(dataMatrix)
    # 测试数据
    flightRadio = float(input('每年获得的飞行常客里程数?'))
    gameRadio = float(input('玩视频游戏所耗时间百分比?'))
    iceRadio = float(input('每周消费的冰激凌数?'))
    inputDatas = np.array([flightRadio, gameRadio, iceRadio])


    # 测试数据:数据清洗：归一化
    inputDatasNorm = (inputDatas-minValues)/scope

    # KNN分类
    result = classifyKNN(inputDatasNorm, normData, labelsList,3)
    resultClass = lablesMapping[result-1]
    print('喜欢吗？', resultClass)

    return resultClass



if __name__=='__main__':
    # group, labels = creatDataSet()
    # result = classifyKNN([-1, -1], group, labels, 4)

    # filePath = r'.\Data\ch01-KNN\datingTestSet2.txt'
    # dataMatrix, labelsList= textToMatrix(filePath)
    # draw(dataMatrix)

    # filePath = r'.\Data\ch01-KNN\datingTestSet2.txt'
    # dataMatrix, labelsList = textToMatrix(filePath)
    # print(dataMatrix[:5])
    # normData, scope, minValues = dataNorm(dataMatrix)
    # print(normData[:5])

    # filePath = r'.\Data\ch01-KNN\datingTestSet2.txt'
    # radio = 0.1
    # k = 5
    # errorRadio = classifyTest(filePath, radio, k)
    # print('ErrorRadio:{:.1%}'.format(errorRadio))

    classify()


