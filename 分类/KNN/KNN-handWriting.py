# -*- coding: utf-8 -*-
import numpy as np
import operator
import os
from PIL import Image
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from    pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def loadTextData(path, featureNum): # path:文件路径, featureNum：特征种类
    file = open(path)
    fileDatas = file.readlines() #逐行读取
    dataMatrix = np.zeros(len(fileDatas), featureNum) # 特征矩阵容器
    labelsList = [] # 类别列表
    i = 0
    for da in fileDatas: # da: 特征1，特征2，特征3，类别
        da = da.strip() # 去除字符串尾部换行符
        da = da.split('\t')  # 字符串按制表符分割为列表
        dataMatrix[i] = da[0:featureNum] # 特征矩阵
        labelsList.append(da[-1]) # 类别列表
        i += 1
    return dataMatrix, labelsList

def loadTextImg(path):
    # 样本数据
    trainFilesList = os.listdir(path + '\\' + 'trainingDigits')
    trainFileNum = len(trainFilesList)
    trainFileLabels = []
    trainMatrix = np.zeros((trainFileNum, 1024))
    for n in range(trainFileNum):
        trainFileLabels.append(trainFilesList[n].split('_')[0])
        file = open(path + '\\' + 'trainingDigits' + '\\' + trainFilesList[n])
        for i in range(32):
            fileDatas = file.readline()
            fileDatas = fileDatas.strip()
            for j in range(32):
                trainMatrix[n,j+i*32] = fileDatas[j]

    # 测试数据
    accuracy = 0
    testFilesList = os.listdir(path + '\\' + 'testDigits')
    testFileNum = len(testFilesList)
    testFileLabels = []
    testMatrix = np.zeros((testFileNum, 1024))
    for n in range(testFileNum):
        testFileLabels.append(testFilesList[n].split('_')[0])
        file = open(path + '\\' + 'testDigits' + '\\' + testFilesList[n])
        for i in range(32):
            fileDatas = file.readline()
            fileDatas = fileDatas.strip()
            for j in range(32):
                testMatrix[n,j+i*32] = fileDatas[j]
        result = classifyKNN(testMatrix[n], trainMatrix, trainFileLabels, 3)
        print('result:', result,'right:', testFileLabels[n])
        if result == testFileLabels[n]:
            accuracy += 1
    print('ErrorRadio:{:.1%}'.format(accuracy/testFileNum))

def ImgToText(path):
    imgFilesList = os.listdir(path + '\\' + 'img')
    imgNum = len(imgFilesList)
    for i in range(imgNum):
        img = Image.open(path + '\\' + 'img' + '\\' + imgFilesList[i])
        # 转换图片为RGBA模式+32*32尺寸
        img = img.convert('RGB')
        img = img.resize((32, 32), Image.LANCZOS)
        imgSize = img.size
        # img.save('test.png')
        imgName = imgFilesList[i].split('.')[0]
        with open(path + '\\testDigits\\' + imgName + '.txt','wb') as f:
            for x in range(imgSize[0]):
                for y in range(imgSize[1]):
                    tmp = img.getpixel((y,x))
                    if(tmp != (255,255,255)):
                        if y != 0 and y % 31 ==0 and x<imgSize[0]-1:
                            f.write(b'1\n')
                        else:
                            f.write(b'1')
                    else:
                        if y != 0 and y % 31 ==0 and x<imgSize[0]-1:
                            f.write(b'0\n')
                        else:
                            f.write(b'0')

def cleanData(model, DataSet): #model: 如何清洗, DataSet：特征样本集
    if model=='NORM':
        minValues = DataSet.min(0)
        maxValues = DataSet.max(0)
        scope = maxValues - minValues
        rows = DataSet.shape[0]
        normDatas = np.zeros(DataSet.shape)
        normDatas = (DataSet - np.tile(minValues, (rows, 1)))/scope

        return normDatas, scope, minValues

def classifyKNN(inData, DataSet, Labels, k): # inData:测试数据, DataSet:特征样本集, Labels：标签样本集, k：取前K个距离集(降序)对应的标签
    rows = DataSet.shape[0]
    distance = (((np.tile(inData, (rows, 1)) - DataSet)**2).sum(axis=1))**0.5
    disSortIndex = distance.argsort()
    labelsClass = {}
    for i in range(k):
        labelsClass[Labels[disSortIndex[i]]] = labelsClass.get(Labels[disSortIndex[i]], 0)+1
    disLabelsSort = sorted(labelsClass.items(), key=operator.itemgetter(1), reverse=True)
    return disLabelsSort[0][0]

def classify():
    # 样本数据
    filePath = ''
    featureNum = 3
    dataMatrix, labelsList = loadTextData(filePath, featureNum)
    normDatas, scope, minValues = cleanData('NORM', dataMatrix)
    lablesMapping = ['不喜欢', '魅力一般', '极具魅力']
    # 测试数据
    inputDatas = np.array([])
    inputDatasNorm = (inputDatas - minValues) / scope
    # 分类
    result = classifyKNN(inputDatasNorm, normDatas, labelsList, 3)
    resultClass = lablesMapping[result-1]

    return resultClass

if __name__=='__main__':
    filePath = r'.\Data\ch01-KNN\digits'
    ImgToText(filePath)
    loadTextImg(filePath)
