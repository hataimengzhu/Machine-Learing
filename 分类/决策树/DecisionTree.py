#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
from math import log
import copy
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def createDataSet():
    dataSet = [[0, 0, 0, 0],  # 数据集
               [0, 0, 0, 1],
               [0, 1, 0, 1],
               [0, 1, 1, 0],
               [0, 0, 0, 0],
               [1, 0, 0, 0],
               [1, 0, 0, 1],
               [1, 1, 1, 1],
               [1, 0, 1, 2],
               [1, 0, 1, 2],
               [2, 0, 1, 2],
               [2, 0, 1, 1],
               [2, 1, 0, 1],
               [2, 1, 0, 2],
               [2, 0, 0, 0,]]
    tab = ['年龄', '有工作', '有自己的房子', '信贷情况']#特征标签
    labels = ['no','no','yes','yes','no','no','no','yes','yes','yes','yes','yes','yes','yes','no']#类别
    dataSet = np.array(dataSet)
    return dataSet, tab, labels  # 返回数据集和分类属性

def loadData(fileName,delim='\t'):#假设原始数：第一行为特征标签+类别名称，最后一列为类别
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()#包含换行符、制表符，每个元素为字符串
    #数据清洗：去换行符、通过制表符分割为列表
    tab = baseData[0].strip().split('\t')
    #删除表头中最后一个元素：类别名称
    tab.pop()
    #删除表头
    baseData.pop(0)
    # 数据清洗：去换行符、通过制表符分割为列表
    baseData = [da.strip().split(delim) for da in baseData]
    baseData = np.array(baseData)
    #存储类别
    labels = baseData[:,-1].tolist()
    # 删除类别列
    dataSet = np.delete(baseData,-1,axis=1)
    return dataSet,tab,labels

class DcTree:
    def __init__(self, features, tab, labels):  # features：特征值，m*n数组,tab:特征标签,1维单层列表,labels:类别，1维单层列表
        self.X, self.Y = features, labels
        self.Tab = tab  # 训练集特征便签 n*1
        self.Tree = None #决策树
    # 特征选择：1、划分子集，2、计算熵，3、计算信息增益
    # 按照第n个特征的特征值划分子集,返回子集对应的类别
    @staticmethod
    def splitDataSet(subSet, subTab, subLabelS, n, x):
        # 划分子集对应的类别->筛选包含特征值x对应的类别
        subFeatureList = []#用于存储第n列特征的特征值x对应的索引
        for i in range(subSet.shape[0]):  # 按特征值x对应的索引值
            if subSet[i, n] == x:
                subFeatureList.append(i)#存储第n列特征的特征值x对应的索引
        tmp = np.array(subLabelS)
        subLabels = tmp[subFeatureList]#筛选出特征值x对应的索引位置的类别
        subLabels = subLabels.tolist()
        # 划分特征标签->删除第n个标签
        subTabTmp = copy.deepcopy(subTab)
        subTabTmp.pop(n)
        # 划分子集->删除第n列特征+筛选包含特征值x对应的特征
        tmp = np.delete(subSet,n,axis=1)
        subSet = tmp[subFeatureList, :]
        return subSet, subTabTmp, subLabels
    # 计算熵
    @staticmethod
    def entropy(subLabelS):
        subLabel = set(subLabelS)
        shannonEnt = 0.0
        for da in subLabel:
            pr = subLabelS.count(da) / len(subLabelS)  # 子集的先验概率
            shannonEnt += pr * np.log2(1 / pr)
        return shannonEnt
    # 计算信息增益->特征选择
    def selectFeature(self, subSet, subTab, subLabelS):
        baseEntropy = self.entropy(subLabelS)  # 计算当前子集的信息熵
        infoGain, maxInfoGain, BestFeatureIndex= 0.0, 0.0, None
        for n in range(len(subTab)):  # 遍历子集的特征标签
            subFeature = set(subSet[:, n])#创建set集合{},元素不可重复
            conEntropy = 0.0#经条件熵
            for da in subFeature:  # 遍历第n个特征的特征值，计算第n个特征的条件熵
                _, _, subLabels = self.splitDataSet(subSet, subTab, subLabelS, n, da)  # 划分子集
                pr = len(subLabels) / len(subLabelS)  # 子集的先验概率
                conEntropy += pr*self.entropy(subLabels)  # 条件熵=sum(子集的先验概率*子集的熵)
            infoGain = baseEntropy - conEntropy  # 计算第n个特征的信息增益
            # 选择信息增益最大的特征划分
            if infoGain > maxInfoGain:
                maxInfoGain = infoGain#更新信息增益，找到最大的信息增益
                BestFeatureIndex = n#信息增益最大的特征的索引值
        return BestFeatureIndex#返回信息增益最大的特征的索引值
    # 训练->生成决策树
    def train(self):
        def fun(subSet, subTab, subLabelS):
            if len(set(subLabelS)) == 1: # 若划分的标签子集的类别都一样，即具备唯一类别，则不用继续划分了
                return subLabelS[0]#直接返回分类结果
            if len(set(subTab)) == 1:# 遍历到最后一个特征时，划分的数据(是否贷款)依然不是唯一(如：都是"是"或都是"否")，已没有新的特征可以进行划分，则按经验选取对应最多的类别(是/否)，作为唯一类别
                return Counter(subLabelS).most_common()[0][0]
            BestFeatureIndex = self.selectFeature(subSet, subTab, subLabelS)
            BestTab = subTab[BestFeatureIndex]
            prTree = {BestTab: {}}# {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
            featureValues = set(subSet[:, BestFeatureIndex])  # 第n列特征值
            for da in featureValues:
                subsubSet, subsubTab, subsubLabels = self.splitDataSet(subSet, subTab, subLabelS, BestFeatureIndex, da)
                prTree[BestTab][da] = fun(subsubSet, subsubTab, subsubLabels)
            return prTree
        self.Tree = fun(self.X, self.Tab, self.Y)
    def predict(self,inX,inTree):#遍历决策树，判断键值
        # {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
        firstNode = next(iter(inTree))#inTree字典转换成迭代器，使用next函数访问迭代器对象中的元素
        nextDic = inTree[firstNode]
        featureIndex = self.Tab.index(firstNode)#获取叶子节点对应的索引位置
        for key in nextDic.keys():
            if inX[featureIndex]==key:
                if type(nextDic[key]).__name__=='dict':# inTree={key:{}}
                    predict = self.predict(inX,nextDic[key])
                else:
                    predict = nextDic[key]
        return predict

if __name__ == '__main__':
    DataSet,Tabs, Labels = createDataSet()
    fileName = 'lenses.txt'
    # DataSet, Tabs, Labels =loadData(r'./Data/' + fileName)
    dcTree = DcTree(DataSet,Tabs, Labels)
    dcTree.train()
    print(dcTree.Tree)
    # testData = np.array(['young','hyper','no','normal'])
    testData = np.array([0,0,0,1])
    print('类别：',dcTree.predict(testData,dcTree.Tree))







