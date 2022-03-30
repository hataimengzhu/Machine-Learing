#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import time
class Apriori:
    def __init__(self,lists,minSupport,minConf):#set:嵌套列表
        self.X = lists
        self.MS = minSupport#最小支持度/先验概率
        self.MC = minConf#最小置信度
        self.supSet = {}#支持度集合
        self.freDic = {}#频繁项集集合
    # 生成组合
    def combination(self,a,n,new,b=[],idx=0):
        if idx == len(a):
            if len(b)==n:
                new+=[b]
        else:
            c=b[:]
            #判断是否为列表
            # if isinstance(a[idx],list):
            b += [a[idx]]
            self.combination(a,n,new,b,idx+1)
            self.combination(a,n,new,c,idx+1)
    # 计算支持度(分子)、减枝
    def support(self,c):#c:候选集。返回频繁集
        counts = np.zeros(len(c))#初始化支持度
        for i in range(len(c)):#
            for j in self.X:#遍历数据集
                if set(c[i])<=set(j):#判断是否为子集
                    counts[i] += 1#计数
        #减枝，提取大于等于最低支持度的项集
        index = np.nonzero(counts/len(self.X)>=self.MS)
        return np.array(c)[index].tolist(),np.array(counts)[index].tolist()
    # 挖掘关联规则
    def rules(self):
        # 初始化关联规则字典-置信度
        rule = {}
        for k, sets in self.freDic.items():
            if k > 1:
                rule[k] = []
                for value in sets:
                    comb = [tuple([i]) for i in value]
                    # if k==2:#A->B,B->A
                    #     rule,_ = self.calConf(value,comb,rule,k)
                    # elif k>2:#合并规则
                    self.merge(value, comb, rule, k)
        return rule
    # 计算置信度、减枝
    def calConf(self, freSet, zuhe, ru, n):  # freSet:频繁项集，comb：后件，ru:规则，n:第n层
        Pab = self.supSet[frozenset(freSet)]  # 联合概率(分子)
        afterList = []
        for zu in zuhe:  # 遍历频繁集每一个元素作为后件
            antecedent = tuple(set(freSet) - set(zu))  # 前件
            afterparts = tuple(set(zu))  # 后件
            conf = Pab / self.supSet[frozenset(antecedent)]  # P(B|A)=P(A,B)/P(A)
            lift = conf / (self.supSet[frozenset(afterparts)] / len(self.X))  # 提升度
            if conf >= self.MC:  # 减枝，保留满足最小置信度的规则
                ru[n] += [[antecedent, afterparts, np.round(conf, 2), np.round(lift, 2)]]  # [前件，后件，置信度]
                afterList += [afterparts]
        return ru, afterList
    #合并规则
    def merge(self, freSet, zuhe, ru, n):  # zuhe的作用就是告诉每个组合的元素个数
        m = len(zuhe[0])  # 后件个数
        if len(freSet) == m + 1:
            new_ru, _ = self.calConf(freSet, zuhe, ru, n)
        elif len(freSet) > m + 1:  # 判断频繁项集freSet是否大到可以移除大小为m的子集
            # 递归拆分组合频繁集的子集(后件)，每次递归：每个组合的元素个数为m+1
            subSetIndex = np.nonzero([set(i) < set(freSet) for i in self.freDic[m + 1]])[0]  # 频繁项集的子集的索引
            # 从频繁集字典里选freSet的子集(每个子集m+1各元素)，无需重新计算满足置信度要求的组合
            subFreSets = np.array(self.freDic[m + 1])[subSetIndex].tolist()
            # 筛选满足置信度要求的后件
            new_ru, subFreSets = self.calConf(freSet, subFreSets, ru, n)
            if len(subFreSets[0]) <=len(freSet)-1:  # 判断能否继续拆分组合，后件每个组合元素个数<=len(freSet)-1次才能组合
                self.merge(freSet, subFreSets, new_ru, n)
    def train(self):
        #初始化候选集:candidate set
        candidate_set = list(set(sum(self.X,[])))#候选集初始化：去重
        k = 0#元素个数
        #寻找频繁项集-支持度
        while True:
            k += 1
            #生成候选集:组合
            C = []
            self.combination(candidate_set,k,C,[])#combination(a,n,new,b=[],idx=0)
            #计算所有项集支持度,减枝低于最小支持度的项集，生成频繁集
            frequents,counts= self.support(C)
            # 判断是否可以继续数据连接
            if len(frequents)<1:
                break
            #更新数据:频繁项集字典，频繁项频数
            # tuples = [tuple(l) for l in frequents]
            frozensets = [frozenset(l) for l in frequents]
            self.freDic[k] = frequents
            # self.supSet.update(dict(zip(tuples,counts)))
            self.supSet.update(dict(zip(frozensets, counts)))
            #若还可以继续数据连接，则单一化候选集元素，在减枝后的元素(频繁集)基础上，继续组合
            candidate_set = list(set(sum(frequents,[])))#若子集为非频繁集，则其超集也为非频繁集。
        #寻找关联规则-置信度、提升度
        rules = self.rules()
        print(rules)

if __name__=='__main__':
    #加载训练集
    # with open('./Data/kosarak.dat') as file:
    #     baseData = [list(map(int,line.split()))for line in file.readlines()]
    dataSet = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    ds = [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]
    simDat = [['r', 'z', 'h', 'j', 'p'],
              ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
              ['z'],
              ['r', 'x', 'n', 'o', 's'],
              ['y', 'r', 'x', 'z', 'q', 't', 'p'],
              ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    dataSet2 = [['e', 'c', 'b'], ['d', 'c', 'a'], ['e', 'd', 'c', 'a'], ['d', 'a'], ['e', 'd']]
    #数据预处理
    #训练
    start = time.time()  # 计时开始
    min_support = 0.1#最小支持度/先验概率
    min_conf = 0.2#最小置信度/条件概率
    xx = Apriori(dataSet2,min_support,min_conf)
    xx.train()
    end = time.time()  # 计时结束
    print("耗时：{}".format(end - start))

# from apyori import apriori
# import pandas as pd
# result = list(apriori(transactions=dataSet,min_support=0.2,min_confidence=0.5))
# result_pd = pd.DataFrame(result)
# result_pd.to_csv('123.csv')