#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import jieba
import numpy as np
def createData():
    with open(r".\Data\Chinese_from_dongxiexidian-master\stopwords.txt",'r', encoding='utf-8') as file:
        stopWords = [da.strip() for da in file.readlines()]
    # 分词、过滤stopwords
    exampleData = [['根据您所选择的上述礼遇，您必须在娱乐城完成总金额（存款+首存奖金）16倍或15倍流水之后，方可申请提款。'],
                   ['理查德·菲利普斯·费曼，美国理论物理学家，量子电动力学创始人之一，纳米技术之父。'],
                   ['奖金将在您完成首存后即可存入您的账户。真人荷官 体育博彩 欢迎来到全新番摊游戏！'],
                   ['在公安机关持续不断的打击下，六*合*彩、私彩赌博活动由最初的公开、半公开状态转入地下。']]
    exampleList = [list(set(jieba.cut(da[0], cut_all=False))) for da in exampleData] # 分词
    # 过滤stopwords+数字+符号+len(词)==1
    shujuList = []
    for da in exampleList:
        tmp = []
        for dd in da:
            if dd!=' ' and len(dd)>1 and not dd.isdigit() and dd not in stopWords:
                tmp.append(dd)
        shujuList.append(tmp)
    # 创建训练样本数据对应的类别
    tag = ['feifa','normal','feifa','normal']
    # 创建词汇列表
    tmp = []
    for da in shujuList:
        tmp.extend(set(da))
    vocabulary = list(set(tmp))
    return tag, shujuList, vocabulary
# 训练数据向量化 测试数据文字向量化，
def getVocabVec(vocabulary, sentence):
    with open(r".\Data\Chinese_from_dongxiexidian-master\stopwords.txt",'r', encoding='utf-8') as file:
        stopWords = [da.strip() for da in file.readlines()]
    # 分词、过滤stopwords
    if not any(isinstance(i, list) for i in sentence): # 判断是否为嵌套列表
        sentenceList = [list(set(jieba.cut(da, cut_all=False))) for da in sentence]  # 分词
        # 过滤stopwords+数字+符号+len(词)==1
        dataList = []
        for da in sentenceList:
            tmp = []
            for dd in da:
                if dd!=' ' and len(dd)>1 and not dd.isdigit() and dd not in stopWords:
                    tmp.append(dd)
            dataList.extend(tmp)
    else:
        sentenceList = sentence
        # 过滤stopwords+数字+符号+len(词)==1
        dataList = []
        for da in sentenceList:
            tmp = []
            for dd in da:
                if dd!=' ' and len(dd)>1 and not dd.isdigit() and dd not in stopWords:
                    tmp.append(dd)
            dataList.append(tmp)


    vocabVec = []
    if any(isinstance(i, list) for i in dataList):
        for i in range(len(dataList)):
            row = [0]*len(vocabulary)
            for j in dataList[i]:
                if j in vocabulary:
                    row[vocabulary.index(j)] = 1
            vocabVec.append(row)
    else:
        row = [0] * len(vocabulary)
        for j in dataList:
            if j in vocabulary:
                row[vocabulary.index(j)] = 1
        vocabVec.append(row)

    return vocabVec
def train(tag, vocabVec, dataList):
    vecaDic = {}
    for j in set(tag):
        vecaDic[j] = np.ones(len(vocabVec[0]))
    for i, value in enumerate(tag):
        for j in set(tag):
            if j==value:
                vecaDic[value] += np.array(vocabVec[i])
    for j in set(tag):
        vecaDic[j] = [vecaDic[j], tag.count(j)]

    for j in set(tag):
        vecaDic[j][0] = np.log(vecaDic[j][0]/(sum(vecaDic[j])+2))
        vecaDic[j][1] = np.log(vecaDic[j][1] / len(dataList))
    return vecaDic
def classify(vecaDic, tag, vocabulary, testData):
    testVec = getVocabVec(vocabulary, testData)
    pClassifyList = {}
    for j in set(tag):
        pClassifyList[j] = np.sum(testVec*vecaDic[j][0])+vecaDic[j][1]
    return pClassifyList

if __name__ == '__main__':
    tag, dataList, vocabulary = createData()
    vocabVec = getVocabVec(vocabulary, dataList)
    vecaDic = train(tag, vocabVec, dataList)
    testData = ['警方召开了全省集中打击赌博违法犯罪活动...']
    print(classify(vecaDic, tag, vocabulary, testData))
