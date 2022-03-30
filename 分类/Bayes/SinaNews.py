import os
import numpy as np
import jieba
import operator
import copy

def loadData(path):
    print('loadData')
    # 加载数据集
    with open(path+r'\ClassList.txt',encoding='utf-8') as fi:
        tmp = fi.readlines()
    classCodeList = [da.strip().split('\t') for da in tmp]
    # 遍历文件,获取原始数据列表、类别表
    shujuList, classList = [], [],  # 词条表、类别表
    for root, dire, files in os.walk(path+r'\Sample'):
        for file in files:
            curFilePath = os.path.join(root, file)
            with open(curFilePath, encoding='utf-8') as txt:
                txtShuju = txt.readlines()
            classList.extend([root.split('\\')[-1]] * len(txtShuju))
            shujuList.extend([[da.strip()] for da in txtShuju])
    # 分词
    shujuList = [list(set(jieba.cut(da[0], cut_all=False))) for da in shujuList]
    # 数据清洗
    shujuList = cleanData(shujuList)
    # 生成词汇表
    chihuiList = list(set(sum(shujuList, [])))

    return shujuList, classList, classCodeList, chihuiList
def cleanData(shujuList):
    print('cleanData')
    # 过滤stopwords+数字+符号+len(词)==1
    # 加载stopwords
    with open(r".\Data\Chinese_from_dongxiexidian-master\stopwords.txt",'r', encoding='utf-8') as file:
        stopWords = [da.strip() for da in file.readlines()]
    # 过滤stopwords+数字+符号+len(词)==1
    cleanList = [0]*len(shujuList)
    if any(isinstance(i, list) for i in shujuList): # 判断是否为嵌套列表
        for i, value in enumerate(shujuList):
            tmp = []
            for j in value:
                if j not in stopWords and j != ' ' and len(j) > 1 and not j.isdigit():
                    tmp.append(j)
            cleanList[i] = tmp
    else:
        tmp = []
        for i in shujuList:
            if i not in stopWords and i != ' ' and len(i) > 1 and not i.isdigit():
                tmp.append(i)
        cleanList = tmp
    return cleanList
def getVectorList(chihuiList, shujuList):
    print('getVectorList')
    shujuVec=[]
    if any(isinstance(i, list) for i in shujuList):
        for i in shujuList:
            tmp = [0]*len(chihuiList)
            for j in i:
                if j in chihuiList:
                    tmp[chihuiList.index(j)] = 1
            shujuVec.append(tmp)
    else:
        shujuVec = [0]*len(chihuiList)
        for i in shujuList:
            if i in chihuiList:
                shujuVec[chihuiList.index(i)] = 1

    return shujuVec
def train(classList, shujuVec):
    print('train')
    trainDic = {}
    for i in set(classList):
        trainDic[i] = np.ones(len(shujuVec[0]))
    for i, value in enumerate(classList):
        for da in set(classList):
            if value == da:
                trainDic[da] += shujuVec[i]
    for i in set(classList):
        trainDic[i] = [trainDic[i], classList.count(i)]
    for i in set(classList):
        trainDic[i][0] = np.log10(trainDic[i][0]/(sum(trainDic[i][0])+2))
        trainDic[i][1] = np.log10(trainDic[i][1]/len(classList))

    return trainDic
def classiy(trainDic, chihuiList, tagCodeList, testTxt):
    print('classify')
    # 测试数据预处理
    wordList = list(set(jieba.cut(testTxt[0], cut_all=False))) # 分词
    testList = cleanData(wordList) # 数据清洗
    testVec = getVectorList(chihuiList, testList) # 获得测试数据词向量

    resultDic = {}
    for key in trainDic.keys():
        for da in tagCodeList:
            if key in da:
                resultDic[da[1]] = 10**(np.sum(testVec*trainDic[key][0]) + trainDic[key][1])
    result = sorted(resultDic.items(), key=operator.itemgetter(1), reverse=True)
    return result
def InterTest():
    pass


if __name__=='__main__':
    # filePath = r'.\Data\SogouC'
    # DataList, TagList, TagCode, Vocabulary = loadData(filePath)
    # np.save('NewsSinaVocabulary.npy', Vocabulary)
    # np.save('NewsSinaTagCode.npy', TagCode)
    Vocabulary = list(np.load('NewsSinaVocabulary.npy', allow_pickle=True))
    TagCode = np.load('NewsSinaTagCode.npy', allow_pickle=True).tolist()
    # DataVec = getVectorList(Vocabulary, DataList)
    # TrainDic = train(TagList, DataVec)
    # np.save('NewsSinaTrainDic.npy', TrainDic)
    TrainDic = np.load('NewsSinaTrainDic.npy', allow_pickle=True).tolist()
    Txt = ['在未来5年里，仿制药市场的成长趋势不可抵挡。ＩＭＳ的分析数据显示，在未来5年中，仿制药的销售额将以14％～17％的速度递增，比整个医药行业的销售预期多9％。Ｂａｉｎ＆Ｃｏｍｐａｎｙ公司认为，仅2008年就将有价值780亿美元的处方药受到仿制药的冲击，而2005年这个数字为200亿美元']
    ClassiyResult = classiy(TrainDic, Vocabulary, TagCode, Txt)
    print(ClassiyResult)