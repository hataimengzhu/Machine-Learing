#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import matplotlib.pyplot as plt
import numpy as np
import Edge as Edge
class CleanData:#fun:归一化数据集、获取数据集边界点、平衡数据类别
    def __init__(self,features,labels):#features:特征，m*n,labels:类别：m*1
        self.oraX,self.oraY = features,labels#原始数据集
        self.X,self.Y = None,None#初始化清洗后的数据集
    #*********归一化:将一个数据集合压缩到同一区间中（0 ~ 1，-1 ~ +1）********************
    #归一化：x=(x-minX)/(maxX-minX) ->[0,1]
    #归一化：x=2(x-minX)/(maxX-minX)-1 ->[-1,1]
    def normData(self,model='alone'):
        minValues = self.oraX.min(0)#获取每列的最小值
        maxValues = self.oraX.max(0)#获取每列的最大值
        meanValues = self.oraX.mean(0)#获取每列的均值
        scope = maxValues - minValues
        m = self.oraX.shape[0]
        self.X = (self.oraX - np.tile(minValues, (m, 1)))/scope
        if model == 'alone':
            self.Y = self.oraY
        elif model == 'all':
            minValues = self.oraY.min(0)  # 获取每列的最小值
            maxValues = self.oraY.max(0)  # 获取每列的最大值
            scope = maxValues - minValues
            self.Y = (self.oraY - np.tile(minValues, (m, 1))) / scope
    #平均归一化：x=(x-mean)/(maxX-minX)
    def averageNnormData(self,model='alone'):
        minValues = self.oraX.min(0)#获取每列的最小值
        maxValues = self.oraX.max(0)#获取每列的最大值
        meanValues = self.oraX.mean(0)#获取每列的均值
        scope = maxValues - minValues
        m = self.oraX.shape[0]
        self.X = (self.oraX - meanValues)/scope
        if model == 'alone':
            self.Y = self.oraY
        elif model == 'all':
            minValues = self.oraY.min(0)  # 获取每列的最小值
            maxValues = self.oraY.max(0)  # 获取每列的最大值
            meanValues = self.oraY.mean(0)  # 获取每列的均值
            scope = maxValues - minValues
            self.Y = (self.oraY - meanValues) / scope
    #非线性归一化：
    def nonlinerNnormData(self,n,model='alone'):
        pass
    #**************************************************
    # 标准化：x=(x-meanX)/标准差
    def z_score(self,model='alone'):#标准化
        meanXValues = self.oraX.mean(0)#获取每列的均值
        xStd = self.oraX.std()#标准差
        #计算
        self.X = (self.oraX-meanXValues)/xStd
        meanYValues = self.oraY.mean(0)#1*m
        if model == 'alone':
            self.Y = self.oraY
        elif model == 'all':
            yStd = self.oraY.std()
            self.Y = (self.oraY - meanYValues) / yStd
    #中心化/零均值化：x = x-meanx
    def zero_centered(self,model='alone'):
        meanXValues = self.oraX.mean(0)#获取每列的均值
        #计算
        self.X = self.oraX-meanXValues
        if model == 'alone':
            self.Y = self.oraY
        elif model == 'all':
            meanYValues = self.oraY.mean(0)#1*m
            self.Y = self.oraY - meanYValues
    # 数据升维,用于多项式回归
    def ascend(self,n,addOne=False,model='普通高阶'):#数据升维,用于多项式回归
        if n<2:
            print('无需升维')
            self.X, self.Y = self.oraX,self.oraY
            return
        #数据升维，增加特征种类
        if model == '交互式特征':#Interaction Features

            pass
        elif model == '普通高阶':#Polynomial Features
            self.X = self.oraX
            for i in range(self.oraX.shape[1]):
                for j in range(2, n + 1):
                    self.X = np.insert(self.X, self.oraX.shape[1]+i, values=np.power(self.oraX[:, i], j).T, axis=1)
        elif model =='组合高阶':
            pass
        if addOne:#为数据集添加数值为1的特征
            self.X = np.insert(self.X, self.X.shape[1], values=np.ones((1, self.X.shape[0])), axis=1)  # 特征+位移量，m*(n+1)
        self.Y = self.oraY
    # 获取边界点，针对2维特征数据集、针对二元分类问题
    def edgeData(self):
        # 数据预处理-提取凸边边界点
        edge = Edge.Edge(self.oraX, self.oraY)
        edge.margin()
        dataTmp, labelsTmp = [], []
        for key in edge.boundaryPoint.keys():# 遍历类别，edge.boundaryPoint[key]：列表
            edge.boundaryPoint[key].pop()#删除重复点(起点==终点)
            dataTmp.extend(edge.boundaryPoint[key])
            #生成边界点对应的类别集合
            # 判断是否是嵌套列表
            if any(isinstance(i, list) for i in dataTmp):#即多层列表
                labelsTmp.extend([key] * len(edge.boundaryPoint[key]))
            else:#只有一层列表
                labelsTmp.extend([key] * 1)
        outX = np.mat(dataTmp)#凸边界点特征集合，m*n
        outY = np.mat(labelsTmp).T#凸边界点类别集合,m*1
        self.X,self.Y = outX, outY
    # 平衡数据集中各个类别的比例，针对不平衡数据集、针对二元分类问题
    def evenData(self,model):#平衡数据集中各个类别的比例，针对不平衡数据集、针对二元分类问题
        #计算类比占比，判断是否为不平衡类数据集

        # model:1、过采样，2、欠采样，3、混合采样
        if model=='moreSample':
            pass
        elif model=='lessSample':
            pass
        elif model=='variousSample':
            pass
    # 主要成分分析：降级数据的复杂性，识别最重要的多个特征
    def PCA(self,k=1):
        m = self.oraX.shape[0]
        meanXValues = self.oraX.mean(0)#获取每列的均值
        self.X = self.oraX-meanXValues#中心化/零均值化
        # self.X = self.oraX
        #计算协方差矩阵：cov=1/m*(X-u).T@(X-u),u=0
        cov = 1/m*self.X.T@self.X
        #生成前k个特征值对应的特征向量(降序)
        eigenValue,eigenVector = np.linalg.eig(cov)#eigens[0]:特征值，eigens[1]:特征向量
        eigenIndex = np.argsort(-eigenValue)#argsort默认升序处理，这里需要降序
        self.PCAeigenVector = eigenVector[:,eigenIndex]#降序特征值对应的全部特征向量
        eigenVector = eigenVector[:,eigenIndex[0:k]]#取前k个特征向量组成n*k矩阵,特征向量可能存在复数
        #将样本数据投影到上一步k个特征向量构建的新空间中:n维特征转换为(降维)k维特征
        self.X = self.X@eigenVector#m*k
        # self.X = (eigenVector.T@self.X.T).T # m*k
        # 计算方差百分比(贡献率)和累加方差百分比
        sumVar = sum(eigenValue)
        ppVar = [da/sumVar for da in eigenValue[eigenIndex]]#方差百分比
        cumVar = np.cumsum(ppVar)#累计方差百分比
        self.ppVar = ppVar
        self.cumVar = cumVar
        return eigenValue,eigenVector
    # 奇异值分界：提取信息：简化数据、去除噪声
    def SVD(self):
        X = self.oraX
        m, n = X.shape
        if m >= n:
            D, U = np.linalg.eig(X @ X.T)  # 特征向量已单位化
            D = np.real(D)
            D[abs(D) < 1e-5] = 0
            index = np.argsort(-D)
            D = D[index]
            D = D ** 0.5
            DD = np.diag(D[D != 0])
            # U = U/np.linalg.norm(U,axis=0)#m*m
            U = np.real(U[:, index])

            S = np.zeros(X.shape)  # m*n
            mn = DD.shape[0]
            S[:mn, :mn] = DD

            DN = np.zeros((m, m))  # m*m
            DN[:mn, :mn] = np.linalg.inv(DD)
            V = np.real(((DN @ U.T @ X)[:n, :n]).T)  # n*n

            return U,list(D[D>0]),V
        else:
            D, V = np.linalg.eig(X.T @ X)  # 特征向量已单位化
            D = np.real(D)
            D[abs(D) < 1e-5] = 0
            index = np.argsort(-D)
            D = D[index]
            D = D ** 0.5
            DD = np.diag(D[D != 0])
            V = V / np.linalg.norm(V, axis=0)
            V = np.real(V[:, index])

            S = np.zeros(X.shape)  # m*n
            mn = DD.shape[0]
            S[:mn, :mn] = DD

            DN = np.zeros((n, n))  # n*m
            DN[:mn, :mn] = np.linalg.inv(DD)

            U = np.real((X @ V @ DN)[:m, :m])  # ]n*n

            U=U[:,len(list(D[D>0]))]
            V = V[:,len(list(D[D>0]))]
            return U,list(D[D>0]),V
    def SVDshow(self,U,lam,V):
        plt.figure()
        m,n = U.shape[0],V.shape[0]
        S = np.zeros((m,n))
        for i in range(len(lam)):
            S[i,i] = lam[i]
            X = U@S@V.T

            plt.ion()  # 动态显示
            plt.clf()  # 清屏
            plt.scatter(X[:,0],X[:,1],c='r')
            # plt.scatter(self.oraX[:, 0], self.oraX[:, 1], c='b')
            plt.show()
            plt.pause(1)
            plt.ioff()#关闭交互模式
        # plt.figure()
        # plt.scatter(X[:, 0], X[:, 1], c='r')
        # plt.show()
    def draw(self):
        plt.figure(1)
        plt.title('数据预处理')
        #绘制原始数据
        plt.subplot(131)
        m = self.oraX.shape[0]
        meanXValues = self.oraX.mean(0)#获取每列的均值
        xy = self.oraX-meanXValues#中心化/零均值化

        plt.scatter(xy[:,0],xy[:,1])
        m = self.PCAeigenVector.shape[0]
        a = np.mat(self.PCAeigenVector[:, 0])
        a = np.insert(a,0,values=np.array(-a),axis=0)
        plt.plot(a[:, 0]*10, a[:, 1]*10, 'r--')

        b = np.mat(self.PCAeigenVector[:, 1])
        b = np.insert(b,0,values=np.array(-b),axis=0)
        plt.plot(b[:,0]*10,b[:,1]*10,'g--')
        # 绘制预处理数据
        plt.subplot(132)
        plt.scatter(self.X[:, 0], self.X[:, 1])
        plt.plot(a[:, 0]*10, a[:, 1]*10,'r--')
        plt.plot(b[:,0]*10,b[:,1]*10,'g--')
        # plt.scatter(self.X[:,0],np.array([0]*self.X.shape[0]))

        #绘制方差
        plt.subplot(133)
        plt.title('')
        plt.xlabel('主成分数目(特征值降序排序)')
        plt.ylabel('方差的百分比/累加和')
        if self.ppVar is not None and self.cumVar is not None:
            plt.plot(self.ppVar,label='方差百分比')
            plt.plot(self.cumVar,label='累加方差百分比')
            plt.legend(['方差百分比','累加方差百分比'])

        plt.show()
#加载数据集，默认数据集的最后一列是标签
def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()#包含换行符、制表符，每个元素为字符串
    #数据清洗：去换行符、通过制表符分割为列表、列表中每一个字符串元素转换为浮点数
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    #判断数据是否存在NaN，若存在，则将其替换为均值
    baseData = np.array(baseData)
    nanIJ = np.argwhere(np.isnan(baseData))
    if len(nanIJ)>0:
        tmp = baseData
        tmp[nanIJ[:,0],nanIJ[:,1]] = 0
        m = baseData.shape[0]
        means = np.mean(tmp,axis=0)
        for line in nanIJ:
            # counts = list(nanIJ[:, 1]).count(line[1])
            # baseData[line[0],line[1]]=means[line[1]]*m/(m-counts)
            baseData[line[0], line[1]] = means[line[1]]
    #样本数据
    if baseData.shape[1]>2:
        dataSet = baseData[:, :-1]
    else:
        dataSet = baseData
    #标签：数据清洗->列表中每一个浮点数元素转换为整数
    labels = np.array(list(map(int,baseData[:,-1]))).T
    return dataSet, labels
if __name__ == '__main__':
    # ******加载数据集**********
    trainData = 'ex2.txt'#testSet.txt/secom.data/ex00.txt,ex0~9.txt
    rawDataSet, rawLabels = loadData('./Data/' + trainData)#m*n,m*1
    xx = CleanData(rawDataSet, rawLabels)
    # xx.edgeData()
    # xx.PCA(2)
    U,lam,V = xx.SVD()
    xx.SVDshow(U,lam,V)

