#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import copy as copy
import math as math
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#基于距离的空间聚类算法
class KMeans:
    def __init__(self,features,labels,k):
        self.M,self.N = features.shape#原始数据：m样本量，n维输入特征
        self.X = features#训练样本特征
        self.Y = labels#样本标签
        self.preY = None#预测类别
        self.K = k#簇中心数量
        self.Cluster = None#簇
        self.C = None#簇头
    def createC(self,n):#生成簇首
        clusterHands = np.zeros((n,self.N))
        clusterHands = np.mat(clusterHands)
        for j in range(self.N):
            minJ = min(self.X[:,j])
            rangeJ = (max(self.X[:,j])-minJ)
            clusterHands[:,j] = minJ+rangeJ*np.random.rand(n,1)
        return clusterHands
    #计算调和平均数
    def harmonicMean(self,inX):
        if type(inX).__name__=='list':
            m = len(inX)
            inXtmp = np.array(inX)
            return m/(sum(1/inXtmp))
        else:
            m= inX.shape[0]
            return m/(sum(1/inX))
    def train(self):#缺点：最终结果跟初始点选择相关，容易陷入局部最优；K值如何选择问题
        clusterC = self.createC(self.K)#生成k个类别中心
        clusterTmp = {}#初始化簇群
        minVar = np.inf#初始化误差平方(距离和：各簇成员到其簇首的距离和)
        iter = 0
        while True:#迭代更新，对任意一个样本，求其到各中心的距离，将该样本归到距离最短的中心所在的类
            iter+=1
            disVar = np.zeros((self.K,1))
            for i in range(self.K):
                clusterTmp[i] = []
            #根据距离最近原则归类簇成员
            self.preY = np.zeros((self.M,1))#初始化预测类别
            for i in range(self.M):
                #计算距离最近的簇首的索引
                dis = np.sqrt(np.sum(np.power(clusterC-np.tile(self.X[i,:],(self.K,1)),2),axis=1))
                No = np.argmin(dis)
                # 存储每个簇的每个成员到簇首的距离之和
                disVar[No] += min(dis)[0,0]
                # 存储簇成员
                clusterTmp[No].append(self.X[i,:].tolist())
                #记录类别
                self.preY[i] = No
            # 若迭代更新后，各簇成员到其簇首距离和保持不变+簇首位置不变，则迭代结束，否则继续迭代
            var = sum(disVar)[0]
            # var = np.mean(disVar)
            # var = np.median(disVar)
            # var = self.harmonicMean(disVar)#调和平均数
            if self.C is not None and np.sum(np.sqrt(np.sum(np.power(self.C-clusterC,2),axis=1)))<1e-3 and abs(var-minVar)<1:
                break
            if var<minVar:
                minVar = var
                self.Cluster = clusterTmp
                self.C = clusterC

            # 利用均值等方法更新该类的中心值
            for j in range(self.K):
                if len(clusterTmp[j])>=1:
                    clusterC[j] = np.mean(clusterTmp[j],axis=0)#均值
                    # clusterC[j] = np.median(clusterTmp[j], axis=0)#中位数
                    # clusterC[j] = self.harmonicMean(clusterTmp[j])  # 调和平均数
                else:#若某个簇首没有分配到簇成员，则重新生成所有簇首位置
                    clusterC = self.createC(self.K)
                    break
            print('迭代次数：',iter)
            # self.draw(1)
    def draw(self,oneZero):
        plt.figure(1)
        if oneZero:
            plt.ion()#动态显示
            plt.clf()#清屏
        plt.title('基于距离的空间聚类算法')
        # 绘制散点
        # for key,values in self.Cluster.items():
        #     if len(values)>1:
        #         values = np.array(values)
        #         plt.scatter(values[:,0],values[:,1])

        plt.scatter(self.X[:,0],self.X[:,1],c=self.preY)
        #绘制簇首
        self.C= np.array(self.C)
        plt.scatter(self.C[:,0],self.C[:,1],marker='*')
        plt.show()
        if oneZero:#是否动态显示
            plt.pause(0.001)
            plt.ioff()# 关闭交互模式
#基于密度的空间聚类算法
class DBSCAN:#基于密度的空间聚类算法
    def __init__(self,features,labels,minNum,r=None):
        self.M,self.N = features.shape#原始数据：m样本量，n维输入特征
        self.X = features#训练样本特征
        self.Y = labels#样本标签
        self.preY = None#预测值，待编写
        self.C = None#核心点
        self.minPoints = minNum#最少点数
        if r is None:
            self.R = self.epsilon()#领域半径
        else:
            self.R = r
        self.Cluster = {}#簇
    #计算调和平均数
    def harmonicMean(self,inX):
        if type(inX).__name__=='list':
            m = len(inX)
            inXTmp = np.array(inX)
            return m/(sum(1/inXTmp))
        else:
            m= inX.shape[0]
            return m/(sum(1/inX))
    #计算最佳半径
    def epsilon(self,inX=None):
        if inX is None:
            inX = self.X
        m, n = np.shape(inX)
        xMax = np.max(inX, axis=0)
        xMin = np.min(inX, axis=0)
        eps = ((np.prod(xMax - xMin) * self.minPoints * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
        return eps
    #寻找核心点、边界点、噪声点
    def findpoint(self,inX=None,minPoints=None):
        if inX is None:
            inX = self.X
        if minPoints is None:
            minPoints = self.minPoints
        m = inX.shape[0]
        corePoints, borderPoints, nosePoints = [], [], []
        for i in range(m):
            # 计算距离
            dis = np.sqrt(np.sum(np.power(inX - np.tile(inX[i, :], (m, 1)), 2), axis=1))
            disR = dis[dis <= self.R]
            n = disR.shape[0] - 1
            # 判断该点包含多少点
            if n >= minPoints:  # 核心点
                corePoints.append(inX[i, :].tolist())
            elif 0 < n < minPoints:  # 边界点
                borderPoints.append(inX[i, :].tolist())
            else:  # 噪声点(离群点)
                nosePoints.append(inX[i, :].tolist())
        if len(corePoints) < minPoints:
            print('核心点太少，请重新设置参数')
            return
        else:
            return corePoints,borderPoints,nosePoints
    def train(self):
        #******************寻找核心点、边界点、噪声点*******************
        corePoints,borderPoints,nosePoints = self.findpoint()
        self.C = np.array(corePoints)#存储核心点
        # self.R = self.epsilon(noNoisePoints)#根据核心点+边界点计算最佳半径
        #****************1、对核心点归类*******************
        numC = 0#簇个数
        self.Cluster[numC] = []#簇初始化
        cp = np.array(corePoints)#核心点
        self.Cluster[numC].append(cp[0, :])#分类
        cp = np.delete(cp,0,axis=0)#删除已归类的核心点
        while cp.shape[0]>0:##若没有尚未归类的核心点，则停止归类
            #1、寻找距离当前簇最近的核心点:与簇中心最近的核心点
            # if len(self.Cluster[numC])>1:
            #     # cluterC = np.mean(self.Cluster[numC], axis=0)#中心
            #     cluterC = self.harmonicMean(self.Cluster[numC])#调和中心
            # else:
            #     cluterC = self.Cluster[numC]
            # disCC = np.sum(np.power(cp-cluterC,2),axis=1)
            # disCCIndex = np.argmin(disCC)
            #2、寻找距离当前簇最近的核心点:与簇边缘最近的核心点
            minDis = np.inf
            for j in range(cp.shape[0]):
                dis = min(np.sqrt(np.sum(np.power(self.Cluster[numC]-cp[j],2),axis=1)))
                if dis<minDis:
                    minDis = dis
                    disCCIndex = j
            #计算该核心点与已选核心点的距离最小值->寻找距离簇最近的核心点
            disCPs = min(np.sqrt(np.sum(np.power(self.Cluster[numC]-cp[disCCIndex],2),axis=1)))
            #判断该核心点是否在self.Cluster[numC]簇内
            if 0<=disCPs<=self.R:#如果在self.R邻域内，说明第i个核心点在该簇内
                # 把该核心点分配到该簇内
                self.Cluster[numC].append(cp[disCCIndex, :])
                #删除已归类的核心点
                cp = np.delete(cp,disCCIndex,axis=0)
            else:#距离第numC个簇最近的核心点不在该簇邻域内，说明第numC个簇已全部分类完毕，可以生成新簇来归类核心点
                numC += 1
                self.Cluster[numC] = []
                self.Cluster[numC].append(cp[0])
                cp = np.delete(cp,0,axis=0)
        #*****************2、对边界点进行归类************************
        borderPoints = np.array(nosePoints)
        if borderPoints.shape[0]>1:
            for i in range(borderPoints.shape[0]):
                for key,values in self.Cluster.items():
                    # 计算第i个边界点到第key个簇中各点的最小距离，判断第i个边界点是否位于第key个簇内
                    disBC = min(np.sqrt(np.sum(np.power(values-borderPoints[i,:], 2), axis=1)))
                    #若是，则将第i个边界点归类到第key个簇内
                    if disBC<=self.R:
                        self.Cluster[key].append(borderPoints[i,:])
    def draw(self,display=False):
        plt.figure(2)
        plt.title('基于密度的空间聚类算法')
        if display:
            plt.ion()#动态显示
            plt.clf()#清屏
        # 绘制散点
        for i in range(self.M):
            plt.scatter(self.X[i,0],self.X[i,1],c='k')

        for key, values in self.Cluster.items():
            values = np.array(values)
            plt.scatter(values[:, 0], values[:, 1],s=100)

        #绘制核心点
        for i in range(self.C.shape[0]):
            plt.scatter(self.C[i, 0], self.C[i, 1], c='r',marker='*')

        plt.show()
        if display:#是否动态显示
            plt.pause(0.001)
            plt.ioff()# 关闭交互模式
#谱聚类
class SC:
    def __init__(self,features,labels,k,sigma,model='KMeans'):
        self.M,self.N = features.shape#原始数据：m样本量，n维输入特征
        self.X = features#训练样本特征
        self.Y = labels#样本标签
        self.preY = None#预测类别
        self.K = k #聚类数量
        self.Sigma = sigma
        self.W = np.zeros((self.M,self.M))#邻接矩阵初始化
        self.model = model#kmeans聚类/GMM聚类
    def train(self):
        #1、生成邻接矩阵(权重)
        for i in range(self.M):
            for j in range(self.M):#self.X:数组.
                #高斯函数度量样本之间的距离，距离越远，权重越小
                self.W[i, j] = np.exp(-1.0 * (self.X[i] - self.X[j]).T @ (self.X[i] - self.X[j]) / (2 * self.Sigma** 2))
        #2、生成度量矩阵D、拉普拉斯矩阵L
        D =np.diag(np.sum(self.W,axis=1))
        L = D-self.W
        #3、标准化拉普拉斯矩阵:Ncut切图:L=D^(-0.5)LD^(-0.5),注意：这里是矩阵叉乘
        NormL = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))
        #4、生成前k个特征值对应的特征向量(升序)
        eigenValue,eigenVector = np.linalg.eig(NormL)#eigens[0]:特征值，eigens[1]:特征向量
        eigenIndex = np.argsort(eigenValue)[0:self.K]
        eigenVector = eigenVector[:,eigenIndex]#前k个特征向量组成m*k矩阵,特征向量可能存在复数
        #5、聚类
        if self.model=='kmeans':#kmeans聚类
            xx = KMeans(np.real(eigenVector), self.Y,self.K)#取特征向量实部
            xx.train()
        elif self.model == 'GMM':#GMM聚类
            xx = GMM(np.real(eigenVector), self.Y, self.K)  # 取特征向量实部
            xx.train()
        self.preY = xx.preY#聚类预测类别
    def draw(self):
        plt.figure(3)
        plt.title('谱聚类')
        # 绘制散点
        plt.scatter(self.X[:, 0], self.X[:, 1],c=np.array(self.preY))
        plt.show()
# 用高斯混合模型（GMM）的最大期望（EM）聚类
class GMM:
    def __init__(self,features,labels,k):
        self.M, self.N = features.shape  # 原始数据：m样本量，n维输入特征
        self.X = np.mat(features)  # 训练样本特征
        self.Y = np.mat(labels)  # 样本标签
        self.preY = None  # 预测类别
        self.K = k  # 聚类数量
        self.pc = None
        self.U = None
        self.Cov = None
    def pdf(self,inX,u,cov):#概率密度函数
        exp = np.exp(-(inX-u)@np.linalg.inv(cov)@(inX-u).T/2.0)
        deno = np.sqrt((2*np.pi)**self.N*np.linalg.det(cov))
        return exp/deno
    def train(self):
        #1初始化k个簇的高斯分布参数(均值和协方差矩阵)*******************************
        #均值
        u = np.mat(np.zeros((self.K,self.N)))
        for j in range(self.N):
            minJ = min(self.X[:,j])[0,0]
            rangeJ = (max(self.X[:,j])-minJ)[0,0]
            u[:,j] = minJ+rangeJ*np.random.rand(self.K,1)

        # u = np.mat(np.random.rand(self.K,self.N))
        # 协方差矩阵
        cov = np.array([np.eye(self.N)]*self.K)
        # cov = np.zeros((self.K,self.N,self.N))
        # for k in range(self.K):
        #     cov[k] = np.eye(self.N)
        #初始化类别先验概率(权重)
        pc = np.array([1/self.K]*self.K)
        #迭代求解最佳参数
        iter,uTmp,pcTmp,covTmp,Ltmp = 0,0,0,0,0
        maxL = -np.inf
        while True:
            iter += 1
        #2根据当前参数计算：样本点属于某个类别的概率(后验概率/响应度)*********************
            pcx = np.zeros((self.M,self.K))#p(c|x)样本点属于某个类别的概率矩阵：m*k
            px = np.zeros((self.M,1))#p(x):样本x的概率
            L = 0#p(X):关于整个样本数据的概率取对数
            for m in range(self.M):
                for k in range(self.K):
                    px[m] += (pc[k]*self.pdf(self.X[m,:],u[k,:],cov[k]))[0,0]
                L += np.log(px[m])
                for k in range(self.K):
                    pdf = self.pdf(self.X[m, :], u[k,:], cov[k])
                    pcx[m,k] =(pc[k]*pdf/px[m])[0,0]
            pcx = np.mat(pcx)
        #3根据验概率/响应度，更新参数:先验概率/权重p(c)、均值u、协方差矩阵cov
            for k in range(self.K):
                pc[k] = sum(pcx[:,k])/self.M
                u[k,:] = np.sum(np.multiply(self.X,pcx[:,k]), axis=0)/pc[k]/self.M
                # for m in range(self.M):
                #     covTmp += pcx[m,k]*(self.X[m,:]-u[k,:]).T*(self.X[m,:]-u[k,:])
                cov[k] = np.multiply((self.X-u[k,:]).T,pcx[:,k].T)*(self.X-u[k,:])
                cov[k] = cov[k]/pc[k]/self.M
        #重复迭代2和3直到算法收敛(每个簇的高斯分布参数变化不大)
            self.U,self.Cov,self.pc = u,cov,pc
            self.preY = np.argmax(pcx, axis=1)
            #动态显示
            self.draw(1)
            #判断是否收敛:L_new=log(∑p(x)),|L_new-L_old|<1e-5
            if L-maxL>1e-5:
                maxL = L
            else:
                break
            # if abs(L-Ltmp)<1e-5:
            #     break
            Ltmp = L
            print('迭代次数:',iter)
    def predict(self,inX=None):
        if inX is None:
            inX = self.X
        #根据当前参数计算：样本点属于某个类别的概率(后验概率/响应度)
        M,N = inX.shape
        pcx = np.zeros((M,self.K))
        px = np.zeros((M,1))
        for m in range(M):
            for k in range(self.K):
                px[m] += (self.pc[k]*self.pdf(inX[m,:],self.U[k,:],self.Cov[k]))[0,0]
            for k in range(self.K):
                pdf = self.pdf(inX[m,:],self.U[k,:],self.Cov[k])
                pcx[m,k] =(self.pc[k]*pdf/px[m])[0,0]
        pcx = np.mat(pcx)
    def draw(self,oneZero=False):
        #绘制分类图
        plt.figure(4)
        if oneZero:
            plt.ion()#动态显示
            plt.clf()#清屏
        plt.title('高斯混合模型')
        # 绘制散点
        xy = self.X.A
        plt.scatter(xy[:, 0], xy[:, 1],c=self.preY.A)
        #绘制簇中心
        plt.scatter(self.U[:, 0].A, self.U[:, 1].A, marker='*')
        plt.show()
        if oneZero:#是否动态显示
            plt.pause(0.001)
            plt.ioff()# 关闭交互模式
        #绘制直方图、绘制概率密度曲线、
#均值迁移
class MeanShift:
    def __init__(self,features,labels,minPoints=4,r=None):
        self.M, self.N = features.shape  # 原始数据：m样本量，n维输入特征
        self.X = features  # 训练样本特征
        self.Y = labels # 样本标签
        self.preY = None  # 预测类别
        self.minPoints = minPoints
        if r is None:
            self.R = self.epsilon()
        else:
            self.R = r  # 半径
        self.Cluster = {}  # 簇
        self.C = None
    # 核函数
    def kernel(self,bandwidth,dis):#高斯核
        left = 1 / (bandwidth * np.sqrt(2 * np.pi))
        right = np.exp(-np.power(dis, 2) / (2 * np.power(bandwidth, 2)))
        sum1 = left*right
        sum2 = sum1*dis
        return np.sum(sum2, axis=0) / np.sum(sum1, axis=0)
    #计算最佳半径
    def epsilon(self,inX=None):
        if inX is None:
            inX = self.X
        m, n = np.shape(inX)
        xMax = np.max(inX, axis=0)
        xMin = np.min(inX, axis=0)
        eps = ((np.prod(xMax - xMin) * self.minPoints * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
        return eps
    #寻找核心点、边界点、噪声点
    def findCore(self,inX=None,minPoints=None):
        if inX is None:
            inX = self.X
        if minPoints is None:
            minPoints = self.minPoints
        m = inX.shape[0]
        corePoints, borderPoints, nosePoints = [], [], []
        for i in range(m):
            # 计算距离
            dis = np.sqrt(np.sum(np.power(inX - np.tile(inX[i, :], (m, 1)), 2), axis=1))
            disR = dis[dis <= self.R]
            n = disR.shape[0] - 1
            # 判断该点包含多少点
            if n >= minPoints:
                corePoints.append(inX[i, :].tolist())
        if len(corePoints) < minPoints:
            print('核心点太少，请重新设置参数')
        else:
            return np.array(corePoints)
    def train(self):
        iter = 0
        corePoints = self.findCore(minPoints=2)#self.X/self.findCore(minPoints=2)#筛选后可以归类的点
        #随机选择一点为初始簇心点->c,初始化簇半径R,初始化偏移量Mtmp，步长step
        m, n = corePoints.shape
        i = np.random.randint(0, m)
        C,R,Mtmp,step = corePoints[i,:],self.R,np.zeros((1,n)),0
        #初始化簇
        numC = 0  # 簇个数
        while True:
            iter += 1
            #计算距离：获取当前距离簇心半径范围内的点
            dis = np.sqrt(np.sum(np.power(corePoints-np.tile(C,(corePoints.shape[0],1)),2),axis=1))
            disR = np.where((dis<R)&(dis>0))[0]
            clusters = corePoints[disR,:]#更新当前范围内的点，生成簇
            #计算迁移向量
            M = np.reshape(self.kernel(1, clusters-C),(1,2))#高斯核
            # M = np.reshape(sum(clusters-C)/pointsNum,(1,2))
            #更新簇的簇心位置
            C = C+M
            #判断是否收敛：偏移向量前后变化量小于阈值
            if np.linalg.norm(M-Mtmp)<1e-5:
                #更新当前簇的半径(距离簇心最远的簇成员之间的距离作为当前簇半径)
                dis = np.sqrt(np.sum(np.power(clusters - np.tile(C, (clusters.shape[0], 1)), 2), axis=1))
                maxIndex = np.argmax(dis)
                R = np.linalg.norm(clusters[maxIndex, :] - C)
                #生成簇
                self.Cluster[numC] = [C,R,clusters]
                self.draw(1)
                #更新corePoints:删除已归类的点
                corePoints = np.delete(corePoints, disR, axis=0)
                if corePoints.shape[0]>1:
                    numC += 1
                    R,i = self.R,np.random.randint(0,corePoints.shape[0])
                    C = corePoints[i,:]
                else:
                    break
                # break
            else:
                Mtmp = M
            #判断是否扩大范围：偏移向量大小即移动距离是否接近0
            if np.linalg.norm(M)<1e-5:#|M|<1e-5
                #更新半径
                step = self.epsilon(clusters)
                R += step
                # self.R = R
            # self.Cluster[numC] = [C, R, clusters]
            # self.draw(1)
            print('迭代次数:',iter)
    def draw(self,oneZero=False):
        #绘制分类图
        plt.figure(5)
        if oneZero:
            plt.ion()#动态显示
            plt.clf()#清屏
        plt.title('均值迁移')
        # 绘制散点
        xy = self.X
        plt.scatter(xy[:, 0], xy[:, 1],c='k')
        # for i in range(self.M):
        #     plt.text(xy[i, 0], xy[i, 1], str('%.1f' % i), ha='center',
        #          va='bottom', fontsize=7.5)
        #绘制簇
        for key, values in self.Cluster.items():
            # 绘制簇中心
            C = np.array(values[0])
            plt.scatter(C[:, 0], C[:, 1], marker='*')
            #绘制簇成员
            members = np.array(values[2])
            plt.scatter(members[:, 0], members[:, 1])
            #绘制圆
            R = np.array(values[1])#np.array(values[1])/self.R
            theta = np.arange(0,2*np.pi,0.01)
            x = C[0,0]+R*np.cos(theta)
            y = C[0,1]+R * np.sin(theta)
            plt.plot(x,y,'--')
        plt.show()
        if oneZero:#是否动态显示
            plt.pause(0.001)
            plt.ioff()# 关闭交互模式
#层次聚类
class HierarchicalClustering:
    def __init__(self,features,labels,k,minPoints=4,r=None):
        self.M, self.N = features.shape  # 原始数据：m样本量，n维输入特征
        self.X = features  # 训练样本特征
        self.Y = labels # 样本标签
        self.CoreX = None#过滤离群点的样本数据
        self.preY = None  # 预测类别
        self.K = k#簇个数
        self.Tree = {}  # 聚类树
        self.minPoints = minPoints
        if r is None:
            self.R = self.epsilon()
        else:
            self.R = r  # 半径
    #寻找核心点、边界点、噪声点
    def findCore(self,inX=None,minPoints=None):
        if inX is None:
            inX = self.X
        if minPoints is None:
            minPoints = self.minPoints
        m = inX.shape[0]
        corePoints, borderPoints, nosePoints = [], [], []
        for i in range(m):
            # 计算距离
            dis = np.sqrt(np.sum(np.power(inX - np.tile(inX[i, :], (m, 1)), 2), axis=1))
            disR = dis[dis <= self.R]
            n = disR.shape[0] - 1
            # 判断该点包含多少点
            if n >= minPoints:
                corePoints.append(inX[i, :].tolist())
        if len(corePoints) < minPoints:
            print('核心点太少，请重新设置参数')
            return
        else:
            return np.array(corePoints)
    #想找相同行对应的索引
    def findRow(self,inX,i,j):
        listI,listJ= [],[]
        for m in range(inX.shape[0]):
            if sum(abs(inX[m,:]-inX[i,:]))<1e-10:
                listI.append(m)
            elif sum(abs(inX[m, :] - inX[j,:]))<1e-10:
                listJ.append(m)
        listI.extend(listJ)
        return listI
    #多维矩阵排序返回升序索引
    def sorts(self,inX):
        m,n = inX.shape
        qq = np.array(list(range(m)))
        j = np.argsort(inX)[:, 0]
        jj = np.reshape(j, (m, 1))
        ij = np.insert(jj, 0, values=qq, axis=1)
        i=np.argsort(inX[ij[:,0],ij[:,1]])
        j = jj[np.argsort(inX[ij[:,0],ij[:,1]])]
        ij = np.insert(j,0,values=i,axis=1)
        return ij
    #计算最佳半径
    def epsilon(self,inX=None):
        if inX is None:
            inX = self.X
        m, n = np.shape(inX)
        xMax = np.max(inX, axis=0)
        xMin = np.min(inX, axis=0)
        eps = ((np.prod(xMax - xMin) * self.minPoints * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
        return eps
    #计算调和平均数
    def harmonicMean(self,inX):
        if type(inX).__name__=='list':
            m = len(inX)
            inXtmp = np.array(inX)
            return m/(sum(1/inXtmp))
        else:
            m= inX.shape[0]
            return m/(sum(1/inX))
    def train(self):
        self.CoreX = self.X#self.X/self.findCore(),self.X:原始样本点：、self.findCore():过滤掉离群点(噪声点)
        if self.CoreX is None:#样本太少，导致找不到足够的核心点
            return
        a = copy.deepcopy(self.CoreX)
        iter = 0
        while len(set(a[:,0]))>1:#迭代次数为样本数
            #计算样本之间的距离矩阵：m*m对称矩阵
            b = a@a.T
            c = np.diag(b)
            d = ((-2*b+c).T+c).T
            #为寻找最小距离的不同样本点/簇，需先将距离为0的点赋值为无穷大
            d[np.where(d==0)]=np.inf
            #寻找距离最小的(>=2)不同点对应的索引listIJ，将训练数据X[listIJ,:]赋其均值(簇心)。最小距离：距离簇心最短
            i,j = np.unravel_index(np.argmin(d), d.shape)#获取多维矩阵xx值对应的索引
            listIJ = self.findRow(a,i,j)#寻找在X中于X[i,:],X[j,:]样本点相同的所有样本点对应的索引
            means = np.mean(self.CoreX[listIJ,:],axis=0)#计算最近的样本点/簇的均值。算数平均
            # means = np.median(self.CoreX[listIJ,:],axis=0)#中位数
            # means = self.harmonicMean(self.CoreX[listIJ,:])#调和平均
            a[listIJ,:] = means#赋予相同的均值即视为归为同一类
            #生成聚类树
            self.Tree[iter] = copy.deepcopy(a)
            iter += 1
            print('迭代次数:',iter)
        #根据树结构，做分类预测
        self.predict(self.CoreX)
    def predict(self,inX=None):
        if inX is None:
            inX = self.X
        #树结构
        self.C = np.unique(self.Tree[len(self.Tree) - self.K], axis=0)
        self.preY = np.zeros((inX.shape[0], 1))
        i = 0
        for key in self.C:
            listI = np.where(self.Tree[len(self.Tree) - self.K]==key)[0]
            self.preY[listI] = i
            i += 1
        #kmeans
        # m = inX.shape[0]
        # self.C = np.unique(self.Tree[len(self.Tree) - self.K],axis=0)
        # self.preY = np.zeros((m, 1))
        # for i in range(m):
        #     # 计算距离最近的簇首的索引
        #     dis = np.sqrt(np.sum(np.power(self.C - np.tile(inX[i, :], (self.C.shape[0], 1)), 2), axis=1))
        #     No = np.argmin(dis)
        #     # 记录类别
        #     self.preY[i] = No
    def draw(self,oneZero=False,k=None):
        #绘制分类图
        plt.figure(6)
        if oneZero:
            plt.ion()#动态显示
            plt.clf()#清屏
        plt.title('层次聚类')
        # 绘制散点
        # plt.scatter(self.X[:, 0], self.X[:, 1], c='k')
        plt.scatter(self.CoreX[:, 0], self.CoreX[:, 1], c=self.preY)
        plt.scatter(self.C[:, 0], self.C[:, 1], marker='*',c='r')#绘制簇心
        plt.show()
        if oneZero:#是否动态显示
            plt.pause(0.001)
            plt.ioff()# 关闭交互模式
#sklearn
def sklearn(inX,model,para=None):
        if model=='kmeans':#para:[簇个数]
            from sklearn.cluster import KMeans
            y_pred = KMeans(n_clusters=para[0], random_state=9).fit_predict(inX)
        elif model=='dbscan':#para:[最少点数，领域半径]
            from sklearn.cluster import DBSCAN
            y_pred = DBSCAN(min_samples=para[0],eps=para[1]).fit_predict(inX)#eps=20, min_samples=10
        elif model=='sc':#para:[簇个数,高斯参数sigma,聚类方式:kmeans/discretize]
            from sklearn.cluster import SpectralClustering
            y_pred = SpectralClustering(n_clusters=para[0], gamma=para[1],assign_labels=para[2]).fit_predict(inX)
        elif model=='GMM':#para:
            from sklearn import mixture
            y_pred = mixture.GaussianMixture(n_components=para).fit_predict(inX)
        elif model=='meanShift':#para:
            from sklearn.cluster import MeanShift
            y_pred = MeanShift().fit_predict(inX)
        elif model=='HierarchicalClustering':#para:
            from sklearn.cluster import AgglomerativeClustering
            y_pred = AgglomerativeClustering(n_clusters=para[0],affinity='euclidean',linkage=para[1]).fit_predict(inX)
        else:
            y_pred = np.zeros((inX[0],1))
        plt.scatter(inX[:, 0], inX[:, 1], c=y_pred)
        plt.show()
#加载数据
def sklearnData(model,noise=None):#sklearn数据集
    from sklearn import datasets
    samplesNum = 200
    centers = [[-2, -8], [-1, 7], [6, 5], [9, 8], [0, 2.3]]#cluster_std=[0.2,0.4,0.6,0.3,0.3]
    # centers = [[-2, -8], [-1, 7], [6, 5]]  # cluster_std=[0.2,0.4,0.6]
    if model=='moons':
        dataSet,labels = datasets.make_moons(n_samples=samplesNum, noise=noise, random_state=0)
    elif model=='circles':
        dataSet, labels = datasets.make_circles(n_samples=samplesNum, noise=noise, factor=0.5, random_state=1)
    elif model=='blobs':
        dataSet, labels = datasets.make_blobs(n_samples=samplesNum, n_features=2,centers=centers,cluster_std=[0.2,0.4,0.6,0.3,0.3])
    elif model=='classification':
        dataSet, labels = datasets.make_classification(n_samples=samplesNum,n_features=2,n_informative=2,n_redundant=0, random_state=1)
    elif model=='gaussian_quantile':
        dataSet, labels = datasets.make_gaussian_quantiles(n_samples=samplesNum, n_classes=2, random_state=500)
    dataSet, labels = np.array(dataSet),np.array(labels)
    # labels[labels == 0] = -1
    return dataSet, labels
def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    dataSet = np.array(baseData)[:, :2]
    labels = np.array(list(map(int,np.array(baseData)[:,-1]))).T
    return dataSet, labels
if __name__=='__main__':
    # ******加载数据集**********
    trainData = 'ex0.txt'  # ex00/ex0~9/testSet
    RawDataSet, OutValue = loadData(r'./Data/' + trainData)  # m*n,m*1

    # RawDataSet, OutValue = sklearnData('blobs',0.1)#moons/circles/blobs/classification/gaussian_quantile

    # from sklearn import datasets
    # centers = [[-2, -8], [-1, 7], [6, 5], [9, 8], [0, 2.3]]
    # RawDataSet, OutValue = datasets.make_blobs(n_samples=300, n_features=2,centers=centers,cluster_std=[0.2, 0.4, 0.6, 0.3, 0.3])

    # X1, y1 = datasets.make_circles(n_samples=2000, factor=.6, noise=.02)
    # # DataSet = X1
    # X2, y2 = datasets.make_blobs(n_samples=400, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
    # RawDataSet = np.concatenate((X1, X2))
    #********数据预处理***********
    import CleanData as clean
    data = clean.CleanData(RawDataSet, OutValue)  # 实例化
    # data.ascend(3)  # 数据集升维处理:普通高阶/组合高阶,用于多项式回归
    # data.zero_centered('all')  # 中心化
    # data.z_score('all')  # 标准正态分布
    # data.normData('all')  # 归一化
    data.PCA(2)
    RawDataSet, OutValue= data.X, data.Y
    # ******训练**********
    #1、kmeans算法***********************************
    clusterNum = 3
    xx1 = KMeans(RawDataSet, OutValue,clusterNum)
    xx1.train()
    xx1.draw(0)
    para = [clusterNum]
    sklearn(RawDataSet,'kmeans',para)
    #2、DBSCAN算法***********************************
    # minPoints,radius = 5,0.1
    # xx2 = DBSCAN(RawDataSet, OutValue,minPoints,radius)
    # xx2.train()
    # xx2.draw()
    # para = [minPoints,radius]
    # sklearn(RawDataSet,'dbscan',para)
    #3、谱聚类***************************************
    # clusterNum,sigma =2,0.1
    # model = 'kmeans'#聚类方式：kmeans/GMM
    # xx3 = SC(RawDataSet, OutValue,clusterNum,sigma,model)
    # xx3.train()
    # xx3.draw()
    # para = [clusterNum,sigma,'kmeans']
    # sklearn(RawDataSet,'sc',para)
    #4、高斯混合模型(GMM)******************************
    # clusterNum = 5
    # xx4 = GMM(RawDataSet, OutValue,clusterNum)
    # xx4.train()
    # xx4.draw()
    # para = clusterNum
    # sklearn(RawDataSet, 'GMM',para)
    #5、均值迁移**************************************
    # xx5 = MeanShift(RawDataSet, OutValue,r=0.04)
    # xx5.train()
    # xx5.draw()
    # sklearn(RawDataSet, 'meanShift')
    #6、层次聚类**************************************
    # clusterNum = 2
    # xx6 = HierarchicalClustering(RawDataSet, OutValue,clusterNum)
    # xx6.train()
    # xx6.draw()
    # para = [clusterNum,'average']#ward/average/complete/single#由于类聚类之间的距离(相似度)衡量方法不同，算法分类结果易受噪声点(离群点)干扰，需数据预处理(过滤噪声点)或者改变距离计算方法。
    # sklearn(RawDataSet, 'HierarchicalClustering',para)
