#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import random
import time
import Edge as Edge
import matplotlib as mplt
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
'''
Input:
dataSet:样本特征值--matrix--m*n
labels: 样本类别标签--matrix--m*1
c:惩罚因子--(0,+∞]，默认值：+∞
kernel:核函数--list--['linear',None]/['rbf',3]//['poly',[1/m,3]]]/['sigmoid',[1/m,0]]
show:是否动态显示SVM分类过程--1:动态显示,0:静态显示,默认值：0
inX:测试集--matrix--1*n
Function:
svm = SVM(dataSet, labels,kernel,c,maxIte):实例化
svm.train(show)：训练数据+分类+内测+绘图
svm.predict(inX)：测试数据，返回预测结果
svm.sklearn(kernel[0]):与sklearn库对照
Example:
DataSet, Labels = loadData('./Data/'+'testSet.txt')
C= 0.2#float('inf')/0.6 惩罚因子
Kernel = ['poly',[1,3]]#['linear',None]/['poly',[1/DataSet.shape[0],3]]]/['rbf',3]/['sigmoid',[1/DataSet.shape[0],0]]#核函数
svm= SVM(DataSet, Labels,Kernel,C)#SVM实例化
show = 1#动态显示分类过程
svm.train(show)#训练+内测+绘图
'''


class SVM:
    def __init__(self,rawFeatures,rawLabels,kernel,c=float('inf')):
        self.oraX = rawFeatures#原始数据
        self.oraY = rawLabels#原始数据样本标签
        self.DataDic = {1: [], -1: []}
        for da in range(len(self.oraY)):
            self.DataDic[int(self.oraY[da])].append(self.oraX[da, :].tolist()[0])
        self.X,self.Y = self.edgeData(rawFeatures,rawLabels)
        self.M, self.N = self.X.shape#edge数据:m样本量，n维输入特征
        self.B = 0.0#位移量
        self.W = None#
        self.Alphas = np.mat([0.0]*self.M).T#原始数据样本拉格朗日乘子矩阵
        self.C = c # 惩罚参数
        self.Kernel = kernel#核函数
        self.MaxIter = 1000#最大迭代次数
        self.K = np.mat(np.zeros((self.M,self.M)))#核函数内积
        for ii in range(self.M):
            self.K[:,ii] = self.kernel(self.X,self.X[ii,:])
        self.xishu = np.multiply(self.Alphas, self.Y).T#lambdai*yi
        self.E = (self.xishu*self.K+self.B).T - self.Y # m*1
        self.Vector = []#支持向量+支持向量特征值+支持向量标签
        self.Nzero = None#np.where(self.Alphas!=0)[0]
        self.tiger = []#目标函数值
    def edgeData(self,rawfeatures,rawlabels):
        # 数据预处理-提取边界点
        edge = Edge.Edge(rawfeatures, rawlabels)
        model = '最大凸边-外边'  # '最大凸边-外边'/'最小凹边'
        edge.margin()
        dataTmp, labelsTmp = [], []
        for key in edge.boundaryPoint.keys():
            edge.boundaryPoint[key].pop()#删除重复点(起点==终点)
            dataTmp.extend(edge.boundaryPoint[key])
            if any(isinstance(i, list) for i in dataTmp):
                labelsTmp.extend([key] * len(edge.boundaryPoint[key]))
            else:
                labelsTmp.extend([key] * 1)
        return np.mat(dataTmp), np.mat(labelsTmp).T
    def kernel(self,dataSet,inX):# 核函数内积：[xi·inx]  m*1
        K = np.mat(np.zeros((dataSet.shape[0],1)))  # m*1
        if self.Kernel[0] == 'linear':
            K = dataSet * inX.T  # m*1
            return K
        if self.Kernel[0] == 'poly':
            tmp = dataSet * inX.T+1 #[xi*x+1]  m*1
            K = np.power(self.Kernel[1][0]*tmp,self.Kernel[1][1])#[(xi*x+1)^p], m*1
            return K
        if self.Kernel[0] == 'rbf':
            for ii in range(dataSet.shape[0]):
                K[ii] = (dataSet[ii]-inX)*(dataSet[ii]-inX).T#||x-y||^2  1*1
            K = np.exp(-K/(2*self.Kernel[1]**2))#[||x-y||^2]  m*1
            return K
        if self.Kernel[0] == 'sigmoid':
            tmp = self.Kernel[1][0]*(dataSet * inX.T)-self.Kernel[1][1]  #[k*xi*xi+1]  m*1
            K = np.tanh(tmp)#tanh([k*xi*x+1]), m*1
            return K
    def kkt(self,order):#KTT条件要求
        yiEi = self.Y[order]*self.E[order]
        if 0<self.Alphas[order]<self.C:
            return yiEi == 0
        elif self.Alphas[order] == 0:
            return yiEi>0
        elif self.Alphas[order]==self.C:
            return yiEi<0
    def f(self,i):
        fx = self.xishu * self.K[:, i][self.Nzero] + self.B  # (1*m)*(m*1)=1*1#  f(xi)=w*xi+b
        return fx
    def fAll(self):
        xishu = np.multiply(self.Alphas, self.Y).T  # (m*1).T lambdai*yi
        fxMatrix = xishu * self.K + self.B  # (1*m)*(m*m)=1*m#  f(xi)=w*xi+b
        return fxMatrix.T
    def w(self):
        return self.xishu * self.X[self.Nzero,:]  # (1*m)*(m*2)=1*2
    def EE(self,i):
        fx = self.xishu * self.K[:, i][self.Nzero] + self.B  # (1*m)*(m*1)=1*1#  f(xi)=w*xi+b
        EE = fx-self.Y[i]#1*1
        return EE
    def eta(self,i,j):
        return self.K[i,i]+self.K[j,j]-2*self.K[i,j]
    def train(self,show=None,model=None):
        start = time.time()  # 计时开始
        entireSet = True #是否全样本遍历
        iter = 0
        pairChanged = -1
        record = 0
        self.tiger.append(self.tigerFun())  # 目标函数值初始化
        recordTiger = self.tiger[-1]
        while pairChanged!=0 and iter<self.MaxIter:#迭代更新alpha
            pairChanged = 0
            if entireSet:#全样本遍历只执行一次
                for i in range(self.M):
                    pairChanged += self.updateAlpha(i,model)
                    # print("全样本遍历:第%d次迭代 样本:%d, alpha:%f,迭代次数:%d" % (iter, i, self.Alphas[i],pairChanged))
                if pairChanged==0:#全样本遍历没有找到支持向量(极点)
                    print('分类模型找不到支持向量，请调整参数或更换分类模型')
                    break
                entireSet = False
            else:
                nonBoundsList = np.nonzero((self.Alphas.A>0) & (self.Alphas.A<self.C))[0]
                for i in nonBoundsList:
                    # j = random.choice(nonBoundsList)
                    pairChanged += self.updateAlpha(i,model)
                    # print("非边界遍历(!=0,!=C):第%d次迭代 样本:%d, alpha:%f,迭代次数：%d" % (iter, i, self.Alphas[i],pairChanged))
            iter +=1
            self.tiger.append(self.tigerFun()) # 记录每次迭代后的目标函数值
            # 限制迭代次数
            if recordTiger==self.tiger[-1]:
                record +=1
                if record>40:
                    break
            else:
                recordTiger = self.tiger[-1]
                record = 0
            if show:
                # 更新self.W
                self.W = self.w()
                self.display(show)  # show=0：静态显示svm，show=1:动态显示SVM
        # 更新self.W
        self.W = self.w()
        #存储支持向量
        vectorIndex = np.where((self.Alphas.A>0) & (self.Alphas.A<=self.C))[0]
        vectorData = self.X[vectorIndex,:]
        vectorAlpha = self.Alphas[vectorIndex]
        vectorLabels = self.Y[vectorIndex]
        self.Vector = [vectorAlpha,vectorLabels,vectorData]
        end = time.time()  # 计时结束
        self.check()#检测是否满足sum(alphai*yi)==0?,alphai>=0?+训练集内测
        print('迭代次数:',iter,"SVM分类器耗时：{}".format(end - start))
        self.display(0)  # show=0：静态显示svm
        #存储支持向量数据到本地
        np.save('./Data/svmVectorAlpha.npy', self.Vector[0])
        np.save('./Data/svmVectorLabels.npy', self.Vector[1])
        np.save('./Data/svmVectorData.npy', self.Vector[2])
    def selectJ(self,i,model=None):
        if model is None:# 随机选取
            l = list(range(self.M))
            seq = l[:i] + l[i + 1:]
            return random.choice(seq)
        else:# 计算Ej_Ei/eta!=0，
            Ei_old= self.E[i]
            Ej_Ei = self.E-np.tile(Ei_old, (self.M, 1))
            etaMatrix = np.mat([self.eta(i,jj) if self.eta(i,jj)!=0 else np.inf for jj in range(self.M)]).T # 计算eta，得到[eta]
            listJ = np.where((Ej_Ei/etaMatrix!=0))[0]
            return random.choice(listJ)
    def selectAlpha(self,i,j,alphaNew=None):
        # 确定alphai、alphaj的上、下界限
        if self.Y[i] == self.Y[j]:
            L = max(0, self.Alphas[i] + self.Alphas[j] - self.C)
            H = min(self.C, self.Alphas[i] + self.Alphas[j])
        else:
            L = max(0, self.Alphas[i] - self.Alphas[j])
            H = min(self.C, self.C + self.Alphas[i] - self.Alphas[j])
        if alphaNew is None:
            if L==H:
                return True
        else:
            if alphaNew>H:
                alphaNew = H
            elif alphaNew<L:
                alphaNew = L
            if alphaNew>self.C and alphaNew- self.C < 1e-5:#对于浮点数，判断是否相等没有意义，有意义的比较是比较两个数是否足够接近
                alphaNew = self.C
            # if alphaNew <1e-5:
            #     alphaNew = 0
            return alphaNew
    def selectB(self,aiNew, ajNew, biNew, bjNew):
        # 位移量B由支持向量决定。支持向量： 0 < alpha <C
        if 0 < aiNew <= self.C and 0 < ajNew <= self.C:
            return (biNew + bjNew) / 2
        if 0< aiNew <=self.C:
            return biNew
        elif 0< ajNew <= self.C:
            return bjNew
        else:
            return self.B
    def getB(self,i,j,aiNew,ajNew,ei,ej,yi,yj,aiOld, ajOld):#计算位移量
        biNew = -ei-yi*self.K[i,i]*(aiNew-aiOld)-yj*self.K[i,j]*(ajNew-ajOld)+self.B
        bjNew = -ej-yi*self.K[i,j]*(aiNew-aiOld)-yj*self.K[j,j]*(ajNew-ajOld)+self.B
        return self.selectB(aiNew, ajNew, biNew, bjNew) # 筛选B
    def updateAlpha(self,i,model,j=None):
        #对于i、j:均选取违背KKT条件的i、j做迭代更新，其中相对于i，j随机选取/(Ej_Ei/eta!=0)
        if j is None:
            j = self.selectJ(i, model)
        eta = self.eta(i, j)
        if eta==0 or self.selectAlpha(i, j):#eta=0?,L=H?
            return 0
        if self.kkt(i) and self.kkt(j):#符合kkt条件？
            # print("样本:%d满足kkt条件："%i)
            return 0
        Ei_old, Ej_old = self.E[i], self.E[j]
        AiOld, AjOld = self.Alphas[i], self.Alphas[j]
        yi, yj = self.Y[i], self.Y[j]
        ai_new_unc = yi*(Ej_old-Ei_old)/eta+AiOld  # 未修剪的alpha
        AiNew = self.selectAlpha(i, j, ai_new_unc)
        aj_new_unc = AjOld - (AiNew - AiOld) * yi * yj
        AjNew = self.selectAlpha(j, i, aj_new_unc)#若对aj_new_unc裁剪，可能导致sum(lambdai*yi)!=0
        if AiNew != AiOld:
            # 更新self.B
            self.B = self.getB(i, j, AiNew, AjNew, Ei_old, Ej_old, yi, yj, AiOld, AjOld)
            # 更新Alphas
            self.Alphas[i], self.Alphas[j] = AiNew, AjNew
            # 更新self.E
            self.Nzero = np.where(self.Alphas!=0)[0]
            self.xishu = np.multiply(self.Alphas[self.Nzero], self.Y[self.Nzero]).T
            self.E = (self.xishu * self.K[:,self.Nzero].T + self.B).T - self.Y#Alphas更新后，更新E
        # else:# AiNew 无变化，自然AjNew也无变化
        #     # print('Alpha无变化')
        return 1
    def tigerFun(self):#目标函数
        alphasTmp = self.Alphas
        alphaLabel = np.multiply(alphasTmp, self.Y)# lambdai*yi
        alpLab = alphaLabel * alphaLabel.T#
        XXT = self.K
        return np.sum(alphasTmp) - 1 / 2 * np.sum(np.multiply(alpLab, XXT))
    def interval(self,xy):# 计算点到决策边界距离
        wNorm2 = np.linalg.norm(self.W, axis=1, keepdims=True)
        dataset = np.mat(xy)
        tmp = np.mat(np.zeros((xy.shape[0],1)))
        for ii in range(xy.shape[0]):
            tmp[ii] = self.xishu*(self.kernel(self.X[self.Nzero,:], dataset[ii, :]))+self.B#(1*m)(m*1)
        d = tmp/wNorm2
        d = np.array(sum(d.getA().tolist(), [])).reshape([int(np.sqrt(xy.shape[0]))]*2)#30*30
        return d, wNorm2
    def check(self,deviation=None):#检查kkt条件：alphai>=0,sum(alphai*yi)=0,是否满足
        print('w:',self.W.A[0][0],self.W.A[0][1],'B:',self.B[0,0])
        print('目标函数最终值:',self.tiger[-1])
        if any(i < 0 for i in self.Alphas):#KKT:lambdai>=0
            print('kkt:lambdai>=0不满足')
        elif sum(np.multiply(self.Alphas, self.Y)).getA()[0][0] != 0:#KKT:sum(lambdai*yi)==0
            print('KKT:sum(lambdai*yi)==0不满足')
        else:
            print('KKT:满足')

        fx = np.mat(np.zeros((self.oraX.shape[0], 1)))#预测函数：原始数据据m*1
        for ii in range(self.oraX.shape[0]):
            # fx[ii] = self.xishu*(self.kernel(self.X[self.Nzero,:], self.oraX[ii, :]))+self.B#(1*m)(m*1)，f(z)=wz+b
            fx[ii] = self.xishu * (self.kernel(self.Vector[2], self.oraX[ii, :])) + self.B  # (1*m)(m*1)，f(z)=wz+b

        predict = [1 if da>0 else -1 for da in fx]
        predict = np.mat(predict).T
        if deviation is None:
            accuracy = abs(predict - self.oraY) == 0.0
        else:
            accuracy = abs(predict-self.oraY)<=deviation
        print('训练集内测错误率：{:.2%}'.format(1-float(sum(accuracy.tolist(), []).count(True) / accuracy.shape[0])))
    def predict(self,inX,vector=None):
        if vector is None:
            vector = self.Vector
        xishu = np.multiply(vector[0],vector[1]).T
        vectorData = vector[2]
        fx = np.mat(np.zeros((inX.shape[0], 1)))#预测函数：原始数据据m*1
        for ii in range(inX.shape[0]):
            fx[ii] = xishu*(self.kernel(vectorData, inX[ii, :]))+self.B#(1*m)(m*1)，f(z)=wz+b
        predict = [1 if da>0 else -1 for da in fx]#预测结果
        # tmp = self.xishu * (self.kernel(self.X[self.Nzero], inX)) + self.B  # (1*m)(m*1) f(inX)=[alphai*yi*<xi·inx>]+b
        return [fx,predict]
    def display(self,oneZero):
        plt.figure(1)
        if oneZero:
            plt.ion()#动态显示
            plt.clf()#清屏
        plt.subplot(121)
        plt.title('分类')
        # 设置X轴标签
        plt.xlabel('x1')  # 设置X/Y轴标签是在对应的figure后进行操作才对应到该figure
        # 设置Y轴标签
        plt.ylabel('x2')
        plt.xlim([min(self.oraX[:,0]).getA()[0][0]-10, max(self.oraX[:,0]).getA()[0][0]+10])
        plt.ylim([min(self.oraX[:,1]).getA()[0][0]-10, max(self.oraX[:,1]).getA()[0][0]+10])
        colors, markers = ['', 'r', 'b'], ['', 'o', 's']
        for key, values in self.DataDic.items():#绘制散点
            values = np.array(values)
            plt.scatter(values[:, 0], values[:, 1])
        for i, alpha in enumerate(self.Alphas):
            if 0 < alpha < self.C:
                x, y = self.X[i, 0], self.X[i, 1]
                plt.scatter(x, y, s=100, c='none', alpha=0.7, linewidth=1.5, edgecolor='k', marker='s')
                plt.text(x + 0.01, y + 0.1, str('%.6f' % alpha.getA()[0][0]), ha='center',
                         va='bottom', fontsize=7.5)
            elif alpha == self.C:
                x, y = self.X[i, 0], self.X[i, 1]
                plt.scatter(x, y, s=100, c='none', alpha=0.7, linewidth=1.5, edgecolor='y', marker='8')
                plt.text(x + 0.01, y + 0.04, 'C', ha='center',
                         va='bottom', fontsize=10)
            elif alpha < 0:
                x, y = self.X[i, 0], self.X[i, 1]
                plt.scatter(x, y, s=100, c='none', alpha=0.7, linewidth=1.5, edgecolor='y', marker='*')
                plt.text(x + 0.01, y + 0.04,str('%.6f' % alpha.getA()[0][0]), ha='center',
                         va='bottom', fontsize=7.5)
        # 要画决策边界，必须要有网格
        axis = np.linspace(min(self.X[:,0]).getA()[0][0]-10,max(self.X[:,0]).getA()[0][0]+10,30)
        ayis = np.linspace(min(self.X[:, 1]).getA()[0][0]-10,max(self.X[:, 1]).getA()[0][0]+10,30)
        # 将特征向量转换为特征矩阵的函数
        # 核心是将两个特征向量广播，以便获取x.shape*y.shape这么多点的横坐标和纵坐标
        # 获取x.shape*y.shape这么多个坐标点
        X,Y = np.meshgrid(axis,ayis)
        XY = np.vstack([X.ravel(),Y.ravel()]).T# ravel()降维函数，把每一行都放在第一行，vstack能够将多个结构一致的一维数组按行堆叠起来
        # 输入每个新样本到决策样本边界的距离                 # xy就是形成的网络，遍及在画布上密集的点
        D,wNorm = self.interval(XY)# D:各个点到决策边界距离,wNomr:|w|,2范数，边缘分界面到决策边界面距离(线性)：1/|w|
        # 画决策边界到平行决策边界的超平面
        plt.contourf(X, Y, D, levels=[-1/wNorm, 0, 1/wNorm], alpha=0.5)
        # 根据预测值对采样点选择颜色填充
        # predict = self.predict(np.mat(XY))[0]
        # predict = np.array(predict).reshape(X.shape)
        # cm_light = mplt.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        # cm_dark = mplt.colors.ListedColormap(['g', 'r', 'b'])
        # plt.pcolormesh(X, Y, predict, alpha=0.5, cmap=cm_light)#'jet'/cm_light #pcolormesh根据预测值对采样点选择颜色填充
        plt.subplot(122)
        plt.title('目标函数值')
        # 设置X轴标签
        plt.xlabel('iters')  # 设置X/Y轴标签是在对应的figure后进行操作才对应到该figure
        # 设置Y轴标签
        plt.ylabel('TigerValue')
        plt.plot(range(len(self.tiger)), self.tiger)
        plt.show()
        if oneZero:#是否动态显示
            plt.pause(0.001)
    def plotROC(self,predStrengths,fit=False):#predStrengths：分类器预测强度,m*1，self.oraY：真实类别
        numPosClas = np.sum(self.oraY== 1.0)  # 统计正类的数量
        numNegClas = len(self.oraY) - numPosClas# 统计负类的数量
        sortedIndex = (-predStrengths.T).argsort() # 预测强度排序,从高到低
        numP,numN,AUCH,AUCW = 0,0,0,1/numNegClas
        coordinate =[]
        for index in sortedIndex.tolist()[0]:
            if self.oraY[index]==1.0:#向上位移
                numP += 1
                TPR = numP/numPosClas
                FPR = numN / numNegClas
            else:
                numN += 1
                FPR = numN/numNegClas
                TPR = numP / numPosClas
                AUCH += TPR
            coordinate.append([FPR,TPR])
        AUC = AUCH*AUCW
        coordinate = np.mat(coordinate)
        #凸包拟合
        plt.figure()
        if fit:# 凸包拟合锯齿边界
            xx = Edge.Edge(coordinate)  # 实例化
            xx.margin()  # 获取凸边界点
            xx.boundaryPoint[1].append(xx.boundaryPoint[1].pop(0))
            plt.plot(np.mat(xx.boundaryPoint[1])[:, 0], np.mat(xx.boundaryPoint[1])[:, 1], 'r')
        else:#锯齿边界
            plt.plot(coordinate[:, 0], coordinate[:, 1], 'r')

        plt.plot([0, 1], [0, 1], 'b--')
        plt.title('ROC曲线')
        plt.xlabel('灵敏度')
        plt.ylabel('特异度')
        plt.axis([-0.01, 1.01, -0.01, 1.01])
        print('AUC面积为:', AUC)  # 计算AUC
        plt.show()
    def sklearn(self,kernel):
        from sklearn.svm import SVC
        plt.figure(100)
        for key, values in self.DataDic.items():
            values = np.array(values)
            plt.scatter(values[:, 0], values[:, 1])
        # 要画决策边界，必须要有网格
        axis = np.linspace(min(self.X[:,0]).getA()[0][0]-5,max(self.X[:,0]).getA()[0][0]+5,30)
        ayis = np.linspace(min(self.X[:, 1]).getA()[0][0]-5, max(self.X[:, 1]).getA()[0][0]+10, 30)
        # 将特征向量转换为特征矩阵的函数，# 核心是将两个特征向量广播，以便获取x.shape*y.shape这么多点的横坐标和纵坐标
        # 获取x.shape*y.shape这么多个坐标点
        X,Y = np.meshgrid(axis,ayis)
        XY = np.vstack([X.ravel(),Y.ravel()]).T # ravel()降维函数，把每一行都放在第一行，vstack能够将多个结构一致的一维数组按行堆叠起来
        # 输入每个新样本到决策样本边界的距离          # xy就是形成的网络，遍及在画布上密集的点
        # 画决策边界到平行决策边界的超平面
        clf =  SVC(kernel=kernel).fit(self.X, self.Y)
        P = clf.decision_function(XY).reshape(X.shape)
        plt.contourf(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
        plt.show()

def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()#包含换行符、制表符，每个元素为字符串
    #数据清洗：去换行符、通过制表符分割为列表、列表中每一个字符串元素转换为浮点数
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    dataSet = np.mat(baseData)[:, :-1]
    # 数据清洗：列表中每一个浮点数元素转换为整数：标签
    labels = np.mat(list(map(int,np.array(baseData)[:,-1]))).T
    return dataSet, labels
def sklearnData(model,noise=None):#sklearn数据集
    from sklearn import datasets
    samplesNum = 100
    if model=='moons':
        dataSet,labels = datasets.make_moons(n_samples=samplesNum, noise=noise, random_state=0)
    elif model=='circles':
        dataSet, labels = datasets.make_circles(n_samples=samplesNum, noise=noise, factor=0.5, random_state=1)
    elif model=='blobs':
        dataSet, labels = datasets.make_blobs(n_samples=samplesNum, centers=2, random_state=5)
    elif model=='classification':
        dataSet, labels = datasets.make_classification(n_samples=samplesNum,n_features=2,n_informative=2,n_redundant=0, random_state=1)
    elif model=='gaussian_quantile':
        dataSet, labels = datasets.make_gaussian_quantiles(n_samples=samplesNum, n_classes=2, random_state=500)
    dataSet, labels = np.mat(dataSet),np.mat(labels).T
    labels[labels == 0] = -1
    return dataSet, labels		
#正对不平衡类问题，改进方向：基于代价函数决策分类，即不同类别对应不同的惩罚因子，

if __name__ == '__main__':
    trainData = 'testSet.txt'#testSet.txt/horseColicTraining2.txt
    rawDataSet, rawLabels = loadData('./Data/' + trainData)
    Kernel = ['rbf',3]  # ['linear',None]/['rbf',3]/['poly',[1/DataSet.shape[0],3]]]/['sigmoid',[1/DataSet.shape[0],0]]#核函数
    C = float('inf')# float('inf')/0.6 惩罚因子
    svm = SVM(rawDataSet, rawLabels,Kernel,C)
    svm.train(0)
    # supportVector = [np.load('./Data/svmVectorAlpha.npy', allow_pickle=False),
    #                  np.load('./Data/svmVectorLabels.npy', allow_pickle=False),
    #                  np.load('./Data/svmVectorData.npy', allow_pickle=False)]
    # print('预测类别：',svm.predict(np.mat([[1,0],[8,0]]),supportVector))
    # print('预测类别：', svm.predict(rawDataSet,supportVector))
    svm.sklearn('rbf')

