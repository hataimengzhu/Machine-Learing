import numpy as np
import random
import copy
import time
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    dataSet = np.mat(baseData)[:, :2]
    labels = np.mat(list(map(int,np.array(baseData)[:,-1]))).T
    dataDic = {1: [], -1: []}
    for da in baseData:
        dataDic[int(da[-1])].append(da[:-1])
    return dataSet, labels, dataDic
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
    labels[labels==0]=-1
    dataDic = {1: [], -1: []}
    for da in range(len(labels)):
        dataDic[int(labels[da])].append(dataSet[da,:].tolist()[0])
    return dataSet, labels, dataDic
class SVM:
    def __init__(self,features,labels,dataDic,c,ksi,kernel,maxiter):
        self.M, self.N = features.shape#原始数据:m样本量，n维输入特征
        self.X = features#原始数据
        self.Y = labels#原始数据样本标签
        self.DataDic = dataDic
        self.B = 0.0#位移量
        self.W = None#
        self.Alphas = np.mat([0.0]*self.M).T##原始数据样本拉格朗日乘子矩阵
        self.C = c # 惩罚参数
        self.Ksi = ksi #松弛
        self.Kernel = kernel#核函数
        self.MaxIter = maxiter#迭代次数
        self.K = np.mat(np.zeros((self.M,self.M)))#核函数内积
        for ii in range(self.M):
            self.K[:,ii] = self.kernel(self.X,self.X[ii,:])
        # self.E = [self.EE(i).getA()[0][0] for i in range(self.M)] #[f(xi)-yi]  m*1
        self.E = (np.multiply(self.Alphas, self.Y).T*self.K+self.B).T - self.Y # m*1
        self.Vector = None#支持向量
        self.tiger = []#目标函数值
    def kernel(self,dataSet,inX):# 核函数内积：[xi·inx]  m*1
        K = np.mat(np.zeros((dataSet.shape[0],1)))  # m*1
        if self.Kernel[0] == 'linear':
            K = dataSet * inX.T  # m*1
            return K
        if self.Kernel[0] == 'poly':
            tmp = dataSet * inX.T+1 #[xi*x+1]  m*1
            K = np.power(Kernel[1][0]*tmp,Kernel[1][1])#[(xi*x+1)^p], m*1
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
        # yifxi = self.Y[order]*self.f(order)
        if 0<self.Alphas[order]<self.C:
            return self.Y[order]*self.E[order] == 0
        elif self.Alphas[order] == 0:
            return self.Y[order]*self.E[order]>0
        elif self.Alphas[order]==self.C:
            return self.Y[order]*self.E[order]<0
    def f(self,i):
        Nzero = np.where(self.Alphas != 0)[0]
        xishu = np.multiply(self.Alphas[Nzero], self.Y[Nzero]).T  # (m*1).T lambdai*yi
        fx = xishu * self.K[:, i][Nzero] + self.B  # (1*m)*(m*1)=1*1#  f(xi)=w*xi+b
        return fx
    def fAll(self):
        xishu = np.multiply(self.Alphas, self.Y).T  # (m*1).T lambdai*yi
        fxMatrix = xishu * self.K + self.B  # (1*m)*(m*m)=1*m#  f(xi)=w*xi+b
        return fxMatrix.T
    def w(self):
        Nzero = np.where(self.Alphas != 0)[0]
        xishu = np.multiply(self.Alphas[Nzero],self.Y[Nzero]).T #(m*1).T lambdai*yi
        return xishu * self.X[Nzero,:]  # (1*m)*(m*2)=1*2
    def EE(self,i):
        Nzero = np.where(self.Alphas!=0)[0]
        xishu = np.multiply(self.Alphas[Nzero], self.Y[Nzero]).T  # (m*1).T lambdai*yi
        fx = xishu * self.K[:, i][Nzero] + self.B  # (1*m)*(m*1)=1*1#  f(xi)=w*xi+b
        EE = fx-self.Y[i]#1*1
        # fxMatrix = xishu * self.K + self.B  # (1*m)*(m*m)=1*m#  [f(x1)=w*x1+b,f(x2)=w*x2+b...]
        # eeMatrix = fxMatrix.T - self.Y
        return EE
    # def EEMatrix(self):
    #     return (np.multiply(self.Alphas, self.Y).T* self.K+self.B).T-self.Y
    def eta(self,i,j):
        return self.K[i,i]+self.K[j,j]-2*self.K[i,j]
    def train(self,model, display,deviation):
        start = time.time()  # 计时开始
        entireSet = True #是否全样本遍历
        iter = 0
        pairChanged = -1
        self.tiger.append(self.tigerFun())  # 目标函数值初始化
        while pairChanged!=0 and iter<self.MaxIter:#迭代更新alpha
            pairChanged = 0
            if entireSet:#全样本遍历只执行一次
                for i in range(self.M):
                    pairChanged += self.updateAlpha(i,model)
                    print("全样本遍历:第%d次迭代 样本:%d, alpha:%f,迭代次数:%d" % (iter, i, self.Alphas[i],pairChanged))
                if pairChanged==0:#全样本遍历没有找到支持向量(极点)
                    print('分类模型找不到支持向量，请调整参数或更换分类模型')
                    break
                entireSet = False
            else:
                nonBoundsList = np.nonzero((self.Alphas.A>0) & (self.Alphas.A<self.C))[0]
                for i in nonBoundsList:
                    # etaMatrix = [self.eta(i, jj) for jj in range(len(nonBoundsList))] # 计算eta，得到[eta]
                    # j = nonBoundsList[np.argsort(etaMatrix)[-1]]
                    pairChanged += self.updateAlpha(i,model)
                    print("非边界遍历(!=0,!=C):第%d次迭代 样本:%d, alpha:%f,迭代次数：%d" % (iter, i, self.Alphas[i],pairChanged))
            iter +=1
            self.tiger.append(self.tigerFun()) # 记录每次迭代后的目标函数值
        #存储支持向量
        self.Vector = np.where(self.Alphas>0)
        end = time.time()  # 计时结束
        print("耗时：{}".format(end - start))
        self.check(deviation)#检测是否满足sum(alphai*yi)==0?,alphai>=0?+训练集内测
        if not display:
            self.display(0)
    def selectJ(self,i,model):
        if model == 'random':  # 随机选取
            l = list(range(self.M))
            seq = l[:i] + l[i + 1:]
            return random.choice(seq)
        elif model == 'Ej_Ei/eta!=0':  # 计算Ej_Ei/eta!=0，
            Ei_old= self.E[i]
            Ej_Ei = self.E-np.tile(Ei_old, (self.M, 1))
            etaMatrix = np.mat([self.eta(i,jj) for jj in range(self.M)]).T # 计算eta，得到[eta]
            listJ = np.where((Ej_Ei/etaMatrix!=0))[0]
            j = random.choice(listJ)
            return j
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
            if alphaNew > H:
                alphaNew = H
            elif alphaNew < L:
                alphaNew = L
            return alphaNew
    def selectB(self,aiNew, ajNew, biNew, bjNew):
        # 位移量B由支持向量决定。支持向量： 0 < alpha <C
        if 0 < aiNew < self.C and 0 < ajNew < self.C:
            return (biNew + bjNew) / 2
        elif 0 < aiNew < self.C:
            return biNew
        elif 0 < ajNew < self.C:
            return bjNew
        else:
            return (biNew+bjNew)/2
    def getB(self,i,j,aiNew,ajNew,ei,ej,yi,yj,aiOld, ajOld):#计算位移量
        biNew = -ei-yi*self.K[i,i]*(aiNew-aiOld)-yj*self.K[i,j]*(ajNew-ajOld)+self.B
        bjNew = -ej-yi*self.K[i,j]*(aiNew-aiOld)-yj*self.K[j,j]*(ajNew-ajOld)+self.B
        return self.selectB(aiNew, ajNew, biNew, bjNew) # 筛选B
    def updateAlpha(self,i,model,j=None):
        #对于i、j:均选取违背KKT条件的i、j做迭代更新，其中相对于i，j随机选取/(Ej_Ei/eta!=0)
        if j is None:
            j = self.selectJ(i, model)
        eta = self.eta(i, j)
        if (self.kkt(i) and self.kkt(j)) or eta==0 or self.selectAlpha(i, j):#符合kkt条件？ eta=0?,L=H?
            print("样本:%d满足kkt条件："%i)
            return 0
        else:
            Ei_old, Ej_old = self.E[i], self.E[j]
            AiOld, AjOld = self.Alphas[i], self.Alphas[j]
            yi, yj = self.Y[i], self.Y[j]

            ai_new_unc = yi*(Ej_old-Ei_old)/eta+AiOld  # 未修剪的alpha
            AiNew = self.selectAlpha(i, j, ai_new_unc)
            AjNew = AjOld - (AiNew - AiOld) * yi * yj
            # AjNew = self.selectAlpha(j, i, aj_new_unc)#若对aj_new_unc裁剪，可能导致sum(lambdai*yi)!=0
            if AiNew != AiOld:
                # 更新self.B
                self.B = self.getB(i, j, AiNew, AjNew, Ei_old, Ej_old, yi, yj, AiOld, AjOld)
                # 更新Alphas
                self.Alphas[i], self.Alphas[j] = AiNew, AjNew
                # 更新self.E
                Nzero = np.where(self.Alphas != 0)[0]#Alphas更新后，更新E
                self.E = (np.multiply(self.Alphas[Nzero], self.Y[Nzero]).T * self.K[:,Nzero].T + self.B).T - self.Y
                # 更新self.W
                self.W = self.w()
            else:# AiNew 无变化，自然AjNew也无变化
                print('Alpha无变化')
        return 1
    def disMargin(self):#计算样本线性间隔距离
        w = self.W
        return 2/np.sqrt(w*w.T).tolist()[0][0]
    def tigerFun(self):#目标函数
        alphasTmp = self.Alphas
        alphaLabel = np.multiply(alphasTmp, self.Y)# lambdai*yi
        alpLab = alphaLabel * alphaLabel.T#
        XXT = self.K
        return np.sum(alphasTmp) - 1 / 2 * np.sum(np.multiply(alpLab, XXT))
    def interval(self,xy,X):# 计算点到决策边界距离
        wNorm2 = np.linalg.norm(self.W, axis=1, keepdims=True)
        dataset = np.mat(xy)
        if self.Kernel[0] =='linear':
            tmp = (dataset * self.W.T + self.B) / wNorm2
        else:
            xishu = np.multiply(self.Alphas, self.Y).T
            tmp = np.mat(np.zeros((xy.shape[0],1)))
            for ii in range(xy.shape[0]):
                tmp[ii] = xishu*(self.kernel(self.X, dataset[ii, :]))+self.B#(1*m)(m*1)
            d = tmp/wNorm2

        d = np.array(sum(tmp.getA().tolist(), [])).reshape(X.shape)
        return d, wNorm2
    def check(self,deviation):#检查kkt条件：alphai>=0,sum(alphai*yi)=0,是否满足
        print('w:',self.W.getA()[0][0],self.W.getA()[0][1],'B:',self.B.getA()[0][0])
        print('目标函数最终值:',self.tiger[-1])
        if any(i < 0 for i in self.Alphas):#KKT:lambdai>=0
            print('kkt:lambdai>=0不满足')
        elif sum(np.multiply(self.Alphas, self.Y)).getA()[0][0] != 0:#KKT:sum(lambdai*yi)==0
            print('KKT:sum(lambdai*yi)==0不满足')
        else:
            print('KKT:满足')
        predict = [1 if da>0 else -1 for da in self.fAll()]
        predict = np.mat(predict).T
        accuracy = abs(predict-self.Y)<=deviation
        print('训练集内测精确率：{:.2%}'.format(float(sum(accuracy.tolist(), []).count(True) / accuracy.shape[0])))
    def test(self,inX):
        Nzero = np.where(self.Alphas != 0)[0]
        xishu = np.multiply(self.Alphas[Nzero], self.Y[Nzero]).T
        tmp = xishu * (self.kernel(self.X[Nzero], inX)) + self.B  # (1*m)(m*1) f(inX)=[alphai*yi*<xi·inx>]+b
        return 1 if tmp>0 else -1
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
        plt.xlim([-2, 11])
        plt.ylim([-10, 10])
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
                plt.text(x + 0.01, y + 0.04, str('%.3f' % alpha.getA()[0][0]), ha='center',
                         va='bottom', fontsize=7.5)
            elif alpha < 0:
                x, y = self.X[i, 0], self.X[i, 1]
                plt.scatter(x, y, s=100, c='none', alpha=0.7, linewidth=1.5, edgecolor='y', marker='*')
                plt.text(x + 0.01, y + 0.04,str('%.6f' % alpha.getA()[0][0]), ha='center',
                         va='bottom', fontsize=7.5)
        # 要画决策边界，必须要有网格
        axis = np.linspace(min(self.X[:,0]).getA()[0][0]-5,max(self.X[:,0]).getA()[0][0]+5,30)
        ayis = np.linspace(min(self.X[:, 1]).getA()[0][0]-5,max(self.X[:, 1]).getA()[0][0]+10,30)
        # 将特征向量转换为特征矩阵的函数
        # 核心是将两个特征向量广播，以便获取x.shape*y.shape这么多点的横坐标和纵坐标
        # 获取x.shape*y.shape这么多个坐标点
        X,Y = np.meshgrid(axis,ayis)
        XY = np.vstack([X.ravel(),Y.ravel()]).T# ravel()降维函数，把每一行都放在第一行，vstack能够将多个结构一致的一维数组按行堆叠起来
        # 输入每个新样本到决策样本边界的距离                 # xy就是形成的网络，遍及在画布上密集的点
        D,wNorm = self.interval(XY,X)# D:各个点到决策边界距离,wNomr:|w|,2范数，边缘分界面到决策边界面距离(线性)：1/|w|
        # 画决策边界到平行决策边界的超平面
        plt.contourf(X, Y, D, levels=[-1, 0, 1], alpha=0.5)
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
if __name__=='__main__':
    #加载数据集
    DataSet, Labels, DataDic = loadData('testSet.txt')
    # DataSet, Labels, DataDic = sklearnData('gaussian_quantile',0.15)#moons/circles/blobs/classification/gaussian_quantile

    C= float('inf')#float('inf')/0.6 惩罚因子
    Ksi = None #松弛变量
    Kernel = ['poly',[1,3]]#['linear',None]/['poly',[1/DataSet.shape[0],3]]]/['rbf',3]/['sigmoid',[1/DataSet.shape[0],0]]#核函数
    interation = 1000#最大迭代次数
    svm= SVM(DataSet, Labels, DataDic,C,Ksi,Kernel,interation)#SVM实例化
    Model = 'Ej_Ei/eta!=0'#random/Ej_Ei/eta!=0 #select j方式
    display = 0#是否动态显示分类过程
    Deviation = 0  # 偏差,样本预测值与真实值偏差，用于测试
    svm.train(Model,display,Deviation)#训练
    svm.sklearn('sigmoid')#调用sklearn库分类



