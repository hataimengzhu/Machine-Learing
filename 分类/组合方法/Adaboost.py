#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import Edge as Edge
import CleanData as clean
# import ROC as roc
import SVMclassify as SVM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
class Adaboost:
    def __init__(self,features,labels,learningRate=1):#features：特征值，m*n矩阵,labels:标签值，m*1矩阵
        self.X,self.Y = features,labels#m*n,m*1
        self.M,self.N = self.X.shape#样本维度
        self.learningRate = learningRate#给每个弱分类器一个衰减，避免过拟合
        self.dataDic = {1:[],-1:[]}#数据字典
        for i in range(self.M):
            self.dataDic[int(self.Y[i])].append(self.X[i,:].tolist()[0])
        self.W = np.mat([1/self.M]*self.M).T#初始化样本权重集合均为1/m
        self.allClassify = []#用来存储所有的弱分类器对象，记录每次迭代产生的分类器属性：方向、阈值、大于/小于等于，分类器权重
        self.error = None#错误率
    def baseClassify(self,n,th):#单层决策树：大于阈值或小于等于阈值的样本的标签为-1其余为1。默认采用小于等于阈值的样本的标签为-1分类方式
        reLabels = np.ones((self.M, 1))#初始化标签值均为1
        style = 'lr'#小于等于
        reLabels[self.X[:,n]<=th]=-1#第n个维度下，小于等于阈值的样本标签值记为-1
        ratioMat = np.ones((self.M,1))#
        ratioMat[reLabels==self.Y]=0#:样本是否分类错误二值集合:错分类：1，正确分类：0
        errorRatio = (self.W.T*ratioMat)[0,0]#错误率：
        if errorRatio>0.5:#，若此时其错误率超过0.5,则说明大于阈值阈值的样本的标签为-1时的错误率相对更低，则采用大于阈值的分类方式
            style = 'dy'#大于
            reLabels = np.ones((self.M, 1))
            reLabels[self.X[:, n] > th] = -1
            ratioMat = np.ones((self.M, 1))
            ratioMat[reLabels == self.Y] = 0  #:样本是否分类错误二值集合:错分类：1，正确分类：0
            errorRatio = (self.W.T * ratioMat)[0, 0]

        return reLabels,[n,style,th],errorRatio#返回当前阈值、维度、方式下的标签值，错误率
    def ratio(self,predict,inY,model):#predict:矩阵：m*1,inY:矩阵：m*1，返回：正确率/错误率
        # ratio = sum((inY - predict).tolist(), []).count(0) / inY.shape[0]
        ratio = np.sum(inY==predict) / inY.shape[0]
        return ratio if model else 1-ratio
    def selectClassify(self):
        numSteps = 10.0
        minError = np.inf
        bestPredict,parClassify = None, None#初始化预测类别、弱分类器
        for n in range(self.N):#每一次迭代都要遍历所有维度+所有阈值+分类方式(dy/lr)
            # 样本点值域分割值作为阈值
            domian = [min(self.X[:,n])[0,0],max(self.X[:,n])[0,0]]#阈值范围
            stepSize = (domian[1]-domian[0])/numSteps
            for i in range(-1,int(numSteps)+1):#遍历阈值在给定本次样本权重的情况下，计算各个阈值的样本分类错误率
                th = domian[0]+stepSize*i
                prLabels, thNratio,errorRatio = self.baseClassify(n,th)#计算每个维度下的每个阈值的分类情况
                if errorRatio<minError:#获取最小错误率下的分类方式(维度、阈值、分类方式、当前错误率)+标签值
                    minError,bestPredict,parClassify = errorRatio,prLabels,thNratio
            # 样本点自身作为阈值
            # tmp0 = sum((self.X[:,n]).tolist(),[])
            # thList = set(tmp0)
            # for th in thList:#遍历阈值
            #     prLabels, thNratio,errorRatio = self.baseClassify(n,th)#计算每个维度下的每个阈值的分类情况
            #     if errorRatio<minError:#获取最小错误率下的分类方式(维度、阈值、分类方式、当前错误率)+标签值
            #         minError,bestPredict,parClassify = errorRatio,prLabels,thNratio
        return minError,bestPredict,parClassify
    def train(self,iters=40):#训练
        predict = np.mat([0.0]*self.M).T#预测值初始化
        for i in range(iters):#每一次迭代，产生一个分类器(维度，阈值，分类方式，分类器权重)
            minError,prLabels,parClassify = self.selectClassify()#筛选出当前样本权重下的最优分类器，每一次迭代都要遍历所有维度+所有阈值+分类方式(dy/lr)
            alpha = self.learningRate*0.5*np.log((1.0-minError)/minError)#计算分类器权重
            parClassify.append(alpha)#添加分类器权重，用于计算加权线性组合
            self.allClassify.append(parClassify)#记录各个分类器属性
            #更新样本权重集合：重新分配样本权重：提高错分样本的权重，这样在selectClassify(W)在筛选哪个分类器是最好的，
            # 会忽略权重低的样本的分类结果(无论是否正确分类)，只关注错分样本情况，如(5*1,5*1,0*1,0*1,0*0)==(5*1,5*1),忽略权重低的样本
            # 相当于权重低的样本不参与运算
            tmp = -1*np.multiply(prLabels,self.Y)*alpha#m*1
            W = np.multiply(self.W,np.exp(tmp))#m*1
            #归一化
            self.W = W/np.sum(W)
            #用于计算所有若分类器线性加权组合的结果
            predict += prLabels*alpha#第i个分类器的预测结果 c1:[y11,y12,y13,...,y1n]+c2:[y21,y22,y23,...,y2n]...
            #对于特征x1：预测值=y11+y21+...+yi1,i:迭代次数，yi1=gi(x1)*alphai
            self.error = [i,self.ratio(np.sign(predict),self.Y,False)]#迭代第i次后的训练集错误率
            # self.draw('2d',True)
            if self.ratio(np.sign(predict),self.Y, False)==0:#若迭代完之前错误率就为0的话，那么就停止迭代
                break
    def predict(self,inX,inY=None):#测试，输出错误率。inX:测试集特征：矩阵，m*n，inY:测试集标签：矩阵，m*1
        m = inX.shape[0]#测试样本维度
        predict = np.mat([0.0] * m).T#初始化预测值
        for da in self.allClassify:#遍历分类器，计算各个分类器的预测结果，并累加求和预测值
            prLabels = np.ones((m, 1))
            if da[1] == 'lr':
                prLabels[inX[:, da[0]] <= da[2]] = -1
            else:
                prLabels[inX[:, da[0]] > da[2]] = -1
            predict += prLabels*da[-1]
        # predict = np.sign(predict)
        # predict /= len(self.allClassify)
        errorRatio = None if inY is None else self.ratio(np.sign(predict), inY, False)
        return [predict,errorRatio]
    def draw(self,model,show=False):#用于2维特征样本可视化
        if show:
            plt.ion()#动态显示
            plt.clf()#清屏
        #设置网格点
        axis = np.linspace(min(self.X[:,0]).getA()[0][0]-10,max(self.X[:,0]).getA()[0][0]+10,30)
        ayis = np.linspace(min(self.X[:, 1]).getA()[0][0]-10,max(self.X[:, 1]).getA()[0][0]+10,30)
        # axis,ayis元素两两组合，获得网格点坐标
        X,Y = np.meshgrid(axis,ayis)
        # ravel()降维函数，把每一行都放在第一行，vstack能够将多个结构一致的一维数组按行堆叠起来
        XY = np.vstack([X.ravel(), Y.ravel()]).T## xy就是形成的网络，遍及在画布上密集的点
        fig = plt.figure(1)
        colors = np.where(self.Y == 1, 'r', 'b')
        colors = sum(colors.tolist(),[])
        if model == '2d':
            plt.title('分类')
            # 设置X轴标签
            plt.xlabel('x1')  # 设置X/Y轴标签是在对应的figure后进行操作才对应到该figure
            # 设置Y轴标签
            plt.ylabel('x2')
            plt.xlim(min(self.X[:, 0]) - 1, max(self.X[:, 0]) + 1)
            plt.ylim(min(self.X[:, 1]) - 1, max(self.X[:, 1]) + 1)
            plt.text(min(self.X[:,0]), max(self.X[:, 0]), '第'+str(self.error[0]+1)+'次迭代后分类错误率 : '+str(round(self.error[1],3)), fontsize=15)
            # 绘制等高线
            # 输入每个新样本到阈值边界的'距离'
            predict = self.predict(XY)[0]
            predict = predict.reshape(X.shape)
            # plt.contourf(X, Y, predict, levels=[-1, 0, 1], alpha=0.5)
            plt.pcolormesh(X, Y, np.sign(predict), alpha=0.5,cmap='jet')  # 预测值的显示。颜色：jet/gray
            # 绘制散点
            # for key, values in self.dataDic.items():
            #     values = np.array(values)
            #     plt.scatter(values[:, 0], values[:, 1])
            plt.scatter(np.array(self.X[:, 0]), np.array(self.X[:, 1]),c=colors)

        else:
            ax3d = Axes3D(fig)
            # 设置轴标签
            ax3d.set_xlabel('X1')
            ax3d.set_ylabel('X2')
            ax3d.set_zlabel('Y')
            #设置视角
            ax3d.view_init(elev=20, azim=45)  # 仰角、 方位角
            #绘制散点
            predict = self.predict(self.X)[0]
            ax3d.scatter(np.array(self.X[:, 0]), np.array(self.X[:, 1]), predict, s=20, cmap="jet", c=colors)
            ax3d.plot_trisurf(sum(self.X[:, 0].tolist(),[]), sum(self.X[:, 1].tolist(),[]), sum(predict.tolist(),[]),
                              alpha=0.5)
            #绘制等高线3D轮廓图
            # 输入每个新样本到阈值边界的'距离'
            # predict = self.predict(self.X)[0]
            # ax3d.plot_trisurf(self.X[:, 0],self.X[:, 1], predict)
        plt.show()
        if show:#是否动态显示
            plt.pause(0.001)
    def plotROC(self,predStrengths,fit = False):#predStrengths：分类器预测强度,m*1，self.Y：真实类别
        numPosClas = np.sum(self.Y == 1.0)  # 统计正类的数量
        numNegClas = len(self.Y) - numPosClas# 统计负类的数量
        sortedIndex = (-predStrengths.T).argsort() # 预测强度排序,从高到低
        numP,numN,AUCH,AUCW = 0,0,0,1/numNegClas
        coordinate =[]
        for index in sortedIndex.tolist()[0]:
            if self.Y[index]==1.0:#向上位移
                numP += 1
                TPR = numP/numPosClas
                FPR = numN / numNegClas
            else:#向右位移
                numN += 1
                FPR = numN/numNegClas
                TPR = numP / numPosClas
                AUCH += TPR
            coordinate.append([FPR,TPR])
        AUC = AUCH*AUCW
        coordinate = np.mat(coordinate)
        #凸包拟合
        plt.figure(10)
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
def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()#包含换行符、制表符，每个元素为字符串
    #数据清洗：去换行符、通过制表符分割为列表、列表中每一个字符串元素转换为浮点数
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    dataSet = np.mat(baseData)[:, :-1]
    # 数据清洗：列表中每一个浮点数元素转换为整数：标签
    labels = np.mat(list(map(int,np.array(baseData)[:,-1]))).T
    return dataSet, labels
def sklearnData(model,samplesNum=100,noise=None):#sklearn数据集
    from sklearn import datasets
    if model=='moons':
        dataSet,labels = datasets.make_moons(n_samples=samplesNum, noise=noise, random_state=3)
    elif model=='circles':
        dataSet, labels = datasets.make_circles(n_samples=samplesNum, noise=noise, factor=0.5, random_state=3)
    elif model=='blobs':
        dataSet, labels = datasets.make_blobs(n_samples=samplesNum, centers=4,cluster_std=0.60, random_state=0)
    elif model=='classification':
        dataSet, labels = datasets.make_classification(n_samples=samplesNum,n_features=2,n_informative=2,n_redundant=0, random_state=1)
    elif model=='gaussian_quantile':
        dataSet, labels = datasets.make_gaussian_quantiles(n_samples=samplesNum, n_classes=2, random_state=500)
    dataSet, labels = np.mat(dataSet),np.mat(labels).T
    labels[labels != 1] = -1
    return dataSet, labels

if __name__ == '__main__':
    # ******加载数据集**********
    trainData = 'testSet.txt'#testSet.txt/horseColicTraining2.txt/ex00
    rawDataSet, rawLabels = loadData('./Data/' + trainData)#m*n,m*1
    # testData = 'horseColicTest2.txt'  # testSet.txt/horseColicTraining2.txt/horseColicTest2.txt
    # testDataSet, testLabels = loadData('./Data/' + testData)
    # rawDataSet, rawLabels = sklearnData('blobs',200,0.1)#moons/circles/blobs/classification/gaussian_quantile
    # rawDataSet = np.mat([[1.,2.1],[1.5,1.6],[1.3,1.],[1.,1.],[2.,1.]])
    # rawLabels = np.mat([1,1,-1,-1,1]).T
    # ******数据预处理*********
    data = clean.CleanData(rawDataSet, rawLabels)  # 实例化
    # data.ascend(3)  # 数据集升维处理:普通高阶/组合高阶,用于多项式回归
    # data.zero_centered('alone')# 中心化
    # data.z_score('alone')#标准正态分布
    data.normData('alone')  # 归一化
    rawDataSet, rawLabels= data.X, data.Y
    #************Adaboost****************
    ad = Adaboost(rawDataSet, rawLabels,1)
    iteration = 40
    ad.train(iteration)
    ad.draw('2d',False)
    result1 = ad.predict(rawDataSet, rawLabels)
    ad.plotROC(result1[0])
    print('训练集错误率：{:.2%}'.format(float(result1[1])))
    # result = ad.predict(testDataSet, testLabels)
    # print('测试集错误率：', result[1])
    # ************SVM****************
    Kernel = ['linear',None]  # ['linear',None]/['rbf',3]/['poly',[1/DataSet.shape[0],3]]]/['sigmoid',[1/DataSet.shape[0],0]]#核函数
    C = float('inf')# float('inf')/0.6 惩罚因子
    svm = SVM.SVM(rawDataSet, rawLabels,Kernel,C)
    svm.train(0)
    svm.sklearn('linear')
    result2 = svm.predict(rawDataSet)[0]
    svm.plotROC(result2)
