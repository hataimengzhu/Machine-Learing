#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import time
import copy
import random
# import 贝塞尔曲线 as bezier
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
'''
    # DataSet, Labels = loadData('./Data/' + 'testSetRBF.txt')
    xx = Edge(DataSet, Labels)
    xx.margin()
    xx.display(0)
'''
class Edge:
    def __init__(self,features,labels=None):
        self.X = features
        self.M,self.N = features.shape
        if labels is None:
            self.Y =np.mat(np.ones((features.shape[0],1)))
        else:
            self.Y = labels
        self.DataDic = {}
        for i in range(len(self.Y)):
            self.DataDic[self.Y[i].A[0][0]]=[]
        for i in range(len(self.Y)):
            self.DataDic[self.Y[i].A[0][0]].append(features[i,:].tolist()[0])
        self.K = len(self.DataDic.keys())#类别个数
        self.Centers = {}#类别中心点
        self.clusterHead()  # 计算各个类别中心点
        self.firstPoint = {}
        self.boundaryPoint = {}#边界点(外边)
        self.curve = {}#拟合曲线点集合
        for key in self.DataDic.keys():
            self.boundaryPoint[key] = []#边界点初始化
    def clusterHead(self):#获得各个簇的中心点
        for key in self.DataDic.keys():
            self.Centers[key]=[]#类别中心点初始化
            for i in range(self.N):
                self.Centers[key].append(np.mean(np.mat(self.DataDic[key])[:,i]))
    @staticmethod
    def endPoint(inx,minxy,maxxy):#该类中定义的该函数并没有使用self相关的变量，因此可以把此函数设为静态方法->去掉函数定义的self，并在函数定义的上一行输入@staticmethod
        if (inx == minxy).all():
            point = np.mat([inx[0,0]-1,inx[0,1]])
        elif (inx == maxxy).all():
            point = np.mat([inx[0, 0] + 1, inx[0, 1]])
        else:
            k = (minxy-maxxy)[0,1]/(minxy-maxxy)[0,0]
            b = minxy[0,1]-k*minxy[0,0]
            if inx[0,1]<=inx[0,0]*k+b:
                point = np.mat([inx[0, 0], inx[0, 1]-1])
            else:
                point = np.mat([inx[0, 0], inx[0, 1] + 1])
        return point
    def alterTheta(self,key,tmp,start,end,model,left=None, right=None):
        if model == '最大凸边-外边':
            if (start == left).all():  # 向左
                alterJ = np.where(tmp[:, 1] > start[0, 1])[0]  # 手背方向的点
            elif (start == right).all():  # 向右
                alterJ = np.where(tmp[:, 1] <= start[0, 1])[0]  # 手背方向的点
            elif start[0, 1] > end[0, 1]:  # 向下
                alterJ = np.where(tmp[:, 0] < start[0, 0])[0]  # 手背方向的点
            else:  # 向上
                alterJ = np.where(tmp[:, 0] >= start[0, 0])[0]  # 手背方向的点
        elif model == '最小凹边':
            k = (start - end)[0, 1] / (start - end)[0, 0]
            b = start[0, 1] - k * start[0, 0]
            if k==0 and end[0, 0]>self.Centers[key][0,0]:#时针向量平行
                alterJ = np.where(tmp[:, 1] < start[0, 1])[0]  # 手背方向的点
            elif k==0 and end[0, 0]<self.Centers[key][0,0]:#时针向量平行
                alterJ = np.where(tmp[:, 1] > start[0, 1])[0]  # 手背方向的点
            elif not np.isfinite(k) and end[0, 1]>self.Centers[key][0,1]:#垂直
                alterJ = np.where(tmp[:, 0] > start[0, 0])[0]  # 手背方向的点
            elif not np.isfinite(k) and end[0, 1] < self.Centers[key][0, 1]:  # 垂直
                alterJ = np.where(tmp[:, 0] < start[0, 0])[0]  # 手背方向的点
            elif k<0:
                if end[:, 1]<start[0, 1]:#yB<yA
                    alterJ = np.where(tmp[:, 1] < tmp[:, 0] * k + b)[0]  # 手背方向的点
                else:#yB>=yA
                    alterJ = np.where(tmp[:, 1] > tmp[:, 0] * k + b)[0]  # 手背方向的点
            else:
                if end[:, 1] < start[0, 1]:  # yB<yA
                    alterJ = np.where(tmp[:, 1] > tmp[:, 0] * k + b)[0]  # 手背方向的点
                else:#yB>=yA
                    alterJ = np.where(tmp[:, 1] < tmp[:, 0] * k + b)[0]  # 手背方向的点
        return alterJ

    def margin(self):#寻找时针时针旋转最小夹角对应的边界点(最大凸边-外边)
        start = time.time()  # 计时开始
        #*****初始化********#
        firstJ = {}#边界初始起点集合
        for key in self.DataDic.keys():
            tmp = np.mat(self.DataDic[key])#数据容器
            m = tmp.shape[0]#样本维度
            # firstJ[key] = np.argmax(np.sum(np.power(np.tile(self.Centers[key],(m,1))-tmp,2),axis=1))#获取边界起点序号
            maxY = max(tmp[:,1])
            maxYlistI = np.where(tmp[:, 1] == maxY)[0]
            maxX = max(tmp[maxYlistI,:][:,0])
            firstJ[key] = np.where((tmp[:,0]==maxX) & (tmp[:,1]==maxY))[0][0]

            # firstJ[key] = np.argmax(tmp[:,1])  # 获取边界起点序号
            self.firstPoint[key]=tmp[firstJ[key]]#获取边界起点坐标
            self.boundaryPoint[key].append(self.firstPoint[key].A.tolist()[0])  # 存储边界点
        # *****寻找每个类别的边界********#
        for key,values in self.firstPoint.items():#遍历每个类别
            tmp = np.mat(self.DataDic[key])#样本容器

            startPoint = values  # 边界起点赋值
            leftI, rightI = np.argmin(tmp[:, 0]), np.argmax(tmp[:, 0])  # 最左边点索引行,最右边点索引行
            left, right = tmp[leftI, :], tmp[rightI, :]  # 最左边点坐标,最右边点坐标
            endPoint = self.endPoint(startPoint, left, right)  # 获取时针向量终点B

            Vector1 = endPoint-startPoint#时针向量AB: B-A
            first = True#用于控制初始起点
            tmp = np.delete(tmp, firstJ[key], axis=0)#删除已经确认是边界的点
            # 逆时针/顺时针旋转：寻找夹角最小的边界点
            while first or (startPoint != values).all():
                Vector2 = tmp-np.tile(startPoint, (tmp.shape[0], 1))#边界向量集合
                # a·b=|a|*|b|cos(theta),注意:arccos()返回的夹角总是最小角
                denominator = np.linalg.norm(Vector1) * np.linalg.norm(Vector2,axis=1)
                numerator = Vector1*Vector2.T#(1*2)*((m*2).T)=1*m
                #theta = arccos(a·b/|a|*|b|),angel = (180/pi)*theta
                angle = np.rad2deg(np.arccos(numerator/denominator).T)
                #修改arccos()返回的夹角，使之返回的是逆时针方向的夹角
                alterJ = self.alterTheta(key,tmp, startPoint, endPoint, '最大凸边-外边',left, right)
                angle[alterJ] = 360 - angle[alterJ]
                #获取逆时针(argmin)/顺时针(argmax)旋转的最小夹角(最大凸边)对应的索引# int(angle[0]*180 /np.pi)
                startJ = np.argmax(angle)
                # 更新起点
                startPoint = tmp[startJ, :]
                # 选择下一次迭代的y轴方向: 只要遇到最左边或最右边的点就改变原方向为相反方向(上->下或者下->上),若不是最左边或最右边的点就不改变方向
                endPoint = self.endPoint(startPoint, left, right)
                # 删除已经确认是边界的点
                tmp = np.delete(tmp, startJ, axis=0)
                # 更新时针向量
                Vector1 = endPoint - startPoint
                #存储新的边界点
                self.boundaryPoint[key].append(startPoint.tolist()[0])
                if first:#只执行一次
                    first = False
                    tmp = np.insert(tmp, tmp.shape[0], values=values, axis=0)#重新添加初始起点，是为了判断最后的startPoint是否到达初始起点即到达终点，停止迭代
            # 凸边曲线拟合
            # print('类别'+str(key)+':凸边开始拟合曲线')
            # self.curve[key] = []
            # pp = copy.deepcopy(self.boundaryPoint[key])
            # for i in range(len(pp)-1):
            #     p0,p2 = pp[i],pp[i + 1]
            #     convex = True#凸拟合曲线还是凹拟合曲线
            #     p1 = bezier.f(p0, p2, self.Centers[key],left.tolist()[0], right.tolist()[0],convex)
            #     points = [p0, p1, p2]
            #     be = bezier.Bezier(points)#实例化
            #     be.run()
            #     self.curve[key].extend(be.curve)
            # self.curve[key].append(self.boundaryPoint[key][-1])
            #
            # self.curve[key] = np.array(self.curve[key])
        end = time.time()  # 计时结束
        print("获取凸边界点数据耗时：{}".format(end - start))
    def concave(self):#凸边变凹边
        listI = {}
        addPoint = {}
        for key,values in self.boundaryPoint.items():
            length = len(values) if any(isinstance(i,list) for i in values) else 1
            if length==len(self.DataDic[key])+1:
                continue
            listI[key] = []
            shuju = np.mat(self.DataDic[key])
            for i in range(length):
                equJ = np.where(shuju==np.mat(values[i]))[0]
                if len(equJ)>=1:
                    listI[key].append(equJ[0])
            listI[key].pop()#删除重复点(起点==终点)
            shuju = np.delete(shuju, listI[key], axis=0)
            tmp = self.boundaryPoint[key]
            if len(shuju)>=1:
                addPoint[key] = []
                for i in range(0,len(tmp)-1):
                    minX = min((tmp[i][0],tmp[i+1][0]))
                    maxX = max((tmp[i][0],tmp[i+1][0]))
                    minY = min((tmp[i][1],tmp[i+1][1]))
                    maxY = max((tmp[i][1],tmp[i+1][1]))
                    a = np.sum(np.power(np.tile(tmp[i],(len(shuju),1))-shuju,2),axis=1)
                    b = np.sum(np.power(np.tile(tmp[i+1],(len(shuju),1))-shuju,2),axis=1)
                    nearJ = np.argmin(a+b)#获取距离边界点a和边界点b最近的点
                    if minX<=shuju[nearJ,0]<= maxX or minY<=shuju[nearJ,1]<=maxY:#若这个最近点在点a、点b附近，则接收该点
                        addPoint[key].extend(shuju[nearJ,:].tolist())
                addPoint[key] = np.unique(addPoint[key], axis=0)
                tmp.extend(addPoint[key].tolist())
            if len(tmp)<=len(self.DataDic[key])+1:
                self.padding(key,tmp)  # 按序连点
                self.concave()#凸边凹化
    def padding(self,key,dataSet):  # 寻找时针时针旋转最小夹角对应的边界点(最小凹边/按序连点)+拟合曲线
        start = time.time()  # 计时开始
        #*****初始化********#
        shuju = {key: np.mat(np.unique(dataSet, axis=0))}
        m = shuju[key].shape[0]#样本维度
        # 获取起始边界序号
        firstJ = {key: np.argmax (np.sum(np.power(np.tile(self.Centers[key], (m, 1)) - shuju[key], 2), axis=1))}  #边界初始起点集合
        self.firstPoint[key]=shuju[key][firstJ[key]]#获取边界起点坐标
        #边界点容器
        self.boundaryPoint[key]=[]
        self.boundaryPoint[key].append(self.firstPoint[key].A.tolist()[0])  # 存储初始边界点
        # *****寻找每个类别的边界********#
        startPoint = np.mat(self.Centers[key])  ##边界起点赋值
        endPoint = self.firstPoint[key]  # 获取时针向量终点B
        Vector1 = endPoint-startPoint#时针向量AB: B-A
        first = True#用于控制初始起点
        shuju[key] = np.delete(shuju[key], firstJ[key], axis=0)#删除已经确认是边界的点
        # 逆时针/顺时针旋转：寻找夹角最小的边界点
        while first or (endPoint != self.firstPoint[key]).all():
            Vector2 = shuju[key]-np.tile(startPoint, (shuju[key].shape[0], 1))#边界向量集合
            # a·b=|a|*|b|cos(theta),注意:arccos()返回的夹角总是最小角
            denominator = np.linalg.norm(Vector1) * np.linalg.norm(Vector2,axis=1)
            numerator = Vector1*Vector2.T#(1*2)*((m*2).T)=1*m
            #theta = arccos(a·b/|a|*|b|),angel = (180/pi)*theta
            angle = np.rad2deg(np.arccos(numerator/denominator).T)
            #修改arccos()返回的夹角，使之返回的是逆时针方向的夹角
            alterJ = self.alterTheta(key,shuju[key],startPoint,endPoint, '最小凹边')
            angle[alterJ] = 360 - angle[alterJ]
            #获取逆时针(argmin)/顺时针(argmax)旋转的最小夹角(最大凸边)对应的索引# int(angle[0]*180 /np.pi)
            listJ = np.where(angle == max(angle))[0]
            if len(listJ) > 1:  # 夹角相同同向共线点做筛选
                J = np.argmin(np.sum(np.power(np.tile(startPoint, (len(listJ), 1)) - shuju[key][listJ], 2),axis=1))  # 获取边界起点序号
                endJ = listJ[J]
            else:
                endJ = listJ[0]
            # 更新起点
            endPoint = shuju[key][endJ, :]
            # 删除已经确认是边界的点
            shuju[key] = np.delete(shuju[key], endJ, axis=0)
            # 更新时针向量
            Vector1 = endPoint - startPoint
            #存储新的边界点
            self.boundaryPoint[key].append(endPoint.tolist()[0])
            if first:#只执行一次
                first = False
                shuju[key] = np.insert(shuju[key], shuju[key].shape[0], values=self.firstPoint[key], axis=0)#重新添加初始起点，是为了判断最后的startPoint是否到达初始起点即到达终点，停止迭代
            # self.display(1)
        end = time.time()  # 计时结束
        print("获取凹边界点数据耗时：{}".format(end - start))
        print('类别'+str(key)+':边界点数量:',len(self.boundaryPoint[key]))
        self.display(1)

    def display(self,oneZero=0):
        plt.figure(1)
        if oneZero:
            plt.ion()  # 动态显示
            plt.clf()  # 清屏
        plt.title('分类')
        # 设置X轴标签
        plt.xlabel('x1')  # 设置X/Y轴标签是在对应的figure后进行操作才对应到该figure
        # 设置Y轴标签
        plt.ylabel('x2')
        plt.xlim([min(self.X[:, 0]).getA()[0][0] - 2, max(self.X[:, 0]).getA()[0][0] + 2])
        plt.ylim([min(self.X[:, 1]).getA()[0][0] - 2, max(self.X[:, 1]).getA()[0][0] + 2])
        colors, markers = ['', 'b', 'r'], ['', 'o', 's']
        for key, values in self.DataDic.items():
            values = np.array(values)
            plt.scatter(values[:, 0], values[:, 1])# 绘制散点
            plt.scatter(self.Centers[key][0], self.Centers[key][1],marker='^')# 绘制中心点
            plt.scatter(self.firstPoint[key][0, 0], self.firstPoint[key][0, 1], marker='s', s=50)# 绘制起始点
            plt.plot(np.mat(self.boundaryPoint[key])[:, 0], np.mat(self.boundaryPoint[key])[:, 1])# 绘制边界
            # plt.plot(self.curve[key][:, 0], self.curve[key][:, 1])
            # plt.fill(np.mat(self.boundaryPoint[key])[:, 0], np.mat(self.boundaryPoint[key])[:, 1],color='y',alpha=0.5)# 填充多边形
        plt.show()
        if oneZero:
            plt.pause(1)
def loadData(fileName):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()#包含换行符、制表符，每个元素为字符串
    #数据清洗：去换行符、通过制表符分割为列表、列表中每一个字符串元素转换为浮点数
    baseData = [list(map(float, da.strip().split('\t'))) for da in baseData]
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
        dataSet, labels = datasets.make_circles(n_samples=samplesNum, noise=noise, factor=0.5, random_state=0)
    elif model=='blobs':
        dataSet, labels = datasets.make_blobs(n_samples=samplesNum, centers=3, random_state=1)
    elif model=='classification':
        dataSet, labels = datasets.make_classification(n_samples=samplesNum,n_features=2,n_informative=2,n_redundant=0, random_state=1)
    elif model=='gaussian_quantile':
        dataSet, labels = datasets.make_gaussian_quantiles(n_samples=samplesNum, n_classes=2, random_state=500)
    dataSet, labels = np.mat(dataSet),np.mat(labels).T
    labels[labels == 0] = -1
    return dataSet, labels

if __name__=='__main__':
    rawDataSet, rawLabels = loadData('./Data/' + 'testSet.txt')
    # rawDataSet, rawLabels = sklearnData('classification',0.1)#moons/circles/blobs/classification/gaussian_quantile
    # rawDataSet = np.mat(np.load('./Data/imtTest1.npy', allow_pickle=False))
    # rawLabels = np.mat([1]*rawDataSet.shape[0]).T
    xx = Edge(rawDataSet,rawLabels)
    xx.margin()#凸边
    xx.display()
    xx.concave()#凸边凹化


