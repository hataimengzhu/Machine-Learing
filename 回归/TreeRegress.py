#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import CleanData as clean
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def standRegress(inXY):  # 标准回归函数，回归系数RC=(X.T*X)^(-1)*X.T*Y
    # 每个样本点的权重都相同，即每个样本点对应的回归系数都相同-->统一的回归系数
    # 由于具备统一的回归系数-->相同的参数-->只需训练一次，对于之后的预测，不再需要使用原始训练数据集
    # 参数学习算法
    y = inXY[:, -1]
    x= inXY[:, 0:inXY.shape[1] - 1]
    x = np.insert(x,x.shape[1],values=np.ones(x.shape[1]),axis=1)
    xTx = x.T * x
    if np.linalg.det(xTx) != 0:  # 是否可逆,行列式==0？
        RC = xTx.I * x.T * y
        # preY = x * RC
        return RC, x,y
    else:
        print('矩阵不可逆')
#回归树
class RegressTree:
    def __init__(self, features, outValue):  # features:特征，m*n,outValue:输出值,m*1
        self.oraX = features
        # self.X = np.insert(self.oraX, self.oraX.shape[1], values=np.ones((1, self.oraX.shape[0])), axis=1)  # 特征+位移量，m*(n+1)
        self.X = features
        self.M,self.N = self.X.shape
        self.Y = outValue
        self.XY = np.insert(self.X, self.X.shape[1], values=self.Y.T, axis=1)
        self.Tree = None  # 决策树
    #回归树-方法：计算样本均值、计算总方差
    @staticmethod
    def regLeaf(inXY):
        return inXY[:,-1].mean()
    @staticmethod
    def regErr(inXY):
        return np.var(inXY[:,-1])*inXY.shape[0]
    #模型树-方法：计算样本回归系数、计算残差平方和
    @staticmethod
    def modelLeaf(inXY):
        RC,x,y = standRegress(inXY)
        return RC
    @staticmethod
    def modelErr(inXY):
        RC, x,y = standRegress(inXY)
        preY = x*RC
        return sum(np.power(y-preY,2))
    #分割数据-二元切分数据
    def splitData(self,inXY,inV):
        lXY = inXY[sum((inXY[:,0] <= inV).tolist(), []), :]
        rXY = inXY[sum((inXY[:,0]> inV).tolist(), []), :]
        return [lXY,rXY]
    #选择最佳切分点
    def nextNode(self,inXY,treeType='constant',tol=[1,4]):#treeType:constant/linearFun:回归树/模型树,tol=[误差，样本切分数]
        #创建回归树还是模型树(分段函数)
        if treeType == 'constant':#回归树
            leafType = self.regLeaf
            errType = self.regErr
        elif treeType == 'linearFun':#模型树
            leafType = self.modelLeaf
            errType = self.modelErr
        # 如果输入数据的样本数小于2*tol[1],则不切分，直接返回样本均值常数或回归系数
        if inXY.shape[0]<=2*tol[1] or len(set(sum(inXY[:,-1].tolist(),[])))==1:
            return None,leafType(inXY)
        #遍历样本(单维特征数据，后续改为面向多维特征数据)，寻找最佳切分点：bestI
        minVar = np.inf#误差初始化
        for i in range(inXY.shape[0]):
            value = inXY[i,0]
            #切分数据：subXY=[左节点数据，右节点数据]
            subXY = self.splitData(inXY, value)
            #如果分割后的数据太少，则跳过
            if subXY[0].shape[0]<tol[1] or subXY[1].shape[0]<tol[1]:
                continue
            #计算总方差/残差平方和
            var = errType(subXY[0])+errType(subXY[1])
            #如果误差估计更小,则更新特征索引值
            if var<minVar:
                minVar = var
                bestI = i
        #遍历完样本获得的最佳切分数据，如果切分的数据的误差减少不大则退出
        if abs(errType(inXY)-minVar)<tol[0]:
            return None,leafType(inXY)
        #返回最佳切分点及其对应的输出值
        return inXY[bestI,0],inXY[bestI,-1]
    #创建回归树/模型树
    def train(self,treeType='constant',tol=[1,4]):#treeType:constant/linearFun:回归树/模型树
        #递归
        def fun(inXY):
            #选择最佳切分点
            nodeI,nodeValue = self.nextNode(inXY,treeType,tol)
            if nodeI is None:
                return nodeValue
            #父节点:索引+对应的输出值/左右节点，{'No': 4.0, 4.38: {'l': [[0.018],[4.255]], 'r': [[0.608],[3.22]]}}
            preTree = {'No':nodeI,nodeValue:{}}
            #根据最佳切分点对样本切分
            subXY = self.splitData(inXY, nodeI)
            #子节点
            preTree[nodeValue]['l'] = fun(subXY[0])
            preTree[nodeValue]['r'] = fun(subXY[1])
            #返回当前树
            return preTree
        self.Tree = fun(self.XY)

    def prune(self,inXY):#剪枝
        pass
    #预测
    def predict(self,inX,inTree):
        # 递归回归树，判断键值
        # {'No': 4.0, 4.38: {'l': [[0.018], [4.255]], 'r': [[0.608], [3.22]]}}
        tree = iter(inTree)#inTree字典转换成迭代器
        firstNo = next(tree)#使用next函数访问迭代器对象中的元素
        firstNode = next(tree)#使用next函数访问迭代器对象中的元素
        nextDic = inTree[firstNode]

        if inX <= inTree[firstNo]:#是否在左节点
            if type(nextDic['l']).__name__=='dict':#判断是否为树
                predict = self.predict(inX,nextDic['l'])#若是树则继续遍历
            else:#直接赋值
                predict = nextDic['l']
        else:#是否在右节点
            if type(nextDic['r']).__name__=='dict':#判断是否为树
                predict = self.predict(inX,nextDic['r'])#若是树则继续遍历
            else:#直接赋值
                predict = nextDic['r']
        return predict

    def draw(self):
        plt.figure(1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(self.X[:, 0].A, self.Y.A, s=20)  # 绘制散点
        #可视化回归线
        xx = np.linspace(min(self.X)[0,0],max(self.X)[0,0], 50)
        yy =[]
        for i in xx:
            re = self.predict([i],self.Tree)
            if type(re).__name__=='matrix':# {'No': 4.0, 4.38: {'l': [[0.018], [4.255]], 'r': [[0.608], [3.22]]}}
                prey = (np.mat([i,1]) * re)[0,0]#根据回归系数，计算预测值
                yy.append(prey)
            else:#{'No': 5.0, 5.8: {'l': 4.6, 'r': 8.175999999999998}}
                yy.append(re)
        # yy = [self.predict([i],self.Tree) for i in xx]
        plt.plot(xx,yy,c='r')
        plt.show()

def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()  # 包含换行符、制表符，每个元素为字符串
    # 数据清洗：去换行符、通过制表符分割为列表、列表中每一个字符串元素转换为浮点数
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    dataSet = np.mat(baseData)[:, :-1]
    # 数据清洗：列表中每一个浮点数元素转换为整数：标签
    outValue = np.mat(baseData)[:, -1]
    return dataSet, outValue

if __name__=='__main__':
    # ******加载数据集**********
    trainData = 'ex3.txt'  # ex0/horseColicTraining2/abalone/ex00
    RawDataSet, OutValue = loadData(r'./Data/' + trainData)  # m*n,m*1
    # ******数据预处理*********
    data = clean.CleanData(RawDataSet, OutValue)  # 实例化
    # data.ascend(3)  # 数据集升维处理:普通高阶/组合高阶,用于多项式回归
    # data.zero_centered('all')  # 中心化
    # data.z_score('all')  # 标准正态分布
    # data.normData('all')  # 归一化
    # RawDataSet, OutValue= data.X, data.Y
    #**********回归树*************5
    xx = RegressTree(RawDataSet, OutValue)#实例化
    Model = 'linearFun'#constant/linearFun 回归树/模型树
    xx.train(treeType=Model,tol=[1,4])
    print(xx.Tree)
    xx.draw()
