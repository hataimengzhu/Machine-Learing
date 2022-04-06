#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import copy as copy
import CleanData as clean
import Edge as Edge
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class Regress:
    def __init__(self, features, outValue):  # features:特征，m*n,outValue:输出值,m*1
        self.oraX = features
        self.X = np.insert(self.oraX, self.oraX.shape[1], values=np.ones((1, self.oraX.shape[0])), axis=1)  # 特征+位移量，m*(n+1)
        self.M,self.N = self.X.shape
        self.Y = outValue
        self.RC = None  # 回归系数2*1 regression coefficient
        self.PreRC = []#记录每个样本点的回归系数(局部加权线性回归)
        self.RCIters = []#记录每次迭代后的回归系数
        self.RCList = []#用于回归系数与log(λ)的关系图
        self.lamList = []#log(λ) ridgeRegress()
        self.preY = None#预测值
        self.Loss = []#记录每次迭代后的代价
        self.LossList = []#用于平方误差与log(λ)的关系图
        self.iters = 0
    # 评估函数：计算预测值和真实值的匹配程度:相关系数
    def evaluation(self,model):
        if self.preY is not None:
            if model=='相关性':
                correlation = np.corrcoef(self.preY.T, self.Y.T)
                print('相关性：',correlation)
                return correlation
            elif model=='平方误差':
                error = np.power((self.Y-self.preY),2).sum()[0,0]
                print('平方误差:',error)
                return error
            elif model =='回归系数、平方误差与log(λ)关系':
                if len(self.lamList)>0 and len(self.LossList)>1:
                    fig = plt.figure(12)
                    ax1 = fig.add_subplot(111)
                    ax1.set_xlabel('log(λ)')
                    ax1.set_ylabel('w')
                    ax1.set_title('回归系数、平方误差与log(λ)关系')
                    ax1.plot(self.lamList, self.RCList)

                    ax2 = ax1.twinx()
                    ax2.set_ylabel('RSS')
                    ax2.plot(self.lamList,self.LossList,'g--')
                    # ax2.plot(self.lamList, np.zeros((len(self.LossList),1)), 'k--')
                    plt.show()
                else:
                    print('self.lamList is 空')
            elif model=='平方误差与log(λ)关系':
                if len(self.LossList)>1:
                    plt.figure(14)
                    plt.xlabel('log(λ)')
                    plt.ylabel('RSS')
                    plt.plot(self.lamList,self.LossList)
                    plt.show()
                else:
                    print('未迭代')
            elif model =='平方误差、回归系数与迭代次数关系':
                if len(self.RCIters)>1 and len(self.LossList)>1:
                    fig=plt.figure(13)
                    ax1 = fig.add_subplot(111)
                    ax1.set_xlabel('迭代次数')
                    ax1.set_ylabel('w')
                    ax1.set_title('迭代次数、平方误差回归系数关系')
                    plt.plot(np.mat(self.RCIters))

                    ax2 = ax1.twinx()
                    ax2.set_ylabel('RSS')
                    ax2.plot(self.LossList,'g--')

                    plt.show()
                else:
                    print('未迭代')
        else:
            print('预测值为None')
    # 标准回归函数-线性回归(最小二乘法):适用于数据集较小即计算量小，可以一步求出回归系数W
    def standRegress(self):  # 标准回归函数，回归系数RC=(X.T*X)^(-1)*X.T*Y
        #每个样本点的权重都相同，即每个样本点对应的回归系数都相同-->统一的回归系数
        #由于具备统一的回归系数-->相同的参数-->只需训练一次，对于之后的预测，不再需要使用原始训练数据集
        #参数学习算法
        xTx = self.X.T * self.X
        if np.linalg.det(xTx) != 0:#是否可逆,行列式==0？
            # self.RC = np.linalg.inv(self.X.T * self.X) * self.X.T * self.Y
            self.RC = xTx.I * self.X.T * self.Y
            self.preY = self.X * self.RC
        else:
            print('矩阵不可逆')
            return
    # 标准回归函数-多项式回归(样本升维):适用于解决欠拟合问题
    def polynomialRegress(self,n):
        data = clean.CleanData(self.oraX, self.Y)  # 实例化
        data.ascend(n,True)  # 数据集升维处理:普通高阶,组合高阶,用于多项式回归
        self.X,self.Y = data.X,data.Y
        self.standRegress()
    # 局部加权线性回归
    def LWLRegress(self,inX=None,theta=0.01,beta=10,eta=1,k=0.01): #回归系数W=(X.T*X)^(-1)*X.T*Y,inX：输入特征
        # 每个样本点的权重都不相同，即每个样本点对应的回归系数都不同-->没有一个统一的回归系数
        # 由于需要为每个样本点计算一个权重，所以算法不具备统一的回归系数-->没有固定的参数-->每次进行预测都需要全部的训练集
        #非参数算法
        if inX is None:
            inX = self.X
        # k = int(self.M * k / 100)  # 取百分比
        for i,values in enumerate(inX):
            #高斯核选取最近点：距离越近，对应的权重越大,反之越接近0
            dif = values- self.X  # 第i个样本的"距离"集合
            #高斯核
            perW = np.diag(np.exp(np.power(dif,2).sum(1)/(-2.0*theta **2)).A.T[0])#第i个样本点的权重。距离越近，对应的权重越大,反之越接近0
            #拉普拉斯核
            # perW = np.diag(np.exp(np.power(dif, 2).sum(1) / (-2.0 * theta)).A.T[0])  # 第i个样本点的权重。距离越近，对应的权重越大,反之越接近0
            #多项式核：不适合线性分布数据拟合
            # perW = np.diag(np.power((np.multiply(values,self.X).sum(1)+eta),beta).A.T[0])
            #sigmoid:不适合线性分布数据拟合
            # perW = np.diag(np.tanh(np.multiply(k*values,self.X).sum(1)-theta).A.T[0])
            #KNN:选取前k个近邻点对应的对角矩阵元素为1，其他都为0
            # sortedIndex = np.power(dif,2).sum(1).T.argsort()
            # sortedIndex = sortedIndex.tolist()[0][:k]
            # dif = np.mat(np.zeros((self.M,1)))
            # dif[sortedIndex] = np.mat(np.ones((k, 1)))
            # perW = np.diag(sum(dif.tolist(),[])) # 第i个样本点的权重。k个对角矩阵元素为1
            # # perW = np.diag([1]*self.M)
            #每次预测时都需要整个训练集参与
            xTwX = self.X.T *(perW* self.X)
            if np.linalg.det(xTwX)!=0:
                preRc = (xTwX.I * self.X.T *perW*self.Y).T
                self.PreRC.append(preRc.tolist()[0])#第i个样本点对应的回归系数
            else:
                print('矩阵不可逆')
                return
        self.PreRC = np.mat(self.PreRC)
        self.preY = np.multiply(inX,self.PreRC).sum(1)#每个样本点的预测值集合
    # 梯度下降+坐标下降：适用于数据集较大，需要迭代求出最终的回归系数W
    def GD_CD(self, model, maxIters=1000, alpha_int=1e-3):  #model='梯度下降'/'坐标下降' maxIters：最大迭代次数,alpha：学习率
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        def run():
            self.RC = np.mat(np.zeros((self.N, 1)))#回归系数初始化
            minLoss = np.inf#代价
            iters = 0#用于控制程序迭代次数
            while True:
                iters += 1
                oldLoss = costs(self.RC)[0, 0]
                alpha = alpha_int
                RC = copy.deepcopy(self.RC)
                if model[0]=='梯度下降':#按照梯度方向更新回归系数
                    if model[1]=='批量梯度下降':#每次迭代先依据全体样本的误差结果更新回归系数
                        # 计算回归系数方式1：计算完一个就更新一个
                        # for j in range(self.N):
                        #     alpha = alpha_int * (1.0 + j + iters)
                        #     RC[j] = RC[j] - alpha / self.M * (self.X * RC - self.Y).T * self.X[:, j]
                        # 计算回归系数方式：都计算完后在更新
                        alpha = alpha_int*(1.0+iters)/self.M#
                        error=self.X * RC - self.Y#误差:全样本
                        RC = RC- alpha*(self.X.T*error)# rc-alpha/len*(m*[n*1])
                    elif model[1]=='随机梯度下降':#每次迭代先依据某个样本的误差结果更新回归系数
                        for i in range(self.M):
                            alpha = alpha_int*(1.0+i+iters)
                            error = (self.X[i]*RC)-self.Y[i]#误差:第i个样本
                            RC = RC-alpha/self.M*self.X[i,:].T*error
                    elif model[1]=='小批量梯度下降':#每次迭代先依据某些样本的误差结果更新回归系数
                        start = 0
                        for i in range(10,self.M,10):
                            alpha = alpha_int*(1.0+i+iters)
                            error = (self.X[start:i]*RC)-self.Y[start:i]#误差:第start~i个样本
                            RC = RC-alpha/self.M*self.X[start:i].T*error
                            start = i
                elif model[0]=='坐标下降':#按照样本特征更新回归系数
                    RCtmp = np.mat(np.zeros((self.N, 1)))
                    for j in range(self.N):
                        RCcut, Xcut = copy.deepcopy(RC), copy.deepcopy(self.X)  # 复制，防止改变原对象
                        RCcut = np.delete(RCcut, j, axis=0)
                        Xcut = np.delete(Xcut, j, axis=1)
                        # 0 = self.X[:,j].T*(Xcut*RCcut-self.Y)+wj*np.power(self.X[:,j],2).sum()
                        wj = -self.X[:, j].T * (Xcut * RCcut - self.Y) / (self.X[:, j].T * self.X[:, j])
                        #计算回归系数方式1： 计算完一个就更新一个
                        RC[j,:] = wj
                    #计算回归系数方式2：都计算完后在更新
                    #     RCtmp[j,:] = wj
                    # RC = RCtmp
                #计算更新后代价
                newLoss = costs(RC)[0,0]
                #***************控制迭代次数+更新*****************
                if newLoss <= minLoss:
                    minLoss = newLoss
                    self.RC = RC#更新回归系数
                    self.RCIters.append(sum(self.RC.tolist(),[]))#记录每次迭代后的回归系数
                    self.LossList.append(newLoss)#记录每次迭代后的代价
                    self.preY = self.X * RC  # 计算当前回归系数下的预测值
                    print('迭代次数：', iters, '代价：', newLoss)
                else:#若大于，则说明跳过了局部最优解，退出
                    break
                if np.abs(newLoss - oldLoss) < 1e-3 or iters >= maxIters:#变化不大或超过最大迭代次数则退出
                    break
                #***************动态显示*****************
                # self.draw(True)
                self.iters+=1
                fig = plt.figure(1)
                #最大化显示
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()

                plt.ion()  # 动态显示
                plt.clf()  # 清屏

                left = fig.add_subplot(121)
                left.set_xlabel('X')
                left.set_ylabel('Y')
                left.set_title('回归')
                left.scatter(self.X[:, 0].A, self.Y.A, s=1)  # 绘制散点
                # 为什么需要排序？因为plot默认是顺序绘制，当点都在一条直线上的时候，重叠在一起，看不出来
                # 当点不在一条直线上的时候，
                srtInd = self.X[:, 0].argsort(0).T.tolist()[0]  # 排序，返回索引值
                xSort = self.X[srtInd][:, 0]
                left.plot(xSort, self.preY[srtInd], 'r')  # 绘制拟合直线

                ax1 = fig.add_subplot(122)
                # 权重与迭代次数关系
                ax1.set_xlabel('迭代次数')
                ax1.set_ylabel('权重')
                ax1.set_title('坐标下降:逐项更新参数')
                ax1.plot(self.RCIters)
                # 残差与迭代次数关系
                ax2 = ax1.twinx()
                ax2.set_ylabel('残差')
                ax2.set_xlabel('迭代次数')
                ax2.plot(self.LossList, 'k--')
                # 保存图片
                # plt.savefig('./Data/Gif/' + str(self.iters).zfill(3) + '.jpg')
                plt.show()
                plt.pause(0.001)
                plt.ioff()  # 关闭交互模式
        run()
    # 岭回归(L2正则化)+标准方程
    def ridgeRegress(self,model,lamb=0.02,scope=30):#标准方程求解
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        if lamb==0:
            print('lamb取值错误，lamb非零！')
            return
        def fun(la):
            xTIx = self.X.T * self.X+la*np.diag([1.0]*self.N)
            self.RC = xTIx.I * self.X.T * self.Y
            self.preY = self.X * self.RC
            self.RCList.append(sum(self.RC.tolist(),[]))
            self.LossList.append(costs(self.RC)[0,0])
        def run():
            for i in range(scope):
                lam = np.exp(i-10)
                self.lamList.append(np.log(lam))
                fun(lam)
        if model:
            run()
        else:
            fun(lamb)
    #lasso回归(L1正则化)：坐标下降+次梯度、梯度下降+次梯度
    def lassoRegress(self,model,lamb=0.02,scope=30):
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        def fun(la):
            RC = np.zeros((self.X.shape[1],1))
            RCtmp = np.zeros((self.X.shape[1],1))
            minLoss = np.inf  # 代价
            iters = 0
            while True:
                self.preY = self.X * RC
                oldLoss = costs(RC)[0, 0]
                for j in range(self.N):#坐标下降+次梯度
                    RCcut,Xcut = copy.deepcopy(RC),copy.deepcopy(self.X)#复制，防止改变原对象
                    RCcut = np.delete(RCcut,j,axis=0)
                    Xcut = np.delete(Xcut, j, axis=1)
                    # 0 = self.X[:,j].T*(Xcut*RCcut-self.Y)+wj*np.power(self.X[:,j],2).sum()+self.M*lamb/self.N*Vj
                    rj = self.X[:,j].T*(Xcut*RCcut-self.Y)
                    zj = np.power(self.X[:,j],2).sum()
                    # wj = 0 if RC[j,:]==0 else -(rj+self.M*la/self.N*np.sign(RC[j]))/zj
                    wj = 0 if np.abs(rj)<=self.M*la/self.N else-(rj+self.M*lamb/self.N*np.sign(RC[j]))/zj

                #都计算完后在更新
                #     RCtmp[j,:] = wj
                # RC = RCtmp

                    #计算完一个就更新一个
                    RC[j,:] = wj
                #计算更新后代价
                newLoss = costs(RC)[0,0]
                #***************控制迭代次数+更新*****************
                if newLoss <= minLoss:
                    minLoss = newLoss
                    self.RC = RC#更新回归系数
                    self.RCIters.append(sum(self.RC.tolist(),[]))#记录每次迭代后的回归系数
                    self.Loss.append(newLoss)#记录每次迭代后的代价
                    self.preY = self.X * self.RC  # 计算当前回归系数下的预测值
                    print('迭代次数：', iters, '代价：', newLoss)
                else:#若大于，则说明跳过了局部最优解，退出
                    break
                if np.abs((newLoss-oldLoss))<1e-6:#迭代停止
                    break
                iters +=1
                # self.draw(True)
            self.RC = RC  # 回归系数
            self.RCList.append(sum(self.RC.tolist(), []))
            self.preY = self.X * self.RC  # 计算当前回归系数下的预测值
            self.LossList.append(costs(self.RC)[0, 0])
        def run():
            for i in range(scope):
                lam = np.exp(i - 10)
                self.lamList.append(np.log(lam))
                fun(lam)
        if model:
            run()
        else:
            fun(lamb)
    # 弹性网络回归:L1正则化+L2正则化：坐标下降+次梯度
    def elasticNetRegress(self,model,lamb=0.02,beta=0.5,scope=30):#lasso回归：beta=0，岭回归：beta=1
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        def fun(la):
            RC = np.zeros((self.X.shape[1],1))
            RCtmp = np.zeros((self.X.shape[1],1))
            minLoss = np.inf  # 代价
            iters = 0
            while True:
                self.preY = self.X * RC
                oldLoss = costs(RC)[0, 0]
                for j in range(self.N):
                    RCcut,Xcut = copy.deepcopy(RC),copy.deepcopy(self.X)#复制，防止改变原对象
                    RCcut = np.delete(RCcut,j,axis=0)
                    Xcut = np.delete(Xcut, j, axis=1)
                    # 0 = self.X[:,j].T*(Xcut*RCcut-self.Y)+wj*(np.power(self.X[:,j],2).sum()+la*beta/self.N)+la*(1-beta)/self.N *Vj
                    rj = self.X[:,j].T*(Xcut*RCcut-self.Y)
                    zj = np.power(self.X[:,j],2).sum()+la*beta/self.N
                    wj = 0 if np.abs(rj)<=la*(1-beta)/self.N else-(rj+la*(1-beta)/self.N *np.sign(RC[j]))/zj
                #都计算完后在更新
                #     RCtmp[j,:] = wj
                # RC = RCtmp
                    #计算完一个就更新一个
                    RC[j,:] = wj
                #计算更新后代价
                newLoss = costs(RC)[0,0]
                #***************控制迭代次数+更新*****************
                if newLoss <= minLoss:
                    minLoss = newLoss
                    self.RC = RC#更新回归系数
                    self.RCIters.append(sum(self.RC.tolist(),[]))#记录每次迭代后的回归系数
                    self.Loss.append(newLoss)#记录每次迭代后的代价
                    self.preY = self.X * self.RC  # 计算当前回归系数下的预测值
                    print('迭代次数：', iters, '代价：', newLoss)
                else:#若大于，则说明跳过了局部最优解，退出
                    break
                if np.abs((newLoss-oldLoss))<1e-6:#迭代停止
                    break
                iters +=1
                # self.draw(True)
            self.RC = RC  # 回归系数
            self.RCList.append(sum(self.RC.tolist(), []))
            self.preY = self.X * self.RC  # 计算当前回归系数下的预测值
            self.LossList.append(costs(self.RC)[0, 0])
            return minLoss, self.RC
        def run():
            minLoss = np.inf
            for i in range(scope):
                lam = np.exp(i - 10)
                self.lamList.append(np.log(lam))
                fun(lam)
        if model:
            run()
        else:
            fun(lamb)
    # 前向逐步回归:向前选择
    def forwardRegress(self,maxIters=1000,step=0.1):
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        def fun():
            RC = np.mat(np.zeros((self.N, 1)))  #
            minLoss = np.inf  # 代价
            RCtmp = np.mat(np.zeros((self.N, 1)))
            iters = 0
            while True:
                oldLoss = costs(RC)[0,0]
                for j in range(self.N):
                    for sign in [-1,1]:#第j个增加或减少，观察前后平方误差变化，若平方误差减小，则更新第j个特征回归系数
                        RCtest = copy.deepcopy(RCtmp)
                        RCtest[j] += step*sign
                        testLoss = costs(RCtest)
                        if testLoss<=minLoss:
                            minLoss = testLoss
                            RC = RCtest
                    RCtmp = RC
                self.RCIters.append(sum(RC.tolist(), []))
                self.RC = RC
                self.preY = self.X * RC  # 计算当前回归系数下的预测值
                newLoss = costs(RC)[0, 0]
                self.LossList.append(np.linalg.norm(newLoss))  # 记录每次迭代后的代价
                if np.abs(newLoss - oldLoss) < 1e-3:#变化不大则退出
                    print(self.RC)
                    break
                self.iters += 1
                print('迭代次数：',self.iters,'平方误差：',costs(self.RC)[0,0])
                # self.draw(True)
                fig = plt.figure(1)
                # 最大化显示
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()

                plt.ion()  # 动态显示
                plt.clf()  # 清屏

                left = fig.add_subplot(121)
                left.set_xlabel('X')
                left.set_ylabel('Y')
                left.set_title('回归')
                left.scatter(self.X[:, 0].A, self.Y.A, s=1)  # 绘制散点
                # 当点不在一条直线上的时候，
                srtInd = self.X[:, 0].argsort(0).T.tolist()[0]  # 排序，返回索引值
                xSort = self.X[srtInd][:, 0]
                left.plot(xSort, self.preY[srtInd], 'r')  # 绘制拟合直线

                ax1 = fig.add_subplot(122)
                # 权重与迭代次数关系
                ax1.set_xlabel('迭代次数')
                ax1.set_ylabel('权重')
                ax1.set_title('前向回归:逐项更新参数，步长:{}'.format(step))
                ax1.plot(self.RCIters)
                # 残差与迭代次数关系
                ax2 = ax1.twinx()
                ax2.set_ylabel('残差')
                ax2.set_xlabel('迭代次数')
                ax2.plot(self.LossList, 'k--')
                # 保存图片
                # plt.savefig('./Data/Gif/' + str(self.iters).zfill(3) + '.jpg')
                plt.show()
                plt.pause(0.001)
                plt.ioff()  # 关闭交互模式
        fun()
    # 后退逐步回归：
    def backwardRegress(self):
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        def fun():
            RCtmp = np.mat(np.ones((self.N, 1)))
            minLoss = np.inf  # 代价
            iters = 0
            while True:
                oldLoss = costs(RCtmp)
                for j in range(self.N):
                    loss = costs(RCtmp)
                    RCtest = copy.deepcopy(RCtmp)
                    RCtest[j,:] = RCtest[j,:]*0
                    testLoss = costs(RCtest)
                    if testLoss< loss:
                        loss = testLoss
                        RCtmp = RCtest
                RC = RCtmp
                self.RCIters.append(sum(RC.tolist(), []))
                self.RC = RC
                self.preY = self.X * RC  # 计算当前回归系数下的预测值
                newLoss = costs(RC)[0, 0]
                if np.abs(newLoss - oldLoss) < 1e-6:  # 变化不大则退出
                    break
                iters += 1
                print('迭代次数：',iters,'代价：',newLoss[0,0])
                # self.draw(True)
        fun()
    # 逐步回归
    def stepwiseRegress(self,step=0.001):
        def costs(rc):  # 代价函数：平方误差
            return (self.Y - self.X * rc).T * (self.Y - self.X * rc)
        def fun():
            RC = np.mat(np.zeros((self.N, 1)))  #
            minLoss = np.inf  # 代价
            RCtmp = np.mat(np.zeros((self.N, 1)))
            while True:
                oldLoss = costs(RC)[0,0]
                for j in range(self.N):
                    #向前选择+向后剔除
                    for sign in [-1000,-100,-10,-1,0,1,10,100,1000]:#第j个特征增加、减少或者删除，观察前后平方误差变化，若平方误差减小，则更新第j个特征回归系数
                        FRCtest = copy.deepcopy(RC)
                        BRCtest = copy.deepcopy(RC)
                        if sign !=0:#向前选择(增加或减少)
                            FRCtest[j] += step*sign
                            testLoss = costs(FRCtest)
                        else:#向后剔除(归零)
                            BRCtest[j] = 0
                            testLoss = costs(BRCtest)
                        if testLoss<=minLoss:
                            minLoss = testLoss
                            RCtmp = FRCtest if sign !=0 else BRCtest
                    RC = RCtmp
                self.RC = RC
                self.preY = self.X * RC  # 计算当前回归系数下的预测值
                self.RCIters.append(sum(self.RC.tolist(), []))
                self.LossList.append(costs(self.RC)[0, 0])
                newLoss = costs(RC)[0, 0]
                if np.abs(newLoss - oldLoss) < 1e-6:#变化不大则退出
                    break
                self.iters += 1
                print('迭代次数：',self.iters,'平方误差：',costs(self.RC)[0,0])
                # self.draw(True)
                #动态演示
                fig = plt.figure(1)
                # 最大化显示
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
                plt.ion()  # 动态显示
                plt.clf()  # 清屏

                left = fig.add_subplot(121)
                left.set_xlabel('X')
                left.set_ylabel('Y')
                left.set_title('回归')
                left.scatter(self.X[:, 0].A, self.Y.A, s=1)  # 绘制散点
                srtInd = self.X[:, 0].argsort(0).T.tolist()[0]  # 排序，返回索引值
                xSort = self.X[srtInd][:, 0]
                left.plot(xSort, self.preY[srtInd], 'r')  # 绘制拟合直线

                ax1 = fig.add_subplot(122)
                # 权重与迭代次数关系
                ax1.set_xlabel('迭代次数')
                ax1.set_ylabel('权重')
                ax1.set_title('逐步回归:逐项更新参数,步长：{}'.format(step))
                ax1.plot(self.RCIters)
                # 残差与迭代次数关系
                ax2 = ax1.twinx()
                ax2.set_ylabel('残差')
                ax2.set_xlabel('迭代次数')
                ax2.plot(self.LossList, 'k--')
                # 保存图片
                # plt.savefig('./Data/Gif/' + str(self.iters).zfill(3) + '.jpg')
                plt.show()
                plt.pause(0.001)
                plt.ioff()  # 关闭交互模式

        fun()
    #最小角回归
    def LAR(self,step=0.2,maxIter=300):
        #回归系数初始化
        m,n = self.oraX.shape
        self.RC=np.mat([0.0]*n).T#n*1
        while True:
            self.iters+=1
            #计算残差
            loss = self.Y-self.oraX*self.RC
            #计算所有特征向量与残差余弦相似度
            a=self.oraX.T*loss/np.linalg.norm(loss)
            b=np.mat(1/np.linalg.norm(self.oraX,axis=0)).T
            cosec = np.multiply(a,b)
            #寻找最大相关的特征向量
            # id = np.argmax(abs(cosec))#同时考虑正负相关强度
            id = np.argmax(cosec)#只考虑正相关
            corrMax = cosec[id][0,0]
            #更新权重系数
            # buchang = step*corrMax/abs(corrMax)#固定步长绝对值大小
            buchang = step*corrMax#不固定步长绝对值大小
            self.RC[id] += buchang
            self.RCIters.extend(self.RC.T.tolist())
            #停止迭代条件：
            newloss = self.Y-self.oraX*self.RC
            self.LossList.append(np.linalg.norm(newloss))  # 记录每次迭代后的代价
            self.preY = self.oraX * self.RC  # 计算当前回归系数下的预测值
            if np.linalg.norm(newloss-loss)<1e-3 or self.iters>maxIter:
                print(self.RC)
                break
            # 动态显示
            # fig=plt.figure(1)
            # # 最大化显示
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()
            # plt.ion()  # 动态显示
            # plt.clf()  # 清屏
            # left = fig.add_subplot(121)
            # left.set_xlabel('X')
            # left.set_ylabel('Y')
            # left.set_title('回归')
            # left.scatter(self.oraX[:, 0].A, self.Y.A, s=1)  # 绘制散点
            # srtInd = self.oraX[:, 0].argsort(0).T.tolist()[0]  # 排序，返回索引值
            # xSort = self.oraX[srtInd][:, 0]
            # left.plot(xSort, self.preY[srtInd], 'r')  # 绘制拟合直线
            #
            # ax1 = fig.add_subplot(122)
            # #权重与迭代次数关系
            # ax1.set_xlabel('迭代次数')
            # ax1.set_ylabel('权重')
            # ax1.set_title('最小角回归:逐项更新第'+str(id)+'列参数,步长：{:.5f}'.format(buchang))
            # ax1.plot(self.RCIters)
            # #残差与迭代次数关系
            # ax2 = ax1.twinx()
            # ax2.set_ylabel('残差')
            # ax2.set_xlabel('迭代次数')
            # ax2.plot(self.LossList, 'k--')
            # # 保存图片
            # # plt.savefig('./Data/Gif/'+str(self.iters).zfill(3)+'.jpg')
            # plt.show()
            # plt.pause(0.001)
            # plt.ioff()  # 关闭交互模式

    def draw(self,show=False):#适用于二维特征数据集绘制
        if show:
            plt.ion()  # 动态显示
            plt.clf()  # 清屏
        plt.figure(1)
        plt.title('回归')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(self.X[:, 0].A, self.Y.A, s=1)#绘制散点
        #为什么需要排序？因为plot默认是顺序绘制，当点都在一条直线上的时候，重叠在一起，看不出来
        #当点不在一条直线上的时候，
        srtInd = self.X[:, 0].argsort(0).T.tolist()[0]  # 排序，返回索引值
        xSort = self.X[srtInd][:,0]
        plt.plot(xSort, self.preY[srtInd],'r')#绘制拟合直线
        # plt.plot(self.X[:, 0], self.preY, 'r')  # 绘制拟合直线

        #保存图片
        # plt.savefig('./Data/Gif/'+str(self.iters).zfill(3)+'.jpg')
        plt.show()

        # plt.figure(2)
        # plt.subplot(311)
        # plt.plot(range(len(self.Loss)), self.Loss)  # 绘制代价迭代值
        # # 绘制回归系数每个元素的代价迭代值
        # self.RCList = np.array(self.RCList)
        # plt.subplot(312)
        # plt.scatter(range(len(self.RCList)),self.RCList[:,0],s=0.25)
        # plt.subplot(313)
        # plt.scatter(range(len(self.RCList)),self.RCList[:,1],s=0.25)
        # plt.show()
        # self.RCList = self.RCList.tolist()
        if show:
            plt.pause(0.001)
            plt.ioff()  # 关闭交互模式

def loadData(fileName,delim='\t'):
    with open(fileName, encoding='utf-8') as file:
        baseData = file.readlines()  # 包含换行符、制表符，每个元素为字符串
    # 数据清洗：去换行符、通过制表符分割为列表、列表中每一个字符串元素转换为浮点数
    baseData = [list(map(float, da.strip().split(delim))) for da in baseData]
    dataSet = np.mat(baseData)[:, :-1]
    # 数据清洗：列表中每一个浮点数元素转换为整数：标签
    outValue = np.mat(baseData)[:, -1]
    return dataSet, outValue


if __name__ == '__main__':
    # ******加载数据集**********
    # trainData = 'ex0.txt'  # ex0/horseColicTraining2/abalone/ex00
    # RawDataSet, OutValue = loadData(r'./Data/' + trainData)  # m*n,m*1
    # tmpData = np.insert(RawDataSet,1,values=OutValue.T,axis=1)

    # from sklearn import datasets
    # dataset = datasets.load_boston()
    # RawDataSet = np.mat(dataset.data)
    # OutValue = np.mat(dataset.target).T

    import pandas as pd
    # 读取数据
    target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    # 其中header和prefix的作用为在没有列标题时指定列名
    df = pd.read_csv(target_url, header=None, prefix='V')
    df['V60'] = df.iloc[:, -1].apply(lambda v: 1.0 if v == 'M' else 0.0)
    # 数据标准化  并拆分成属性x和标签y
    norm_df = (df - df.mean()) / df.std()
    RawDataSet = np.mat(df.values[:, :-1])
    OutValue = np.mat(df.values[:, -1]).T


    # from sklearn import datasets
    # centers = [[-2, -8], [-1, 7], [6, 5], [9, 8], [0, 2.3]]
    # RawDataSet, OutValue = datasets.make_blobs(n_samples=300, n_features=2,centers=centers,cluster_std=[0.2, 0.4, 0.6, 0.3, 0.3])
    # tmpData = np.mat(RawDataSet)
    #
    # qq = Edge.Edge(tmpData)
    # qq.margin()#凸边
    # qq.concave()# 凸边凹化
    # qq.display()

    # ******数据预处理*********
    data = clean.CleanData(RawDataSet, OutValue)  # 实例化
    # data.ascend(3)  # 数据集升维处理:普通高阶/组合高阶,用于多项式回归
    # data.zero_centered('all')# 中心化
    data.z_score('all')#标准正态分布
    # data.normData('all')  # 归一化
    RawDataSet, OutValue= data.X, data.Y
    #********回归*************
    xx = Regress(RawDataSet, OutValue)#实例化
    # 标准方程
    # xx.standRegress()
    # 多项式线性回归->特征升维+标准方程
    # xx.polynomialRegress(6)
    # 局部加权线性回归+标准方程
    # xx.LWLRegress()
    # ['梯度下降','批量梯度下降'/'随机梯度下降/小批量梯度下降','线性回归'/'逻辑回归']/['坐标下降''线性回归'/'逻辑回归']+迭代
    # xx.GD_CD(model=['坐标下降'])
    # 前向逐步回归+迭代
    # xx.forwardRegress()
    # 后退逐步回归+迭代
    # xx.backwardRegress()
    # 逐步回归+迭代
    # xx.stepwiseRegress()
    # lasso回归:坐标轴下降+次梯度+迭代,model:是否遍历多个λ来观察回归系数与log(λ)关系
    # xx.lassoRegress(model=True)
    # 岭回归+标准方程，model:是否遍历多个λ来观察回归系数与log(λ)关系
    # xx.ridgeRegress(model=True)
    # 弹性网络回归+迭代
    # xx.elasticNetRegress(model=True)
    # xx.draw()
    #最小角回归-非常适用于样本数<特征数
    xx.LAR()

    # xx.evaluation('回归系数、平方误差与log(λ)关系')#相关性、,平方误差、回归系数与log(λ)关系、回归系数与迭代次数关系、平方误差与log(λ)关系
    # xx.evaluation('平方误差、回归系数与迭代次数关系')
