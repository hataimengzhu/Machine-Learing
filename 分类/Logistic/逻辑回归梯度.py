import numpy as np
import matplotlib.pyplot as plt
import operator
import random
import datetime as dtime
import math
def getDataSet():
    with open('testSet.txt', encoding='utf-8') as file:
        baseData = file.readlines()
    dataSet = []
    label = []
    for da in baseData:
        da = da.strip().split()
        dataSet.append([1.0,float(da[0]),float(da[1])])
        label.append(int(da[2]))

    dataSet = np.mat(dataSet)
    # dataSet = np.insert(dataSet, 0, values=np.ones(len(dataSet)), axis=1)
    label = np.mat(label).T

    return dataSet, label
def normData(shuju):
    columnMin = [min(shuju[:,col]).getA()[0][0] for col in range(shuju.shape[1])]
    columnMax = [max(shuju[:,col]).getA()[0][0] for col in range(shuju.shape[1])]
    miaRange = np.abs(list(map(operator.add,columnMax,[-x for x in columnMin])))
    for i in np.where(miaRange==0.0):
        for j in i:
            miaRange[j] = columnMax[j]
            columnMin[j] = 0.0
    normShuju = (shuju-columnMin)
    for i in range(len(miaRange)):
        normShuju[:,i] = normShuju[:,i]/miaRange[i]
    return normShuju

def drawBaseDataSet(shuju,biaoqian,theta,thetaList,tigerList):
    shuju = shuju.getA()
    biaoqian = biaoqian.getA()

    dataDic = {}
    colors = ['b', 'r', 'g', 'c', 'y', 'm']
    for ta in biaoqian:
        dataDic[ta[0]] = []
    for i, [value] in enumerate(biaoqian):
        dataDic[value].append(shuju[i,1:])

    plt.figure(1)
    plt.subplot(121)
    for key,value in dataDic.items():
        value = np.array(value)
        plt.scatter(value[:,0], value[:,1], c=colors[list(dataDic.keys()).index(key)])

    plt.subplot(121)
    x = np.arange(-3.0, 3.0, 0.1)
    y=-(theta[0][0]+theta[1][0]*x)/theta[2][0]
    plt.plot(x,y)

    plt.subplot(122)
    plt.scatter(range(len(tigerList)), tigerList,s=0.25)

    plt.figure(2)
    plt.subplot(311)
    plt.scatter(range(len(thetaList)), thetaList[:,0],s=0.25)
    plt.subplot(312)
    plt.scatter(range(len(thetaList)), thetaList[:,1],s=0.25)
    plt.subplot(313)
    plt.scatter(range(len(thetaList)), thetaList[:,2],s=0.25)
    plt.show()

def sigmoid(wx,model):
    if model>0:
        return 1.0 / (1 + np.exp(-wx))
    else:
        return 1/(1+np.exp(wx))
def tigerFun(shuju,tag,theta):
    temp1 = np.log(sigmoid(shuju * theta,1))
    temp2 = -tag.T*temp1

    temp3 = np.log(sigmoid(temp1,-1))
    temp4 = -(1-tag).T*temp3

    return (temp2.getA()+temp4.getA())[0][0]
def BGD(shuju,tag):
    inters = 500
    alpha_int = 0.01

    theta= np.mat(np.ones(shuju[0].shape[1])).T
    thetaList = [np.mat(np.ones(shuju[0].shape[1])).T]
    tigerList = []

    for i in range(inters):
        alpha = alpha_int+len(shuju)/(1.0+i)
        temp1 = sigmoid(shuju * theta, 1)
        temp2 = shuju.T * (temp1 - tag)
        temp = theta - alpha * temp2
        theta = temp
        # 更新
        thetaList.append(temp)
        tigerList.append(tigerFun(shuju,tag,theta))

    thetaList = np.array(thetaList)
    return theta.getA(), thetaList, tigerList
def SGD(shuju,tag):
    inters = 20
    alpha_int = 0.01
    theta= np.mat(np.ones(shuju[0].shape[1])).T
    thetaList = [np.mat(np.ones(shuju[0].shape[1])).T]
    tigerList = []
    for j in range(inters):
        for i,da in enumerate(shuju):
            alpha = alpha_int + len(shuju) / (1.0 +i+j)
            h = sigmoid(da * theta, 1)
            error = (h-tag[i]).getA()[0][0]
            temp = theta-alpha*da.T*error
            theta = temp
            # 更新
            thetaList.append(temp)
            tigerList.append(tigerFun(da,tag[i],theta))


    thetaList = np.array(thetaList)
    return theta.getA(), thetaList, tigerList
def SSGD(shuju,tag):
    inters = 20
    alpha_int = 0.01
    theta= np.mat(np.ones(shuju[0].shape[1])).T
    thetaList = [np.mat(np.ones(shuju[0].shape[1])).T]
    tigerList = []
    for j in range(inters):
        i = 0
        shujuTmp = shuju
        tagTmp = tag
        while shujuTmp.shape[0]:
            ii = random.randrange(0, shujuTmp.shape[0])
            alpha = alpha_int + shujuTmp.shape[0] / (1.0 +i+j)

            h = sigmoid(shujuTmp[ii] * theta, 1)
            error = (h-tagTmp[ii]).getA()[0][0]
            temp = theta-alpha*shujuTmp[ii].T*error
            theta = temp
            # 更新
            thetaList.append(temp)
            tigerList.append(tigerFun(shujuTmp[ii],tagTmp[ii],theta))

            shujuTmp = np.delete(shujuTmp, ii, axis=0)
            tagTmp = np.delete(tagTmp, ii, axis=0)

            i += 1

    thetaList = np.array(thetaList)
    return theta.getA(), thetaList, tigerList

def test(shuju,tag, theta):
    LineInput= sigmoid(shuju*theta,1)>0.5
    tag = tag>0
    accuracy = LineInput ==tag
    print('精确率：%.2f%%' % float(sum(accuracy.tolist(), []).count(True)/accuracy.shape[0]))

if __name__=='__main__':
    dataSet, label = getDataSet()
    # normDataSet = normData(dataSet)
    #批量梯度下降-BatchGD ：迭代：(依次遍历完全部样本后更新回归系数theta)  精确率：0.96
    # theta, thetaList, tigerList = BGD(dataSet, label)
    #随机梯度下降-SGD: 迭代：(依次遍历一个样本后更新一次回归系数theta) 精确率：0.93
    theta, thetaList, tigerList = SGD(dataSet, label)
    #改进随机梯度下降-SSGD 迭代：(随机遍历一个样本后更新一次回归系数theta) 。精确率：0.96
    # theta, thetaList, tigerList = SSGD(dataSet, label)

    drawBaseDataSet(dataSet, label,theta, thetaList, tigerList)
    # test(dataSet, label, theta)

