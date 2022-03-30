import numpy as np
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []                                                        #创建数据列表                                                     #创建标签列表
    fr = open('testSet.txt')                                            #打开文件
    for line in fr.readlines():                                            #逐行读取
        lineArr = line.strip().split()                                    #去回车，放入列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])])        #添加数据
    fr.close()
    dataMat = np.array(dataMat)
    plt.scatter(dataMat[:,0], dataMat[:,1],s=2)  # s=0.25比较合适
    plt.show()
    #关闭文件
    return dataMat
def lossFun(xy):
    x_train = xy[:,0]
    y_train = xy[:,1]
    listK = y_train/x_train
    minK = min(listK)
    maxK = max(listK)
    minB = min(y_train)
    maxB = max(y_train)
    dense = 200
    k = np.linspace(minK-1, maxK+1, dense)
    b = np.linspace(minB-1, maxB+1, dense)

    # y = kx+b
    def get_loss_value(k, b):
        return np.square(k * x_train + b - y_train).sum() / len(x_train)/2

    minLoss=99999
    for i in range(dense):
        for j in range(dense):
            loss = get_loss_value(k[i], b[j])
            if minLoss>=loss:
                minLoss = loss
                Jk = k[i]
                Jb = b[j]
    print('loss:',loss,'minLoss:',minLoss)
    plt.scatter(x_train, y_train,s=2)  # s=0.25比较合适
    x= np.linspace(min(x_train)-1, max(x_train)+1, 10)
    y = Jk*x+Jb

    plt.plot(x, y)

    plt.title('Loss Func Contour Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__=='__main__':
    dataMat =loadDataSet()
    lossFun(dataMat)