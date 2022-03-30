# --*-- coding:utf-8 --*--
import numpy as np


class Node:  # 结点
    def __init__(self, data, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild


class KdTree:  # kd树
    def __init__(self):
        self.kdTree = None

    def create(self, dataSet, depth):  # 创建kd树，返回根结点  参数1：数据集   参数2：树的深度（同时决定：排序依据的维度）
        if (len(dataSet) > 0):
            m, n = np.shape(dataSet)  # 求出样本行，列
            midIndex = int(m / 2)  # 中间数的索引位置
            axis = depth%2  # 判断以哪个轴划分数据
            sortedDataSet = self.sort(dataSet, axis)  # 进行排序
            node = Node(sortedDataSet[midIndex])  # 将节点数据域设置为中位数，具体参考下书本
            # 将两边的数切割成两个集合，然后递归调用create方法
            leftDataSet = sortedDataSet[: midIndex]  # 将中位数的左边创建2改副本
            rightDataSet = sortedDataSet[midIndex+1:]
            print(leftDataSet)
            print(rightDataSet)
            node.lchild = self.create(leftDataSet, depth + 1)  # 将中位数左边样本传入来递归创建树
            node.rchild = self.create(rightDataSet, depth + 1)
            return node
        else:
            return None

    def sort(self, dataSet, axis):  # 采用冒泡排序，利用aixs作为轴进行划分
        sortDataSet = dataSet[:]  # 由于不能破坏原样本，此处建立一个副本
        m, n = np.shape(sortDataSet)
        for i in range(m):
            for j in range(0, m - i - 1):
                if (sortDataSet[j][axis] > sortDataSet[j + 1][axis]):  # 把数值大的一个一个向下沉
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j + 1]
                    sortDataSet[j + 1] = temp
        print(sortDataSet)
        return sortDataSet

    def preOrder(self, node):  # 前序遍历
        if node != None:
            print("tttt->%s" % node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    def search(self, tree, x):  # 搜索
        self.nearestPoint = None  # 保存最近的点
        self.nearestValue = 0  # 保存最近的值

        def travel(node, depth=0):  # 递归搜索
            if node != None:  # 递归终止条件
                n = len(x)  # 特征数
                axis = depth % n  # 计算轴
                if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
                    travel(node.lchild, depth + 1)
                else:
                    travel(node.rchild, depth + 1)

                # 以下是递归完毕后，往父结点方向回朔，对应算法3.3(3)
                distNodeAndX = self.dist(x, node.data)  # 目标和节点的距离判断
                if (self.nearestPoint == None):  # 确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX

                print(node.data, '\t', depth, '\t', self.nearestValue, distNodeAndX, '\t', node.data[axis], x[axis])
                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                    if x[axis] < node.data[axis]:
                        travel(node.rchild, depth + 1)
                    else:
                        travel(node.lchild, depth + 1)

        print('位置\t树深度\t当前最近距离\t当前点与目标点距离\t')
        travel(tree)  # tree的属性与根结点是一样的，所以可以带入定义方法参数时，里面的node属性
        return self.nearestPoint

    def dist(self, x1, x2):  # 欧式距离的计算
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5

def dataNorm(dataSet):
    minValues = dataSet.min(0) #获取每列的最小值
    maxValues = dataSet.max(0)  # 获取每列的最大值
    normDataSet = np.zeros(dataSet.shape)
    h = dataSet.shape[0]
    scope = maxValues - minValues
    normDataSet = dataSet - np.tile(minValues,(h, 1))
    normDataSet = normDataSet/ scope
    return normDataSet, scope, minValues

if __name__ == '__main__':
    nodeLocations = np.load('WSN.liveNodes3D.npy')
    # normDataSet, scope, minValues = dataNorm(nodeLocations)
    dataSet = [[2, 3,1],
               [5, 4,2],
               [9, 6,3]]
    x = [20, 6,2]
    kdtree = KdTree()
    tree = kdtree.create(dataSet, 0)  # 创建KD树
    print('打印先根遍历的输出顺序')
    kdtree.preOrder(tree)  # 先序遍历测试
    print("给定的数据的最近邻", kdtree.search(tree, x))  # 在树中搜索给定的 X的最近邻