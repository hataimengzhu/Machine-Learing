#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import copy as copy

class TreeNode:#树节点
    def __init__(self,name,count,parent):
        self.Name = name
        self.Count = count
        self.NodeLink = None#用于链接相似的元素项
        self.Parent = parent
        self.Children = {}
    def disp(self, ind=1):
        # print(' |'*ind, self.Name, ' ', self.Count)
        print(' |' * ind, self.Name, self.Count)
        for child in self.Children.values():
            child.disp(ind+1)
class Tree:
    def __init__(self,lists,minSup):
        self.lists = lists#事务列表
        self.workDic = {}#过滤+降序后的事务
        self.minSup = minSup#最小支持度
        self.hand={}#元素频数指针
        self.tree = TreeNode('root', 1, None)#初始化根节点
    #**********构建FP树+元素指针表(包含相同元素链表的起始指针)***************
    def pointing(self,node,model,pt=[],targetNode=None):#指针指向指定位置
        if model=='down':#向下
            while node.NodeLink is not None:
                node = node.NodeLink
            node.NodeLink = targetNode
        elif model=='root':#回溯到根节点
            if node.Parent is not None:
                pt.append(node.Name)
                self.pointing(node.Parent,'root',pt)
    def growth(self,branch,tree,count):#树苗(sapling)生长(添加节点-枝条)
        #1、判断branch的第一个元素是否已作为子节点(children)
        if branch[0] in tree.Children:#若已作为子节点，则该子节点计数加1
            tree.Children[branch[0]].Count += count
        else:#若还不是子节点，则创建一个新的树节点并将其作为一个子节点添加到树中
            # 创建新的分支
            tree.Children[branch[0]] = TreeNode(branch[0],count,tree)
            #更新指针链表指向
            newPosition = tree.Children[branch[0]]
            if self.hand[branch[0]][1] is None:
                self.hand[branch[0]][1] = newPosition
            else:
                self.pointing(self.hand[branch[0]][1],'down',targetNode=newPosition)
        #2、添加branch剩余的节点
        if len(branch)>1:#判断branch是否有其他元素
            self.growth(branch[1:],tree.Children[branch[0]],count)
    def plantTree(self,l=None):#构建FP树
        if l == None:
            l=len(self.lists)
        #******过滤、排序原始事务***********
        #1、统计各元素频数、过滤
        comb = sum(self.lists,[])
        sets = set(comb)
        counts=np.array([comb.count(i) for i in sets])
        filterIndex = np.nonzero(counts>=self.minSup*l)#满足最小支持度的元素的索引值
        if len(filterIndex[0]) ==0:
            self.hand = None
            return self.hand,None
        arrays,counts = np.array(list(sets))[filterIndex],counts[filterIndex]#保留满足最小支持度的元素
        # filterIndex = np.argsort(-counts)#排序：降序
        # arrays, counts = arrays[filterIndex], counts[filterIndex]#已过滤和排除
        hand = {}
        hand.update(dict(zip(arrays, counts)))#元素字典：元素频数记录
        hand = sorted(hand.items(), key=lambda p: (p[1], p[0]),reverse=True)#降序排序
        arrays = []
        for h in hand:
            arrays+=h[0]
            self.hand[h[0]]=[h[1],None]##self.hand[key]=[元素个数,指向位置]
        arrays = np.array(arrays)#降序
        #2、对每一个事务中的元素进行过滤，注意：set()元素输出顺序不是元素的原始顺序
        filterList = [list((set(li)&set(arrays))) for li in self.lists if len(set(li)&set(arrays))!=0]#保留满足最小支持度的元素的事务
        #2、对每一个事务中的元素进行降序排列
        sortList = []#初始化事务
        lists,sets = arrays.tolist(),set(arrays)#参考列表(已过滤+降序)和集合：lists：过滤、降序后的元素列表，sets：过滤、降序后的元素集合
        for li in filterList:#遍历已过滤的事务列表，进行排序
            copyLists = copy.deepcopy(lists)
            if set(li)==sets:#li事务和元素列表相同(元素相同)
                sortList=copyLists
            elif set(li)<sets:#li事务是元素集合的子集
                if len(li)==1:
                    sortList = li
                else:#保留交集并且降序排列
                    tmp = sets-set(li)#差集
                    for j in tmp:#删除差集
                        copyLists.remove(j)
                    sortList = copyLists
            self.workDic[tuple(sortList)]=self.workDic.get(tuple(sortList),0)+1
        # 3、构建FP树
        for key,value in self.workDic.items():
            self.growth(key,self.tree,value)
        return self.hand,self.tree
    #**********从FP树中寻找频繁项集**************************************
    #查找已给定元素项结尾的所有路径
    def findPath(self,basePat,hand=None):
        if hand is None:
            hand = self.hand
        paths={}#初始化路径集合
        #对同一元素的每条路径方向回溯，直到根节点
        treeNode = hand[basePat][1]
        while treeNode is not None:
            pt = []#单条路径
            #回溯该元素(basePat)到根节点
            self.pointing(treeNode, 'root', pt)
            #更新路径集合
            if len(pt)>1:
                paths[tuple(pt)[1:]] = treeNode.Count#当前节点的元素的频数
            #切换下一条路径方向
            treeNode = treeNode.NodeLink
        return paths
    #构建条件FP树:递归查找频繁项集
    def train(self):
        self.plantTree()#构建FP树
        self.tree.disp()#打印FP树
        #递归查找频繁项集
        freSet = set()#初始化频繁集
        freSetList = []
        l = len(self.lists)
        def fun(preFix,freqItemList,inTree=self.tree,hand=self.hand):
            # lists = list(hand.keys())[::-1]
            lists = [v[0] for v in sorted(hand.items(), key=lambda p:(p[1][0], p[0]))]
            for li in lists:
                newFreqSet = preFix.copy()
                newFreqSet.add(li)
                freqItemList.append(newFreqSet)

                paths = self.findPath(li,hand)
                tmp = []
                #字典转换为列表
                for key,values in paths.items():
                    if values>1:
                        tmp+=[list(key)]*values
                    else:
                        tmp.append(list(key))
                self.lists = tmp
                self.workDic = {}  # 过滤+降序后的事务
                self.hand = {}  # 元素频数指针
                self.tree = TreeNode('root', 1, None)  # 初始化根节点
                newHand,newTree =self.plantTree(l)#self.plantTree的参数为列表
                if newHand is not None:
                    print('conditional tree for: ', newFreqSet)
                    newTree.disp(1)
                    fun(newFreqSet,freqItemList,newTree,newHand)
        fun(freSet,freSetList)
        print(freSetList)


dataSet = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
ds =[[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3]]
simDat = [['r', 'z', 'h', 'j', 'p'],
          ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
          ['z'],
          ['r', 'x', 'n', 'o', 's'],
          ['y', 'r', 'x', 'z', 'q', 't', 'p'],
          ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

if __name__=='__main__':
    with open('./Data/kosarak.dat') as file:
        baseData = [list(map(int,line.split()))for line in file.readlines()]
    dataSet = {}
    for da in baseData:
        dataSet[tuple(set(da))] = dataSet.get(tuple(set(da)),0)+1

    xx=Tree(simDat,0.1)
    xx.train()
