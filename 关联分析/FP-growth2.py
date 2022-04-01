#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import numpy as np
import copy as copy
import time

class TreeNode:#树节点
    def __init__(self,name,count,parent):
        self.Name = name
        self.Count = count
        self.NodeLink = None#用于链接相似的元素项
        self.Parent = parent
        self.Children = {}
    def disp(self, ind=1):
        print(' |' * ind, self.Name, self.Count)
        for child in self.Children.values():
            child.disp(ind+1)
class Tree:
    def __init__(self,lists,minSup,minConf):
        self.lists = lists#事务列表
        self.l = len(self.lists)#原始事务条数
        self.workDic = {}#过滤+降序后的事务
        self.freDic = {}#频繁集
        self.supSet = {}  # 支持度集合
        for li in self.lists:
            self.workDic[tuple(set(li))]=self.workDic.get(tuple(set(li)),0)+1
        self.MS = minSup#最小支持度/先验概率
        self.MC = minConf#最小置信度
        self.minSup = minSup#最小支持度
        self.hand={}#元素频数指针
        self.tree = None#初始化根节点(Name,Count,Parent)
        self.rules = None#关联规则
    # **********构建FP树+元素指针表(包含相同元素链表的起始指针)***************
    # 构建FP树
    def plantTree(self):
        #********初始化*******
        self.hand={}#元素频数指针
        self.tree = TreeNode('root', 1, None)#根节点(Name,Count,Parent)
        # ******过滤、排序原始事务***********
        #1、生成头指针列表
        for work in self.workDic:
            for item in work:
                self.hand[item] = self.hand.get(item,0)+self.workDic[work]
        #2、过滤低于置信度的元素
        [self.hand.pop(key) for key in list(self.hand.keys()) if self.hand[key]<self.minSup*self.l]
        if len(self.hand.keys())==0:return None,None#没有满足最低置信度要求的项集
        #3、排序、添加链
        self.hand=dict(sorted(self.hand.items(),key=lambda p:(p[1],p[0]),reverse=True))#降序
        lists = list(self.hand.keys())#降序列表
        sets = set(self.hand.keys())#无序集合
        for key,value in self.hand.items():
            self.hand[key] = [value,None]
        #******构建FP树********************
        for work,count in self.workDic.items():
            #*****过滤+排序事务******
            sortList = copy.deepcopy(lists)#降序列表
            workTmp = set(work)&sets#过滤+排序
            if len(workTmp)>0:#过滤+排序
                tmp = sets-workTmp#差集
                [sortList.remove(t) for t in tmp]#过滤差集:得到降序交集
                #*****生成FP树**********
                self.growth(sortList,self.tree,count)
        return self.tree,self.hand
    #FP树添加节点
    def growth(self, branch, tree, count):  # 树苗(sapling)生长(添加节点-枝条)
        #*********添加节点+更新指针元素链表指向*********
        # 1、判断branch的第一个元素是否已作为子节点(children)
        if branch[0] in tree.Children:  # 若已作为子节点，则该子节点计数加1
            tree.Children[branch[0]].Count += count
        else:  # 若还不是子节点，则创建一个新的树节点并将其作为一个子节点添加到树中
            # 创建新的分支
            tree.Children[branch[0]] = TreeNode(branch[0], count, tree)
            # 更新指针链表指向
            newPosition = tree.Children[branch[0]]
            if self.hand[branch[0]][1] is None:
                self.hand[branch[0]][1] = newPosition
            else:
                self.pointing(self.hand[branch[0]][1], 'down', targetNode=newPosition)
        # 2、添加branch剩余的节点
        if len(branch) > 1:  # 判断branch是否有其他元素
            self.growth(branch[1:], tree.Children[branch[0]], count)
    # 指针指向指定位置
    def pointing(self,node,model,pt=[],targetNode=None):
        if model=='down':#向下
            while node.NodeLink is not None:
                node = node.NodeLink
            node.NodeLink = targetNode
        elif model=='root':#回溯到根节点
            if node.Parent is not None:
                pt.append(node.Name)
                self.pointing(node.Parent,'root',pt)
    #**********从FP树中寻找频繁项集:1、前缀路径，2、条件FP树**************************************
    #查找已给定元素项结尾的所有路径
    def findPath(self,basePat,hand=None):
        if hand is None:
            hand = self.hand
        paths={}#初始化路径集合
        #对同一元素的每条路径方向回溯，直到根节点
        treeNode = hand[basePat][1]#hand:{basePat:[count,nextLink]}
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
    # 构建条件FP树:递归查找频繁项集
    def train(self):
        self.plantTree()#构建FP树
        self.tree.disp()#打印FP树
        # 递归查找频繁项集
        freSets = set()#初始化频繁集
        freSetLists = []
        def fun(preFix,freqItemList,hand=self.hand):
            lists = [ord[0] for ord in sorted(hand.items(), key=lambda p: (p[1][0], p[0]), reverse=False)] # 升序,从下往上寻找元素频繁集
            for li in lists:
                newFreqSet = preFix.copy()
                newFreqSet.add(li)
                freSetLists.append(newFreqSet)

                # self.supSet[tuple(list(newFreqSet))] = hand[li][0]
                self.supSet[frozenset(newFreqSet)] = hand[li][0]
                if len(newFreqSet) in self.freDic.keys():
                    self.freDic[len(newFreqSet)].append(newFreqSet)
                else:
                    self.freDic[len(newFreqSet)]=[]
                    self.freDic[len(newFreqSet)].append(newFreqSet)
                #查找指定元素的前缀路径
                paths = self.findPath(li,hand)
                # print(paths)
                self.workDic = paths#更新当前元素的事务
                conTree,newHand = self.plantTree()#生成新的元素指针列表和条件FP树
                # print(newHand)
                # print(freSetLists)
                if newHand is not None:
                    # conTree.disp()
                    fun(newFreqSet,freqItemList,newHand)
        fun(freSets,freSetLists)
        #挖掘关联规则-置信度、提升度
        self.rules = self.getRules()
        print('关联规则：',self.rules)
    # 挖掘关联规则
    def getRules(self):
        # 初始化关联规则字典-置信度
        rule = {}
        for k, sets in self.freDic.items():
            if k > 1:
                rule[k] = []
                for value in sets:
                    comb = [tuple([i]) for i in value]
                    # if k==2:#A->B,B->A
                    #     rule,_ = self.calConf(value,comb,rule,k)
                    # elif k>2:#合并规则
                    self.merge(value, comb, rule, k)
        return rule
    # 计算置信度、减枝
    def calConf(self, freSet, zuhe, ru, n):  # freSet:频繁项集，comb：后件，ru:规则，n:第n层
        Pab = self.supSet[frozenset(freSet)]  # 联合概率(分子)
        afterList = []
        for zu in zuhe:  # 遍历频繁集每一个元素作为后件
            antecedent = tuple(set(freSet) - set(zu))  # 前件
            afterparts = tuple(set(zu))  # 后件
            conf = Pab / self.supSet[frozenset(antecedent)]  # P(B|A)=P(A,B)/P(A)
            lift = conf / (self.supSet[frozenset(afterparts)] / self.l)  # 提升度
            if conf >= self.MC:  # 减枝，保留满足最小置信度的规则
                ru[n] += [[antecedent, afterparts, np.round(conf, 2), np.round(lift, 2)]]  # [前件，后件，置信度]
                afterList += [afterparts]
        return ru, afterList
    #合并规则
    def merge(self, freSet, zuhe, ru, n):  # zuhe的作用就是告诉每个组合的元素个数
        m = len(zuhe[0])  # 后件个数
        if len(freSet) == m + 1:
            new_ru, _ = self.calConf(freSet, zuhe, ru, n)
        elif len(freSet) > m + 1:  # 判断频繁项集freSet是否大到可以移除大小为m的子集
            # 递归拆分组合频繁集的子集(后件)，每次递归：每个组合的元素个数为m+1
            subSetIndex = np.nonzero([set(i) < set(freSet) for i in self.freDic[m + 1]])[0]  # 频繁项集的子集的索引
            # 从频繁集字典里选freSet的子集(每个子集m+1各元素)，无需重新计算满足置信度要求的组合
            subFreSets = np.array(self.freDic[m + 1])[subSetIndex].tolist()
            # 筛选满足置信度要求的后件
            new_ru, subFreSets = self.calConf(freSet, subFreSets, ru, n)
            if len(subFreSets[0]) <=len(freSet)-1:  # 判断能否继续拆分组合，后件每个组合元素个数<=len(freSet)-1次才能组合
                self.merge(freSet, subFreSets, new_ru, n)
if __name__ == '__main__':
    #加载训练集
    # with open('./Data/kosarak.dat') as file:
    #     baseData = [list(map(int,line.split()))for line in file.readlines()]
    # dataSet1 = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5], [1, 2]]
    dataSet2 = [['e', 'c', 'b'], ['d', 'c', 'a'], ['e', 'd', 'c', 'a'], ['d', 'a'], ['e', 'd']]
    ds = [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]
    simDat = [['r', 'z', 'h', 'j', 'p'],
              ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
              ['z'],
              ['r', 'x', 'n', 'o', 's'],
              ['y', 'r', 'x', 'z', 'q', 't', 'p'],
              ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    #训练
    start = time.time()  # 计时开始
    xx=Tree(dataSet2,0.1,0.2)
    xx.train()
    end = time.time()  # 计时结束
    print("耗时：{}".format(end - start))












