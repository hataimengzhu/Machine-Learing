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
def drawBaseDataSet(shuju,biaoqian,theta):
    shuju = shuju.getA()
    biaoqian = biaoqian.getA()
    dataDic = {}
    colors = ['r', 'b', 'g', 'c', 'y', 'm']
    for ta in biaoqian:
        dataDic[ta[0]] = []
    for i, [value] in enumerate(biaoqian):
        dataDic[value].append(shuju[i,1:])
    plt.figure(1)
    plt.subplot(111)
    for key,value in dataDic.items():
        value = np.array(value)
        plt.scatter(value[:,0], value[:,1], c=colors[list(dataDic.keys()).index(key)])

    x = np.arange(-3.0, 3.0, 0.1)
    y=-(theta[0][0]+theta[1][0]*x)/theta[2][0]

    plt.subplot(111)
    plt.plot(x,y)
    plt.show()

class Team(object):
    dataSet, tag = getDataSet()
    inters = 500
    sumLearns = 10# 学生数量
    students = None
    def rank():
        gradeDic = {}
        for stu in Team.students: # 排名
            gradeDic[stu.id] = min(stu.tiger)
            # print(stu.id,min(stu.tiger),"************")
        gradeDic = sorted(gradeDic.items(), key=operator.itemgetter(1), reverse=False)
        #分割样本
        studentsTmp = []
        for i in range(len(gradeDic)):
            if i <Team.sumLearns: # 当前学生数量
                for stu in Team.students:
                    if stu.id == gradeDic[i][0]:
                        studentsTmp.append(stu)
        dataSetList = []
        for stu in studentsTmp:
            stu.example = np.insert(stu.example, stu.example[0].shape[1], values=stu.tag.T, axis=1)
            dataSetList.extend(stu.example.getA().tolist())
        #样本按行随机排列
        newDataSetList = []
        while dataSetList:
            ii = random.randrange(0,len(dataSetList))
            newDataSetList.append(dataSetList[ii])
            dataSetList.pop(ii)
        newDataSetList = np.mat(newDataSetList)
        #更新
        Team.dataSet = newDataSetList[:,:3]
        Team.tag = newDataSetList[:, -1]
        Student.intLearner()
class Student(object):
    alpha_init = 0.1
    def __init__(self,shuju,tag):
        self.id = None # 学号
        self.example = shuju
        self.tag = tag
        self.alpha = Student.alpha_init
        self.theta = np.mat(np.ones(shuju[0].shape[1])).T
        self.tiger = []
        self.scort = None # 成绩
        self.type = 'N'       # 'N':普通学员,'G':组长
        self.head_id = None
        self.leave = False
    def intLearner():
        students = []
        j = 0
        step = round(len(Team.dataSet)/Team.sumLearns)
        for i in range(0,len(Team.dataSet),step):  #重新设计，均分样本
            temp = Student(Team.dataSet[i:i+step,:],Team.tag[i:i+step])
            temp.id = j
            temp.scort = 0
            students.append(temp)
            j += 1
        Team.students = students
class Learner(object):
    heads = None # 组长
    memebrs =None
    cluster = None
    learnScores = np.array(np.zeros((Team.inters,2)))

    def sigmoid(wx,model):
        if model>0:
            return 1.0 / (1 + np.exp(-wx))
        else:
            return 1/(1+np.exp(wx))

    def tigerFun(stu):
        temp1 = stu.example * stu.theta
        temp2 = Learner.sigmoid(temp1, 1)
        temp3 = np.log(temp2)
        temp4 = -stu.tag.T*temp3

        temp5 = np.log(Learner.sigmoid(temp1,-1))
        temp6 = -(1-stu.tag).T*temp5

        return (temp4.getA()+temp6.getA())[0][0]
    def study():
        students = Team.students
        for stu in students:
            temp1 = stu.example*stu.theta
            temp2 = Learner.sigmoid(temp1,1)
            temp3 = stu.example.T*(temp2-stu.tag)
            stu.theta -= stu.alpha*temp3
            #更新Team
            Team.students[students.index(stu)].theta = stu.theta
            Team.students[students.index(stu)].tiger.append(Learner.tigerFun(stu))
            # print(Learner.tigerFun(stu))
    def findPole():
        students = Team.students
        for r in range(Team.inters):
            # print('***第'+str(r+1)+'周考试***')
            for stu in students:
                stu.type = 'N'
                stu.alpha = stu.alpha + len(stu.example) / (1.0 + r)
            Learner.study()
            if (r+1)%50==0 and Team.sumLearns>1:
                print('rank')
                Team.sumLearns = math.ceil(Team.sumLearns * 0.5)  # 更新学生数量
                print('Team.sumLearns:', Team.sumLearns)
                Team.rank()
        return Team.students[0].theta


if __name__=='__main__':
    start = dtime.datetime.now()

    Team()
    Student.intLearner()
    theta =Learner.findPole()
    theta = theta.getA()

    end = dtime.datetime.now()
    print("耗时：{}".format(end - start))

    dataSet, label = getDataSet()
    drawBaseDataSet(dataSet, label,theta)

