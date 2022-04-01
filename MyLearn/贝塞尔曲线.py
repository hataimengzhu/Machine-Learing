#!user/bin/python3
# _*_ coding:utf-8 _*_
# author TingXiao-UI
import copy
import numpy as np
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class Bezier:
    def __init__(self,points,baseRatio=1e-5):
        self.curve = []
        self.ratio = baseRatio
        self.Points = points
    def findP(self):  # start,end:矩阵:1*2,1*2。更新p0,p1。ratio:比例：[0,1]
        self.curve.append(self.Points[0])
        self.ratio=self.ratio+1e-2#使raito非固定值,否则递归深度过深，容易栈溢出
        # 寻找pNew
        for ii in range(1, len(self.Points)):
			'''
			*  a(x0,y0)
			.*  b(x,y)
			...
			....
			....*  c(x1,y1)
			(x-x0)/(x1-x0)=0.4-->x=(x1-x0)*0.4+x0
			(y-y0)/(y1-y0)=0.4-->y=(y1-y0)*0.4+y0
			'''
            pNew = [(self.Points[ii][0] - self.Points[ii - 1][0]) * self.ratio + self.Points[ii - 1][0],
                    (self.Points[ii][1] - self.Points[ii - 1][1]) * self.ratio + self.Points[ii - 1][1]]
            # 更新
            self.Points[ii - 1] = pNew
		#停止递归条件：起点与初始终点重合
        if np.sqrt(sum(np.power((self.Points[-1][0] - self.Points[0][0],self.Points[-1][1] - self.Points[0][1]), 2)))>1e-5:
            self.findP()
    def run(self):
        self.findP()
        self.curve.append(self.Points[-1])

def f(start,end,center,left,right,convex=True):#寻找中间点(r0)+控制点(m)朝外还是朝内
    dis1 = [start[0] - end[0],start[1] - end[1]]
    r0 = [dis1[0]*0.5+end[0],dis1[1]*0.5+end[1]]#中间点
    if convex:
        d = np.sqrt(sum(np.power(dis1, 2))) * 0.5#两点距离的一半
    else:
        d = -np.sqrt(sum(np.power(dis1, 2))) * 0.5
    dis2 = [left[0]-right[0],left[1]-right[1]]

    # 控制中间点朝外还是朝内
    if dis1[1]==0:#垂直
        if r0[1]<center[1]:#两点在中心点上方
            d = -d
        m=[r0[0],r0[1]+d]#控制点
    elif dis1[0]==0:#平行
        if r0[0]<center[0]:
            d = -d
        m=[r0[0]+d,r0[1]]
    else:#两点之间为斜线
        if dis2[1]==0 and r0[1] < center[1]:#离散点中最左和最右点连线垂直
            d = -d
        elif dis2[0]==0 and r0[0]<center[0]:#离散点中最左和最右点连线平行
            d = -d
        else:#离散点中最左和最右点连线斜线
            kk = dis2[1] / dis2[0]
            bb = left[1] - kk * left[0]
            if r0[1] >= r0[0] * kk + bb:
                d = -d
				
		#两点连线方程：f(x):kx+b-y=0->w:[k,-1],X:[x1,x2]=[x,y],wX+b=0
        k = dis1[1] / dis1[0]#两点连线斜率 
        b = start[1] - k * start[0]#两点连线相对原点平移量
		#两点连线的垂直线方程：f(x):k0x+b0-y=0->w:[k0,-1],X:[x1,x2]=[x,y],wX+b=0
        k0 = -1 / k#两点连线的垂直线的斜率
        b0 = r0[1] - k0 * r0[0]#两点连线的垂直线相对原点平移量
		#d=(wX+b)/||w||,||w||=(k^2+(-1)^2)^0.5,(wX+b)=(kx0-y0+b),y0=k0x0+b0
        x0 = (np.sqrt(k**2+1)*d-(b-b0))/(k-k0)
        y0 = k0*x0+b0
        m = [x0,y0]
    return m

if __name__=='__main__':
    # baseRatio = 1e-5
    # p0,p2=[1,1],[2,2]
    center = [1.5,1.5]
    left,right = [1,1],[2,2]
    convex =1#凸拟合曲线还是凹拟合曲线
    Points = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    tmp = np.mat(copy.deepcopy(Points))
    curve = []
    for i in range(len(Points)-1):
        p0, p2 = Points[i],Points[i+1]
        p1 = f(p0, p2, center,left,right,convex)
        p=[p0,p1,p2]
        xx = Bezier(p)
        xx.run()
        curve.extend(xx.curve)
    curve = np.array(curve)
    plt.figure(1)
    plt.xlim([min(tmp[:, 0]).getA()[0][0] - 1, max(tmp[:, 0]).getA()[0][0] + 1])
    plt.ylim([min(tmp[:, 1]).getA()[0][0] - 1, max(tmp[:, 1]).getA()[0][0] + 1])
    plt.scatter(sum(tmp[:,0].tolist(),[]),sum(tmp[:,1].tolist(),[]))
    plt.plot(curve[:,0],curve[:,1])
    plt.show()
