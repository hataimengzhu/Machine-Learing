import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
def ImgToText(path):
    imgFilesList = os.listdir(path)
    imgNum = len(imgFilesList)
    imgMatrix = {}#图像矩阵容器
    reSize = 512
    for i in range(imgNum):
        imgMatrix[i]=[]#图像矩阵容器初始化
        img = Image.open(path +'\\' + imgFilesList[i])
        # 转换图片为RGBA模式+reSize*reSize尺寸
        img = img.convert('RGB')
        img = img.resize((reSize, reSize), Image.LANCZOS)
        imgSize = img.size
        # img.save(r'.\Data\test'+str(i)+'.png')
        #写入文件
        # imgName = imgFilesList[i].split('.')[0]
        # with open(path + imgName + '.txt','wb') as f:
        #     for x in range(imgSize[0]):
        #         for y in range(imgSize[1]):
        #             tmp = img.getpixel((y,x))
        #             if(tmp != (255,255,255)):
        #                 if y ==imgSize[0]-1:
        #                     f.write(b'1\n')
        #                 else:
        #                     f.write(b'1')
        #             else:
        #                 if y ==imgSize[0]-1:
        #                     f.write(b'0\n')
        #                 else:
        #                     f.write(b'0')
        #写入矩阵
        for x in range(imgSize[0]):
            for y in range(imgSize[1]):
                pixel = img.getpixel((x,y))
                if(pixel!= (255,255,255)):
                    imgMatrix[i].append([x,-y])
        imgMatrix[i] = np.mat(imgMatrix[i])
        np.save('./Data/imtTest'+str(i)+'.npy', imgMatrix[i])
        plt.figure(i)
        plt.scatter(imgMatrix[i][:,0].tolist(),imgMatrix[i][:,1].tolist(),s=1)
        plt.show()

if __name__=='__main__':
    filePath = r'.\Data\img'
    ImgToText(filePath)