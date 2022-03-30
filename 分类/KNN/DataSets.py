import pandas as pd
import numpy as np

# 从互联网读入手写体图片识别任务的训练数据
# digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
#                            header=None)

# def loadTextData(path): # path:文件路径, featureNum：特征种类
#     digits_train = pd.read_csv(path,header=None)
#     trainTexts = digits_train[np.arange(64)]
#     labelTexts = digits_train[64]
#     TextsNames = np.zeros((trainTexts.shape[0],1))
#
#
#     for i in range(trainTexts.shape[0]):
#         trainMatrix = np.zeros((40, 1))
#         for j in range(0, 64, 8):
#             re = ''
#             for d in range(8):
#                 x = cover(trainTexts[i][j+d], 2, 5)
#                 re += x
#             trainMatrix[j//8][0] = re
#             print(trainMatrix[j//8][0])
#         print(trainMatrix[j//8][0])
def loadTextData(path): # path:文件路径, featureNum：特征种类
    digits_train = pd.read_csv(path,header=None)
    trainTexts = digits_train[np.arange(64)]
    labelTexts = digits_train[64]
    TextsNames = np.zeros((trainTexts.shape[0],1))
    trainMatrix = np.random.randint([8,4,4])

    for i in range(trainTexts.shape[0]):
        for j in range(0, 64):
            trainMatrix[j//8]





def cover(dec, n, bit):
    def tem(dec, n):
        re = ''
        if dec:
            re = tem(dec//2, n)
            return re +str(dec%n)
        else:
            return re

    result = tem(dec, n)
    if dec==0:
        result = '0'*bit + result
        return result
    if len(result)<bit:
        result = '0'*(bit-len(result)) + result
        return result
    else:
        return result

def loadTextImg(path):
    # 样本数据
    trainFilesList = os.listdir(path + '\\' + 'trainingDigits')
    trainFileNum = len(trainFilesList)
    trainFileLabels = []
    trainMatrix = np.zeros((trainFileNum, 1024))
    for n in range(trainFileNum):
        trainFileLabels.append(trainFilesList[n].split('_')[0])
        file = open(path + '\\' + 'trainingDigits' + '\\' + trainFilesList[n])
        for i in range(32):
            fileDatas = file.readline()
            fileDatas = fileDatas.strip()
            for j in range(32):
                trainMatrix[n,j+i*32] = fileDatas[j]

    # 测试数据
    accuracy = 0
    testFilesList = os.listdir(path + '\\' + 'testDigits')
    testFileNum = len(testFilesList)
    testFileLabels = []
    testMatrix = np.zeros((testFileNum, 1024))
    for n in range(testFileNum):
        testFileLabels.append(testFilesList[n].split('_')[0])
        file = open(path + '\\' + 'testDigits' + '\\' + testFilesList[n])
        for i in range(32):
            fileDatas = file.readline()
            fileDatas = fileDatas.strip()
            for j in range(32):
                testMatrix[n,j+i*32] = fileDatas[j]
        result = classifyKNN(testMatrix[n], trainMatrix, trainFileLabels, 3)
        print('result:', result,'right:', testFileLabels[n])
        if result == testFileLabels[n]:
            accuracy += 1
    print('ErrorRadio:{:.1%}'.format(accuracy/testFileNum))

def ImgToText(path):
    imgFilesList = os.listdir(path + '\\' + 'img')
    imgNum = len(imgFilesList)
    for i in range(imgNum):
        img = Image.open(path + '\\' + 'img' + '\\' + imgFilesList[i])
        # 转换图片为RGBA模式+32*32尺寸
        img = img.convert('RGB')
        img = img.resize((32, 32), Image.LANCZOS)
        imgSize = img.size
        # img.save('test.png')
        imgName = imgFilesList[i].split('.')[0]
        with open(path + '\\testDigits\\' + imgName + '.txt','wb') as f:
            for x in range(imgSize[0]):
                for y in range(imgSize[1]):
                    tmp = img.getpixel((y,x))
                    if(tmp != (255,255,255)):
                        if y != 0 and y % 31 ==0 and x<imgSize[0]-1:
                            f.write(b'1\n')
                        else:
                            f.write(b'1')
                    else:
                        if y != 0 and y % 31 ==0 and x<imgSize[0]-1:
                            f.write(b'0\n')
                        else:
                            f.write(b'0')

if __name__=='__main__':
    dataSetPath = 'optdigits.tra'
    # print(cover(15, 2, 5))
    loadTextData(dataSetPath)
