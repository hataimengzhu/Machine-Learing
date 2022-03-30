def getVectorList():
    print('getVectorList')
    with open('dataList.txt',  'r', encoding='utf-8') as file:
        basicData = file.readlines()
    shujuList = [da.strip().split('\t') for da in basicData]
    with open('chihuiList.txt',  'r', encoding='utf-8') as file:
        basicData = file.readlines()
    chihuiList = [da.strip() for da in basicData]

    shujuVec=[]
    if any(isinstance(i, list) for i in shujuList):
        for i in shujuList:
            tmp = [0]*len(chihuiList)
            print(shujuList.index(i),len(shujuList))
            for j in i:
                if j in chihuiList:
                    tmp[chihuiList.index(j)] = 1
            shujuVec.append(tmp)
    else:
        shujuVec = [0]*len(chihuiList)
        for i in shujuList:
            if i in chihuiList:
                shujuVec[chihuiList.index(i)] = 1

    with open('shujuVec.txt', 'w', encoding='utf-8') as file:
        for i,  value in enumerate(shujuVec):
            print(i, str('/'), len(shujuVec))
            file.write(value + '\n')
    return shujuVec
if __name__=='__main__':
    getVectorList()