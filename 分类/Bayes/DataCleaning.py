def cleanData():
    print('cleanData')
    with open('shujuList.txt',  'r', encoding='utf-8') as file:
        basicData = file.readlines()
    shujuList = [da.strip().split('\t') for da in basicData]
    # 过滤stopwords+数字+符号+len(词)==1
    # 加载stopwords
    with open("stopwords.txt",'r', encoding='utf-8') as file:
        stopWords = [da.strip() for da in file.readlines()]
    # 过滤stopwords+数字+符号+len(词)==1
    cleanList = [0]*len(shujuList)
    if any(isinstance(i, list) for i in shujuList): # 判断是否为嵌套列表
        for i, value in enumerate(shujuList):
            print(str(i) + '/' + str(len(shujuList)))
            tmp = []
            for j in value:
                if j not in stopWords and j != ' ' and len(j) > 1 and not j.isdigit():
                    tmp.append(j)
            cleanList[i] = tmp
    else:
        tmp = []
        for i in shujuList:
            if i not in stopWords and i != ' ' and len(i) > 1 and not i.isdigit():
                tmp.append(i)
        cleanList = tmp

    with open('cleanList.txt', 'w', encoding='utf-8') as file:
        for da in cleanList:
            file.write(da + '\n')

    return cleanList
if __name__=='__main__':
    cleanData()