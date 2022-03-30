# import numpy as np
def getVocabulary():

    with open('cleanList.txt',  'r', encoding='utf-8') as file:
        basicData = file.readlines()
    cleanList = [da.strip().split('\t') for da in basicData]
    chihuiList = set()
    for i,  value in enumerate(cleanList):
        print(i, str('/'), len(cleanList))
        chihuiList = chihuiList|set(value)
    chihuiList = list(chihuiList)
    # np.save('chihuiList.npy',chihuiList)
    # chihuiList = np.load('toutiaoChiHuiList.npy', allow_pickle=True).tolist()
    with open('chihuiList.txt', 'w', encoding='utf-8') as file:
        for i,  value in enumerate(chihuiList):
            print(i, str('/'), len(chihuiList))
            file.write(value + '\n')
if __name__=='__main__':
    getVocabulary()