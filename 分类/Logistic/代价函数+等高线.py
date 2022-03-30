
import numpy as np
import matplotlib.pyplot as plt
def lossFun():
    x_train = np.array([0.5, 1, 2, 3, 4, 5])
    y_train = np.array([1.1, 2.2, 3.8, 4.1, 4.9, 5.2])

    dense = 200
    k = np.linspace(0, 2, dense)
    b = np.linspace(-2, 4, dense)
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(x_train, y_train, s=4)  # s=0.25比较合适
    # y = kx+b
    def get_loss_value(k, b):
        return np.square(k * x_train + b - y_train).sum() / len(x_train)

    def draw_contour_line(dense, isoheight):  # dense表示取值的密度，isoheight表示等高线的值
        minLoss = 99999
        list_k = []
        list_b = []
        list_loss = []
        for i in range(dense):
            for j in range(dense):
                loss = get_loss_value(k[i], b[j])
                if minLoss>=loss:# 穷举
                    minLoss = loss
                    Jk = k[i]
                    Jb = b[j]
                if 1.05 * isoheight > loss > 0.95 * isoheight:#这里用loss的一个范围表示k,b,是因为dens不够大，如果密度大到无穷，那么这里就可以表示为，round(loss,2) ==isoheight。
                    list_k.append(k[i])
                    list_b.append(b[j])
                    list_loss.append(loss)
                else:
                    pass
        print(minLoss)
        plt.subplot(211)
        x = np.linspace(min(x_train)-1, max(x_train)+1, dense)
        y = Jk*x+Jb
        plt.plot(x,y)


        plt.subplot(212)
        plt.scatter(list_k, list_b, s=0.25)  # s=0.25比较合适
        plt.text(list_k[np.argmin(list_loss)],list_b[np.argmin(list_loss)],str('%.2f'% min(list_loss)))

    draw_contour_line(dense, 0.21)
    draw_contour_line(dense, 0.29)
    draw_contour_line(dense, 0.38)
    draw_contour_line(dense, 0.48)
    draw_contour_line(dense, 0.57)
    draw_contour_line(dense, 0.67)
    draw_contour_line(dense, 0.76)
    draw_contour_line(dense, 0.95)



    plt.title('Loss Func Contour Line')
    plt.xlabel('k')
    plt.ylabel('b')

    # listK = y_train/x_train
    # minK = min(listK)
    # maxK = max(listK)
    # minB = min(y_train)
    # maxB = max(y_train)
    # plt.axis([minK, maxK, minB, maxB])
    # plt.axis([0, 2, -2, 4])
    plt.show()


if __name__=='__main__':
    lossFun()