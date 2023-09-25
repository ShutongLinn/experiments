from pylab import *
import cv2
import my_sobelFunction
import my_cannyFunction
import matplotlib.pyplot as plt

def SobelResult(img):
    # 获取x方向梯度, y方向梯度, 梯度幅度，梯度角度
    convolveSobelX, convolveSobelY, convolveSobel, theta = my_sobelFunction.Sobel_filter(img)

    # theta归一化处理
    (h, w) = theta.shape[:2]
    max = 0
    min = 0
    for i in np.arange(0, h):
        for j in np.arange(0, w):
            theta[i, j] = (180*theta[i, j])//math.pi
    print(theta)

    # 展示图片
    plt.subplot(2, 2, 1), plt.imshow(convolveSobelX, 'gray'), plt.title('SobelX')
    plt.subplot(2, 2, 2), plt.imshow(convolveSobelY, 'gray'), plt.title('SobelY')
    plt.subplot(2, 2, 3), plt.imshow(convolveSobel, 'gray'), plt.title('Sobel')
    plt.subplot(2, 2, 4), plt.imshow(theta), plt.title('theta')
    # 调整图片间距
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0.3)
    plt.show()

# 图像的读取和转灰度图
img = cv2.imread('/Users/linshutong/Desktop/lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
SobelResult(img)
my_cannyFunction.Canny_fliter(img)