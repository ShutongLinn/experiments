# my_sobelFunction.py
# 功能:自定义sobel滤波
# 输入:图片img
# 输出:[x方向梯度, y方向梯度, 梯度幅度, 梯度角度]

from pylab import *
import my_convolveFunction

def Sobel_filter(img):
    # 获取图像高和宽
    (img_h, img_w) = img.shape[:2]

    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ), dtype='int')

    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ), dtype='int')

    convolveSobelX = my_convolveFunction.convolution(img, sobelX)
    convolveSobelY = my_convolveFunction.convolution(img, sobelY)

    convolveSobel = fabs(convolveSobelY) + fabs(convolveSobelX)

    # 构建梯度函数
    theta = np.zeros((img_h, img_w), dtype='float32')
    # 获取梯度角度
    for y in np.arange(0, img_h):
        for x in np.arange(0, img_w):
            if convolveSobelX[x, y] != 0:
                sobel_angle = math.atan( convolveSobelY[x, y]/convolveSobelX[x, y] )
            else:
                sobel_angle = math.pi/2
            theta[x, y] = sobel_angle

    output = [convolveSobelX, convolveSobelY, convolveSobel, theta]

    return output
