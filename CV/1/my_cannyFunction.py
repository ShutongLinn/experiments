# my_cannyFunction.py
# 功能:自定义函数实现canny检测
# 输入:图片img
# 输出:canny检测结果图

from pylab import *
import my_sobelFunction
import my_convolveFunction


# 高斯模糊函数
def Gussianblur(img, sigma):
    # 构建高斯函数
    kernel = np.zeros((3, 3), dtype='float32')
    add = 0
    for y in np.arange(-1, 2):
        for x in np.arange(-1, 2):
            g = (1 / (2 * math.pi * sigma * sigma) * pow(e, -(x * x + y * y) / (2 * sigma * sigma)))
            kernel[x, y] = g
            add += g

    kernel = kernel/add

    output = my_convolveFunction.convolution(img, kernel)

    return output


# 获取梯度和梯度方向，并转化为0度、45度、90度、135度四个方向
def GradGet(img):
    sobelx, sobely, sobel, theta = my_sobelFunction.Sobel_filter(img)

    # 获取图片高度和宽度
    (img_h, img_w) = img.shape[:2]

    for y in np.arange(0, img_h):
        for x in np.arange(0, img_w):
            value = theta[x, y]
            # 0度方向
            if value <= math.pi / 8 and value >= -math.pi/8:
                theta[x, y] = 0
            elif value >= 7*math.pi/8:
                theta[x, y] = 0
            elif value <= -7*math.pi/8:
                theta[x, y] = 0
            # 90度方向
            elif value > (3 * math.pi) / 8 and value < (5 * math.pi) / 8:
                theta[x, y] = 90
            elif value > -5*math.pi/8 and value < -3*math.pi/8:
                theta[x, y] = 90
            # 45度方向
            elif (value > math.pi / 8 and value < (3 * math.pi) / 8) or (
                    value > (-7 * math.pi) / 8 and value < (-5 * math.pi) / 8):
                theta[x, y] = 45
            else:
                theta[x, y] = 135
    output = [sobel, theta]
    return output


# 非极大值抑制
def Suppression(grad, theta):
    # 获取高和宽
    (h, w) = grad.shape[:2]

    grad_new = np.zeros((h, w), dtype='float32')

    for y in np.arange(1, h-1):
        for x in np.arange(1, w-1):
            if theta[x, y] == 0:  # 0度方向
                if grad[x, y] > grad[x, y-1] and grad[x, y] > grad[x, y-1]:
                    grad_new[x, y] = grad[x, y]
            elif theta[x, y] == 90:  # 90度方向
                if grad[x, y] > grad[x-1, y] and grad[x, y] > grad[x+1, y]:
                    grad_new[x, y] = grad[x, y]
            elif theta[x, y] == 45:
                if grad[x, y] > grad[x + 1, y + 1] and grad[x, y] > grad[x - 1, y - 1]:
                    grad_new[x, y] = grad[x, y]
            elif theta[x, y] == 135:
                if grad[x, y] > grad[x + 1, y - 1] and grad[x, y] > grad[x - 1, y + 1]:
                    grad_new[x, y] = grad[x, y]
    return grad_new


# 双阈值处理
def DoubleThreshold(img, high, low):
    # 获取高和宽
    (h, w) = img.shape[:2]

    for y in np.arange(0, h):
        for x in np.arange(0, w):
            if img[x, y] > high:
                img[x, y] = 255
            elif img[x, y] < low:
                img[x, y] = 0
    return img


# 弱化边缘，孤立点检测
def WeakenEdge(img):
    # 获取高和宽
    (h, w) = img.shape[:2]
    img_new1 = img_new2 = img
    for y in np.arange(1, h - 3):
        for x in np.arange(1, w - 3):

            # 判断领域内是否有强边缘
            count = 0
            if img[x, y] != 255 and img[x, y] != 0:
                for j in (0, 4):
                    for i in (0, 4):
                        if img[x - 1 + i, y - 1 + j] == 255:
                            count += 1
                if count >= 4:
                    img_new1[x, y] = 255

            # 判断是否有孤立点
            count = 0
            if img[x, y] == 255:
                for j in (0, 4):
                    for i in (0, 4):
                        if img_new1[x - 1 + i, y - 1 + j] == 0:
                            count += 1
                if count >= 7:
                    img_new2[x, y] = 0

    return img_new2

# 主函数
def Canny_fliter(im):
    (h, w) = im.shape[:2]
    # 高斯模糊
    img = np.zeros((h, w), dtype='float32')
    img = im.copy()
    img_new = Gussianblur(img, 1)

    # 获取梯度和梯度方向
    img_new_copy = np.zeros((h, w), dtype='float32')
    img_new_copy = img_new.copy()
    [grad, theta] = GradGet(img_new_copy)
    # 极大值抑制
    grad_copy = np.zeros((h, w), dtype='float32')
    grad_copy = grad.copy()
    img_grad = Suppression(grad_copy, theta)

    # 双阈值处理
    img_grad_copy = np.zeros((h, w), dtype='float32')
    img_grad_copy = img_grad.copy()
    img_threshold = DoubleThreshold(img_grad_copy, 200, 100)
    # 弱化边缘
    img_threshold_copy = np.zeros((h, w), dtype='float32')
    img_threshold_copy = img_threshold.copy()
    img_weaken = WeakenEdge(img_threshold_copy)

    plt.subplot(2, 3, 1), plt.imshow(img, 'gray'), plt.title('img')
    plt.subplot(2, 3, 2), plt.imshow(img_new, 'gray'), plt.title('Gussianblur')
    plt.subplot(2, 3, 3), plt.imshow(grad, 'gray'), plt.title('Sobel')
    plt.subplot(2, 3, 4), plt.imshow(img_grad, 'gray'), plt.title('Suppression')
    plt.subplot(2, 3, 5), plt.imshow(img_threshold, 'gray'), plt.title('DoubleThreshold')
    plt.subplot(2, 3, 6), plt.imshow(img_weaken, 'gray'), plt.title('WeakenEdge')
    # 调整图片间距
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.3, hspace=0.3)
    plt.show()