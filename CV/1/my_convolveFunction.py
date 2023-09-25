# my_convolveFunction.py
# 功能:实现卷积操作
# 输入:图片img,卷积核kernel
# 输出:卷积后的图片

from pylab import *
import cv2

# 自定义二维图像卷积
def convolution(img, kernel):
    # 卷积核翻转
    kernel = kernel[::-1, ::-1]

    # 读取图片和卷积核高、宽
    (img_h, img_w) = img.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]

    # 边缘填充(复制填充)
    pad = (kernel_h - 1) // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # 构建空的输出图像
    output = np.zeros((img_h, img_w), dtype='float32')

    # 遍历图片，依次移动卷积核
    for y in np.arange(pad, img_h + pad):
        for x in np.arange(pad, img_w + pad):
            # 求运算区域box
            box = img[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # 滤波后(x,y)坐标值
            value = (box * kernel).sum()

            output[y - pad, x - pad] = value

    return output
