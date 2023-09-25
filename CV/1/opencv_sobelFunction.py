# opencv_sobelFunction.py
# 功能:利用openCV自带函数实现sobel滤波
# 输入:图片img
# 输出:滤波后的图片

import cv2

def sobelFunction(img):
    x = cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)
    y = cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)
    x = cv2.convertScaleAbs(x)
    y = cv2.convertScaleAbs(y)
    new = cv2.addWeighted(x, 0.5, y, 0.5, 0)
    return new
