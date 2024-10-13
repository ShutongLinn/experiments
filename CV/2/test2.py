# 主函数
import cv2
import numpy as np
import img_stitch
import matplotlib.pyplot as plt

img1 = cv2.imread('a.jpg')
img2 = cv2.imread('b.jpg')

# 两张重置成同样尺寸
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

# 两张图压到栈内，进行拼接
inputs = np.hstack((img1, img2))

# 获取单应性矩阵
H = img_stitch.get_homo(img1, img2)

# 图像变换进行拼接
result_img = img_stitch.stitch_img(img1, img2, H)

plt.subplot(1, 2, 1), plt.imshow(img1[:, :, ::-1]), plt.title('img-left')
plt.subplot(1, 2, 2), plt.imshow(img2[:, :, ::-1]), plt.title('img-right')
plt.show()

cv2.imshow('output', result_img)
cv2.waitKey()