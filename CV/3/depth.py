import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img_left = cv2.imread('L.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('R.png', cv2.IMREAD_GRAYSCALE)

(left_h, left_w) = img_left.shape[:2]

plt.subplot(1, 2, 1), plt.imshow(img_left, 'gray'), plt.title('Left')
plt.subplot(1, 2, 2), plt.imshow(img_right, 'gray'), plt.title('Right')
plt.show()

# 求视差图
stereo = cv2.StereoBM_create(numDisparities=80, blockSize=25)

disparity = stereo.compute(img_left, img_right)

im = plt.imshow(disparity)
plt.colorbar(im)
plt.title('parallax_img')
plt.show()

# 基线距离
baseline = 500
# 焦距归一化
f = 7/(6.4 * 1e-3)
# f = 7

# 初始化深度图矩阵
depth = np.zeros((left_h, left_w), dtype='uint16')

for i in range(0, left_h):
    for j in range(0, left_w):
        if disparity[i, j] != 0:
            num = (f * baseline)/disparity[i, j]
            depth[i, j] = num.astype('uint16')

depth = depth.astype('uint16')

plt.imshow(depth, 'gray')
plt.title('depth_img')
plt.show()