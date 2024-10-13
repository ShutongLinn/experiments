import cv2
import numpy as np
import matplotlib.pyplot as plt

# 找特征点，描述子，计算单应性矩阵
def get_homo(img1, img2):
    # 创建特征转换对象
    sift = cv2.SIFT_create()

    # 获取特征点和描述子
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # 创建特征匹配器(暴力特征匹配)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    # 过滤特征，找到最有效的特征匹配点
    # 设定阈值
    verify_ratio = 0.8
    verify_matches = []
    # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
    for m1, m2 in matches:
        if m1.distance < verify_ratio * m2.distance:
            verify_matches.append(m1)

    # 匹配点多于8个进行拼接
    min_matches = 8
    if len(verify_matches) > min_matches:
        img1_pts = []
        img2_pts = []

        for m in verify_matches:
            # 获取匹配点坐标数组[(x1, y1), (x2, y2),...]
            img1_pts.append(k1[m.queryIdx].pt)
            img2_pts.append(k2[m.trainIdx].pt)
            # print(img1_pts)

        img1_pts = np.array(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.array(img2_pts).reshape(-1, 1, 2)

        # 计算视角变换矩阵
        # cv2.findHomography():计算多个二维点对之间的最优单映射变换矩阵 H,使用最小均方误差或者RANSAC方法
        # 作用:利用基于RANSAC的鲁棒算法选择最优的配对点，再计算转换矩阵H并返回,以便于反向投影错误率达到最小
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return H
    # 配对小于等于8时，error
    else:
        print('err:Not enough matches!')
        exit()

# 图像变换拼接
def stitch_img(img1, img2, H):
    #获取原始图像的宽度和高度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 获得每张图像的四个角点
    # [左上角[0, 0]， 左下角[0, h], [w, h], [w, 0]]
    img1_dims = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # 获取变换后的坐标
    img1_transform = cv2.perspectiveTransform(img1_dims, H)
    img2_transform = cv2.perspectiveTransform(img2_dims, H)
    print('img1_dims:\n', img1_dims)
    print('**********************')
    print('img2_dims:\n', img2_dims)
    print('**********************')
    print('img1_transform:\n', img1_transform)

    # 创建空的大图，拼接图像(横向拼接)
    result_dims = np.concatenate((img2_dims, img1_transform), axis = 0)
    # print(result_dims)

    # x、y方向的最小值和最大值
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel()-0.5)# 向下取整
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel()+0.5)# 向上取整

    # 平移的距离(正值)
    transform_dist = [-x_min, -y_min]

    # 乘以矩阵
    # [1, 0, dx]
    # [0, 1, dy]
    # [0, 0, 1 ]
    # 达到平移效果
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    result_img = cv2.warpPerspective(img1, transform_array.dot(H), (x_max-x_min, y_max-y_min))

    result_img[transform_dist[1]:transform_dist[1]+h2, transform_dist[0]:transform_dist[0]+w2] = img2

    return result_img