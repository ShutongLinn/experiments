import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
import csv
import codecs
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# SIFT算法提取一张图的128维描述子向量
def SIFT(img):
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(img, None)
    return features


# 高斯混合模型聚类
# k-means
def Clustering(features, num):
    kmeans = KMeans(n_clusters=num).fit(features)
    labels = kmeans.predict(features)
    centres = kmeans.cluster_centers_

    # gmm = GMM(n_components=num).fit(features)
    # labels = gmm.predict(features)
    # centres = gmm.means_
    return centres, labels


# 存储一张图的特征向量
def Store(features, i):
    mid = pd.DataFrame(features)
    mid.to_csv('features/' + i + '.csv', header=False, index=False)


# 读取一张图的特征向量
def ReadFeatures(i):
    mid = np.loadtxt('features/' + i + '.csv', delimiter=',', dtype=float)
    node_pair = []
    for item in mid:
        node_pair.append(list(item))
    node_pair = np.array(node_pair)
    return node_pair


# 求余弦相似度
def Cosine(a, b):
    cos = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos


# 构建一张图直方图
def Histogram(features_single, centres):
    histo = np.zeros(centres.shape[0])

    for j in range(0, centres.shape[0]):
        count = 0
        c = centres[j]
        for k in range(0, features_single.shape[0]):
            f = features_single[k]
            if Cosine(c, f) > 0.9:
                count += 1

        histo[j] = count

    return histo


# 存储向量
'''count = 0
for i in range(0, 1000):
    first = int(i/100)
    img = cv2.imread("corel/" + str(first) + "/" + str(i) + ".jpg", -1)
    features = SIFT(img)
    Store(features, str(i))
    count += features.shape[0]
    print(first, i, end=None)
    print(" Done!")

print(count)
print("**********************")'''
count = 695661

# 读取所有特征向量
'''all_features = np.zeros([count, 128])
p = 0
for i in range(0, 1000):
    temp = ReadFeatures(str(i))
    for j in range(0, temp.shape[0]):
        all_features[p] = temp[j]
        p += 1
    print(i)

Store(all_features, 'all_features')
print("store done!")'''

# all_features = ReadFeatures('all_features')

# 将所有特征向量进行聚类
'''centres, labels = Clustering(all_features, 700)
Store(centres, '700Cluster_centres')
Store(labels, '700Cluster_labels')
print('700cluster done!')'''

centres = ReadFeatures('700Cluster_centres')

# 构建直方图
'''histo_all = []
for i in range(0, 1000):
    features_single = ReadFeatures(str(i))
    histo = Histogram(features_single, centres)
    histo_all.append(histo)
    print(i)
Store(histo_all, 'histo/all_700_0.95')
print("histo done!")'''

histo_all = ReadFeatures('histo/all_700_0.9')


def Draw(input, pic, title):
    fig = plt.figure(figsize=(12, 4), dpi=200)
    plt.subplot(2, 6, 1), plt.imshow(input[..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[10])
    plt.subplot(2, 6, 2), plt.imshow(pic[0][..., ::-1], cmap='gray'), plt.axis('off')
    plt.subplot(2, 6, 3), plt.imshow(pic[1][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[0])
    plt.subplot(2, 6, 4), plt.imshow(pic[2][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[1])
    plt.subplot(2, 6, 5), plt.imshow(pic[3][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[2])
    plt.subplot(2, 6, 6), plt.imshow(pic[4][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[3])
    plt.subplot(2, 6, 7), plt.imshow(pic[5][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[4])
    plt.subplot(2, 6, 8), plt.imshow(pic[6][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[5])
    plt.subplot(2, 6, 9), plt.imshow(pic[7][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[6])
    plt.subplot(2, 6, 10), plt.imshow(pic[8][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[7])
    plt.subplot(2, 6, 11), plt.imshow(pic[9][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[8])
    plt.subplot(2, 6, 12), plt.imshow(pic[10][..., ::-1], cmap='gray'), plt.axis('off'), plt.title(title[9])
    plt.show()


def BOF(centres, histo_all, index):
    # 输入图像
    input = cv2.imread("test/" + str(index) + "/2.jpg", -1)
    input_features = SIFT(input)
    input_histo = Histogram(input_features, centres)

    # 计算相似度
    sim = []
    for k in range(0, 1000):
        h = histo_all[k]
        i = input_histo
        sim.append(Cosine(h, i))

    # 找出相似度最大的11张图
    large_index = []
    large_value = []
    num = -1

    for i in range(0, 11):
        max_index = 0
        max_value = 0
        for j in range(0, 1000):
            if num == -1:
                if max_value < sim[j]:
                    max_value = sim[j]
                    max_index = j
                continue
            if num != -1:
                if max_value < sim[j] and sim[j] < large_value[num]:
                    max_value = sim[j]
                    max_index = j

        large_index.append(max_index)
        large_value.append(max_value)
        num += 1

    print(large_index)

    pic = []
    for i in range(0, 11):
        first = int(large_index[i] / 100)
        p = cv2.imread("corel/" + str(first) + '/' + str(large_index[i]) + '.jpg', -1)
        pic.append(p)

    acc_count = 0
    title = []
    for i in range(1, 11):
        if int(large_index[i] / 100) == index:
            acc_count += 1
            title.append(" ")
        else:
            title.append("wrong")

    acc = acc_count / 10
    title.append('acc:' + str(acc))
    return input, pic, acc, title


for i in range(0, 10):
    input, pic, acc, title = BOF(centres, histo_all, i)
    print("index:", i, "acc:", acc)
    print(" ")
    Draw(input, pic, title)

'''def BOF(centres, histo_all, index):
    acc = []
    for j in range(0, 100):
        # 输入图像
        second = index*100 + j
        input_features = ReadFeatures(str(second))
        input_histo = Histogram(input_features, centres)

        # 计算相似度
        sim = []
        for k in range(0, 1000):
            h = histo_all[k]
            i = input_histo
            sim.append(Cosine(h, i))

        # 找出相似度最大的11张图
        large_index = []
        large_value = []
        num = -1

        for i in range(0, 11):
            max_index = 0
            max_value = 0
            for j in range(0, 1000):
                if num == -1:
                    if max_value < sim[j]:
                        max_value = sim[j]
                        max_index = j
                    continue
                if num != -1:
                    if max_value < sim[j] and sim[j] < large_value[num]:
                        max_value = sim[j]
                        max_index = j

            large_index.append(max_index)
            large_value.append(max_value)
            num += 1


        acc_count = 0
        for i in range(1, 11):
            if int(large_index[i] / 100) == index:
                acc_count += 1

        acc.append(acc_count / 10)

    max = 0
    max_index = 0
    for j in range(0, 100):
        if max < acc[j]:
            max = acc[j]
            max_index = j


    return max, max_index


for i in range(0, 10):
    max, max_index = BOF(centres, histo_all, i)
    print(i, "==", "acc:", max, "pic_index:", max_index)'''
