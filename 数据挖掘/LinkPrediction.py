import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入数据
ratings_names = ["userId", "movieId", "rating", "timestamp"]
data = pd.read_table('ml-latest-small/ratings.csv', sep=",", engine='python')
ratings = pd.read_table('ml-latest-small/ratings.csv', sep=",", engine='python')
#print('data')
#print(data)

# 打乱数据集
mask = np.random.rand(len(ratings)) < 0.9
# 数据总数
total = len(ratings)
# 训练集、测试集数据
train_set = ratings[mask]
test_set = ratings[~mask]

# 训练集
# 建立邻接矩阵

# 训练集人数
train_users = train_set.userId.unique()
train_users_size = len(train_users)
#print('train_users_size:', train_users_size)
# 电影数
movies = ratings.movieId.unique()
movies_size = len(movies)
#print('movies_size:', movies_size)
# 测试集人数
test_users = test_set.userId.unique()
test_users_size = len(test_users)
#print('test_users_size:', test_users_size)

# 总用户人数
all_user = ratings.userId.unique()
all_users_size = len(all_user)
#print('all_users_size', all_users_size)

# 建立索引
# train_uid_idx = {uid: k   for k, uid in enumerate(train_users)}
# test_uid_idx = {uid: k   for k, uid in enumerate(test_users)}
all_uid_idx = {uid: k  for k, uid in enumerate(all_user)}
mid_inx = {mid: k   for k, mid in enumerate(movies)}

# 构建邻接矩阵
A = np.zeros((all_users_size, movies_size))

for _, rating in train_set.iterrows():
    # print('user:     ',rating.userId)
    # print('movie:    ',rating.movieId)
    # print('评分：     ', rating.rating)
    # print('uid:      ', uid_idx[rating.userId])
    # print('mid:      ', mid_inx[rating.movieId])
    if(rating.rating > 3):
        A[all_uid_idx[rating.userId], mid_inx[rating.movieId]] = 1
#print('A')
#print(A)
# 建立资源配额矩阵
# 求K_movies 和K_users

# K_users--某人评价了多少部电影
k_users = np.zeros(all_users_size)
for i in range(0, train_users_size):
    count = 0
    for j in range(0, movies_size):
        if A[i][j] >0:
            count+=1
    k_users[i] = count
# print(k_users)

# K_movies--某部电影被多少人评价过
k_movies = np.zeros(movies_size)
for i in range(0, movies_size):
    count = 0
    for j in range(0, train_users_size):
        if A[j][i] >0:
            count+=1
    k_movies[i] = count
# print(k_movies)

# 计算W--Wij表示产品j愿意分配给产品i的资源配额
W =np.zeros((movies_size, movies_size))

# 时间长
'''
for i in range(0, movies_size):
    for j in range(0, movies_size):
        sum = 0
        for l in range(0, train_users_size):
            sum += (A[l][i] * A[l][j])/k_users[l]
        sum = sum/k_movies[j]
        W[i][j] = sum
'''
# print(W)

# 矩阵运算
temp = A/k_users.reshape((-1, 1)) #ail/kl
temp[np.isnan(temp)] = 0
W = np.dot(temp.T, A) #ail*ajl/kl
W = W/k_movies # 求得W
W[np.isnan(W)] = 0
h, w = W.shape[:2]
#print(h, w)
#print('W')
#print(W)

# 测试集计算推荐评分矩阵f,f_size = test_size * movies_size

# 建立测试集邻接矩阵
# 构建资源分配矢量f=W*f0
f= np.zeros((all_users_size, movies_size))
f0 = A
f = np.dot(W, f0.T).T
#print("f")
#print(f)

#得分排序
f_sorted = np.zeros((all_users_size, movies_size))
#排序索引
index = np.argsort(f, axis=1)
for i in range(0, all_users_size):
    for j in range(0, movies_size):
        f_sorted[i, index[i][j]] = movies_size - j

#print("f_sorted")
#print(f_sorted)

L = np.zeros((all_users_size))
for i in range(0, all_users_size):
     L[i] = all_users_size - k_users[i]
#print("L")
#print(L)

r = np.zeros((all_users_size, movies_size))

#测试集
B = np.zeros((all_users_size, movies_size))
for _, rating in test_set.iterrows():
    if(rating.rating > 3):
        B[all_uid_idx[rating.userId], mid_inx[rating.movieId]] = 1

for i in range(0, all_users_size):
    for j in range(0, movies_size):
        if B[i][j] > 0:
            r[i][j] = f_sorted[i][j]/L[i]

r_aver = np.average(r)
print("r_aver")
print(r_aver)

#绘制ROC
TPR, FPR = [], []

T_ = np.sum(B)  # 正样本
F_ = np.sum(B == False)  # 负样本

for threshold in np.arange(0, 1, 0.01):
    F_out = f_sorted < (movies_size * threshold)
    F_out = F_out.astype(int)

    TP = np.sum(B * F_out)
    FP = np.sum((1 - B) * F_out)

    TPR.append(TP / T_)
    FPR.append(FP / F_)

plt.plot(FPR, TPR)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xlabel("FP")
plt.ylabel("TP")

plt.show()

# ROC曲线的积分

AUC = np.sum([0.01 * tpr for tpr in TPR])

print(AUC)