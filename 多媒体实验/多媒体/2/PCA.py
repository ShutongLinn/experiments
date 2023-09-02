import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

# 数据集Array从32维降到k维
def My_PCA(Array, k):
    # 样本中心化
    u = Array.mean(0)# 求各列平均值，返回68040*1矩阵
    Centra = np.subtract(Array, u) #减去向量，u自动扩充,68040*32
    #print("*************Centra*************")
    #print(Centra.shape[0], Centra.shape[1])

    # 协方差矩阵
    Cov = np.dot(Centra.T, Centra) * (1 / Centra.shape[0])#32*32
    print("************Cov************")
    print(Cov)
    #print("*************Cov*************")
    #print(Cov.shape[0], Cov.shape[1])

    # 计算特征值和特征向量
    W, V = np.linalg.eig(Cov)
    #print("*************V*************")
    #print(V.shape[0], V.shape[1])

    #数组降序前k个值的索引
    index = np.argsort(-W)[:k]

    #求投影矩阵P
    P = np.zeros((V.shape[0], k))
    for i in range(V.shape[0]):
        for j in range(k):
            P[i][j] = V[i, index[j]]

    #print("*************P*************")
    #print(P.shape[0], P.shape[1])
    #print(P)

    # 降维结果
    Output = np.dot(P.T, Centra.T) # 5*68040
    print("***************output*****************")
    print(Output.T)

    Cov_new = np.cov(Output)

    print("************Cov_new************")
    print(Cov_new)

# 数据读取
data_read = pd.read_csv('ColorHistogram.asc', delim_whitespace=True, header=None, index_col=0)
list = data_read.values.tolist()
Array = np.array(list)

# 调用库
pca = PCA(n_components=5)
pca.fit(Array)

print("************Reference************")
print(pca.transform(Array))

# 实现
My_PCA(Array, 5)
