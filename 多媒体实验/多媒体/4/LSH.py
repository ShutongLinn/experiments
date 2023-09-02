'''在corel数据集上实现LSH索引，并分别进行近邻搜索，查询数据集前1000点，
查找前10个最近邻，统计搜索算法的性能(召回率，准确率，时间)'''
import time

import numpy as np
from numpy import dot
from numpy.random import random
import random as ran
import pandas as pd
from collections import Counter

#计时
time_start = time.time()

# 导入数据
input = np.loadtxt('corel', delimiter=' ', dtype=float)
data = []
for item in input:
    data.append(list(item))

data = np.array(data).T
data = data[1:].T
# print(data)
h = data.shape[0]
w = data.shape[1]


# 投影函数 读取高维数据，投影到一条直线上 v_new = av+b
def Project(data, a, b):
    P = dot(data, a) + b
    return P

# 行数表示投影次数； 列数表示点
def AND(data, times):
    h = data.shape[0]
    w = data.shape[1]
    array_and = np.zeros([times, h])
    for i in range(0, times):
        a = random([w, 10])
        b = ran.randint(0, 1)
        output = Project(data, a, b)

        for j in range(0, h):
            array_and[i][j] = output[j][0]
    return array_and

# 两点投影到同一份线段后的距离 d = (v1_new - v2_new)/node; node表示一条线段内可容纳的点数,这里node的引入表示node越大，投影后的距离越近
def Possibility(array, node, k):
    array_new = (array / node).T
    array_new = np.around(array_new, decimals=0)# 取整

    index_and = []
    for i in range(0, k):
        row = array_new[i]
        temp = np.argwhere((array_new == row).all(1))
        temp = list(temp)

        index_and.append(temp)
    return index_and


def OR(data, node, bands, k):
    index_all = []
    for i in range(0, bands):
        ##
        ##
        array_and = AND(data, 10)
        index_and = Possibility(array_and, node, k)
        index_all.append(index_and)
    time_end = time.time()
    delta_time = time_end - time_start
    print("all AND 用时:", delta_time)
    index_all = np.array(index_all)

    index_result = []
    for i in range(0, index_all.shape[1]):
        temp = list()
        for j in range(0, index_all.shape[0]):
            if index_all[j][i] != []:
                temp = np.append(temp, index_all[j][i])
        temp = Counter(temp)
        temp = sorted(temp.items(), key=lambda s: (-s[1]))
        index_result.append(temp)
    time_end = time.time()
    delta_time = time_end - time_start
    print("all OR 用时:", delta_time)

    return index_result

##
##
'''out = OR(data, 0.2, 1200, 1000)
out = np.array(out)


index_uncorrect = []
output = []
for i in range(0, 1000):
    count = 0
    temp_index = []
    for j in range(0, len(out[i])):
        if j > 10:
            break
        if j != 0:
            temp_index.append(out[i][j][0])
            count += 1
    if count != 10:
        index_uncorrect.append(i)
    output.append(temp_index)

print(index_uncorrect)
time_end = time.time()
delta_time = time_end - time_start
print("all  用时:", delta_time)

# 存储
##
##
mid = pd.DataFrame(output)
mid.to_csv('0.2-10-1200.csv', header=False, index=False)'''

#读取 数据
mid = np.loadtxt('0.15-10-1200.csv', delimiter=',', dtype=float)
output = []
for item in mid:
    output.append(list(item))
output = np.array(output)


# 求准确率
# 导入数据
resu = np.loadtxt('real_index.txt', delimiter=' ', dtype=float)
result = []
for item in resu:
    result.append(list(item))

correct_num = 0
TP = 0
FP = 0
FN = 0
TN = 0
for i in range(0, 1000):
    temp = list(set(result[i]).intersection(set(output[i])))
    TP += len(temp)
    FN += 10 - len(temp)
    FP += 10 - len(temp)

TN = 990*h - FP
print("召回率: ", TP / (1000*10))
print("准确率: ", (TP + TN) / (1000*h))