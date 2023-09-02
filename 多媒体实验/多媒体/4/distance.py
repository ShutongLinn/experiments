import heapq
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

dis = dot(data, data.T)

number = []
for i in range(0, 1000):
    l = list(dis[i])
    l = sorted(enumerate(l), key=lambda  x:x[1])
    min_number = [x[0] for x in l]
    number.append(min_number[:11])
    print(i, " static")

print(number)

mid = pd.DataFrame(number)
mid.to_csv('real_index.txt', header=False, index=False)