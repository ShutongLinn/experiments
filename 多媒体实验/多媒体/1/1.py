dic = eval(input("请输入字典"))
str = input("请输入待编码字符串")


# 二进制转换
def change(num):
    bin = []
    count2 = 0
    while (1):
        num = 2 * num
        count2 += 1
        if(num > 1):
            bin.append(1)
            num -= 1
        else:
            bin.append(0)
        if(count2>16):
            break
        if(num == 0):
            break
    for i in range(0, count2):
        print(bin[i], end='')

# 编码
p_min = [0]
p_max = [1]
count=1

for i in str:
    q_min = 0
    q_max = 0
    if i not in dic:
        print("error")
    else:
        for key in dic:
            if key == i:
                q_max += dic[key]
                len = p_max[count-1] - p_min[count-1]
                # print("len:" and  len)
                # print("p_max[count-1]:" and p_max[count-1])
                # print("p_min[count-1]:"  and  p_min[count - 1])
                q_max = q_max * len + p_min[count-1]
                q_min = q_min * len + p_min[count-1]
                break
            else:
                q_min += dic[key]
                q_max += dic[key]
    p_min.append(q_min)
    p_max.append(q_max)
    # print(i)
    # print(q_min, q_max)
    # print("***********************************")
    count += 1

print("编码结果:")
print(p_min[count-1])
change(p_min[count-1])


# 解码
a = p_min[count-1]
uncode = []

for i in range(0, count-1):
    sum = 0
    for key in dic:
        sum += dic[key]
        if a < sum:
            uncode.append(key)
            a = (a - sum + dic[key]) / dic[key]
            break

print()
print("解码:")
for i in range(0, count-1):
    print(uncode[i], end='')
