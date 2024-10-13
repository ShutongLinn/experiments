import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np

# 定义超参数
BATCH_SIZE = 16  # 每批处理的数据
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU用GPU训练，否则用CPU
EPOCHS = 10  # 训练数据集的次数

# 构建pipeline，对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # (标准差，均值),标准化作用：模型出现过拟合现象（loss低，accuracy低）时，降低模型复杂度
])

# 下载、加载数据
from torch.utils.data import DataLoader

# 下载数据集，自动将数据集下载到指定的MNISTdata文件夹下面
train_set = datasets.MNIST("MNISTdata", train=True, download=True, transform=pipeline)
# transform 对数据集输入进来的每一张图片都进行转换(转换成tensor然后标准化)

test_set = datasets.MNIST("MNISTdata", train=False, download=True, transform=pipeline)

# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# shuffle 对每批的16张图片进行打乱，提高训练精度

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 插入代码，显示图片，rb以二进制格式读取
with open("./MNISTdata/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    file = f.read()
# 提取里面的图片
image1 = [int(str(item).encode('ascii'), 10) for item in file[16: 16 + 784]]
print(len(image1))
print(image1)

image1_np = np.array(image1, dtype=np.uint8).reshape(28, 28, 1)  # 图片是灰度，所以通道数为1  转换成数组，类型是无符号整型，
print(image1_np.shape)  # 打印图片的长宽高等信息
cv2.imwrite("digit.jpg", image1_np)  # 保存图片


# 构建网络模型
class FirstNet(nn.Module):  # 继承父类
    def __init__(self):  # 构造方法
        super(FirstNet, self).__init__()  # 调用父类的构造方法
        # super().__init__()
        #  定义属性方法等
        # 定义卷积层
        self.conv = nn.Sequential(
            # 模型实例输入为32*32像素大小的图片，而MINIST图片大小为28*28，因此对模型做一些改动
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),  # 卷积层 28x28x1-->24x24x6 滤波器尺寸5*5；步长1

            nn.Sigmoid(), # 激活层：使线性关系转换为非线性关系，增强神经网络表达能力

            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层：压缩图片 12x12x6 滤波器尺寸2*2；步长2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 卷积层 8x8x16 滤波器尺寸5*5；步长1

            nn.Sigmoid(), # 激活层

            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层 4x4x16  滤波器尺寸2*2；步长2
        )

        # 定义全连接层,全连接层为线性层；过滤器尺寸5*5
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),# 输出节点数120
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),# 输出节点数84
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10),
        )
        # 前向传播过程
        def forward(self, img):
            feature = self.conv(img)
            img = self.fc(feature.view(img.shape[0], -1))
            output = F.log_softmax(img, dim=1)  # （损失函数）计算分类后，每个数字的概率值
            return output


# 定义优化器
# 更新训练模型
model = FirstNet().to(DEVICE)

optimizer = optim.Adam(model.parameters())  # 优化器作用：更新模型参数，使得测试结果达到最优值


# 定义训练的方法
def train_model(model, device, train_loader, optimizer, epoch):  # 模型， 设备， 数据， 优化器， 循环次数
    # 模型训练
    model.train()
    # target标签：记录实际值
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE上去（cpu/gpu）
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果，预测
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target) # 预测结果output与实际值target差距
        # 方向传播
        loss.backward()
        # 参数优化（更新）
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch :{} \t Loss : {:.6f}".format(epoch, loss.item()))


# 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 测试：不计算梯度，也不进行反向传播
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[1]  # 值，索引  # 1表示横轴，或者 pred = output.argmax(dim=1)
            # pred = torch.max(output, dim=1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset) # 平均损失
        print("Test——Average loss :{:.4f}, Accuracy :{:.3f}\n".format(test_loss, 100 * correct / len(test_loader.dataset)))


# 调用方法
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)

