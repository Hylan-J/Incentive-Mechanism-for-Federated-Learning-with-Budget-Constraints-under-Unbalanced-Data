import torch.nn as nn
from torch.nn import functional as F
from torchstat import stat


class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1
        # 输入通道1, 输出通道32, 卷积核大小5, 卷积步长1, 填充2
        # size计算: 28+4-5+1=28
        # [1, 28, 28]->[32, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        # 最大池化层1
        # 输入通道32, 输出通道32, 池化核大小2, 池化步长2, 填充0
        # size计算: 28/2=14
        # [32, 28, 28]->[32, 14, 14]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2
        # 输入通道32, 输出通道64, 卷积核大小5, 卷积步长1, 填充2
        # size计算: 14+4-5+1=14
        # [32, 14, 14]->[64, 14, 14]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # 最大池化层2
        # 输入通道64, 输出通道64, 池化核大小2, 池化步长2, 填充0
        # size计算: 14/2=7
        # [64, 14, 14]->[64, 7, 7]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1
        # 输入64*7*7, 输出512
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        # 全连接层2
        # 输入512, 输出10
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        # 根据输入的图像的batchsize, 自动矫正
        tensor = inputs.view(-1, 1, 28, 28)
        # 卷积后, relu激活
        tensor = F.relu(self.conv1(tensor))
        # 最大池化
        tensor = self.pool1(tensor)
        # 卷积后, relu激活
        tensor = F.relu(self.conv2(tensor))
        # 最大池化
        tensor = self.pool2(tensor)
        # 根据输入的图像的batchsize, 自动矫正
        tensor = tensor.view(-1, 7 * 7 * 64)
        # 全连接后, relu激活
        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        # softmax输出
        # tensor = F.softmax(tensor, dim=1)
        tensor = F.log_softmax(tensor, dim=1)
        return tensor


if __name__ == '__main__':
    model = MNIST()
    # 评估模型参数
    stat(model, (1, 28, 28))

    # 保存网络的参数
    # torch.save(model.state_dict(), '../../configs/MNIST.pt')
    # 保存整个网络
    # torch.save(model, '../../configs/MNIST.pt')