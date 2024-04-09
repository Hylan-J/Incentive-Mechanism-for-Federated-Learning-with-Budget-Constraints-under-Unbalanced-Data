import torch.nn as nn
from torch.nn import functional as F
from torchstat import stat


class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1
        # 输入通道3, 输出通道32, 卷积核大小5, 卷积步长1, 填充2
        # size计算: 32+4-5+1=32
        # [3, 32, 32]->[32, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # 最大池化层1
        # 输入通道32, 输出通道32, 池化核大小2, 池化步长2, 填充0
        # size计算: 32/2=16
        # [32, 32, 32]->[32, 16, 16]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2
        # 输入通道32, 输出通道64, 卷积核大小5, 卷积步长1, 填充2
        # size计算: 16+4-5+1=16
        # [32, 16, 16]->[64, 16, 16]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # 最大池化层2
        # 输入通道64, 输出通道64, 池化核大小2, 池化步长2, 填充0
        # size计算: 16/2=8
        # [64, 16, 16]->[64, 8, 8]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1
        # 输入64*8*8, 输出512
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # 全连接层2
        # 输入512, 输出10
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        # 根据输入的图像的batchsize, 自动矫正
        tensor = inputs.view(-1, 3, 32, 32)
        # 卷积后, relu激活
        tensor = F.relu(self.conv1(tensor))
        # 最大池化
        tensor = self.pool1(tensor)
        # 卷积后, relu激活
        tensor = F.relu(self.conv2(tensor))
        # 最大池化
        tensor = self.pool2(tensor)
        # 根据输入的图像的batchsize, 自动矫正
        tensor = tensor.view(-1, 64 * 8 * 8)
        # 全连接后, relu激活
        tensor = F.relu(self.fc1(tensor))
        # 全连接后, relu激活
        tensor = F.relu(self.fc2(tensor))
        # softmax输出
        # tensor = F.softmax(tensor, dim=1)
        tensor = F.log_softmax(tensor, dim=1)
        return tensor


if __name__ == '__main__':
    model = CIFAR10()
    stat(model, (3, 32, 32))

    # 保存网络的参数
    # torch.save(model.state_dict(), '../../configs/CIFAR_10.pt')
    # 保存整个网络
    # torch.save(model, '../../configs/CIFAR_10.pt')
