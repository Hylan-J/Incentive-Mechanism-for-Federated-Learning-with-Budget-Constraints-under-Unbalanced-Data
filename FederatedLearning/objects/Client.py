from torch import optim, nn
from torch.utils.data import DataLoader


class Client:
    def __init__(self, id, trainset, device, local_net, hyperparameter):
        self.id = id
        # 客户端数据集
        self.trainset = trainset
        # 客户端使用设备
        self.device = device
        # 客户端网络模型
        self.local_net = local_net.to(self.device)
        # 客户端相关超参数
        self.learning_rate = hyperparameter["learning_rate"]
        self.batch_size = hyperparameter["batch_size"]
        self.train_epochs = hyperparameter["local_epochs"]
        # 客户端使用交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        # 客户端使用随机梯度下降优化器
        self.optimizer = optim.SGD(self.local_net.parameters(), lr=self.learning_rate)

    def train(self):
        train_dataloader = DataLoader(dataset=self.trainset, batch_size=self.batch_size, shuffle=True)
        self.local_net.train()
        loss = 0
        for epoch in range(self.train_epochs):
            for images, labels in train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                # 前向传播,即网络如何根据输入得到输出的
                outputs = self.local_net(images)
                # loss计算
                loss = self.loss_fn(outputs, labels)
                # 反向传播与优化,反向传播算法的核心是代价函数对网络中参数（各层的权重和偏置）的偏导表达式和。
                self.optimizer.zero_grad()  # 梯度清零：重置模型参数的梯度。默认是累加，为了防止重复计数，在每次迭代时显式地将它们归零。
                loss.backward()  # 反向传播计算梯度：计算当前张量w.r.t图叶的梯度。
                self.optimizer.step()  # 参数更新：根据上面计算的梯度，调整参数

        return loss
