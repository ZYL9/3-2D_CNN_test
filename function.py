import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


# class HybridSN(nn.Module):
#     def __init__(self):
#         super(HybridSN, self).__init__()
#         self.conv3d_1 = nn.Sequential(
#             nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True),
#         )
#         self.conv3d_2 = nn.Sequential(
#             nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True),
#         )
#         self.conv3d_3 = nn.Sequential(
#             nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv2d_4 = nn.Sequential(
#             nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.fc1 = nn.Linear(18496, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 16)
#         self.dropout = nn.Dropout(p=0.4)
#
#     def forward(self, x):
#         out = self.conv3d_1(x)
#         out = self.conv3d_2(out)
#         out = self.conv3d_3(out)
#         out = self.conv2d_4(out.reshape(out.shape[0], -1, 19, 19))
#         out = out.reshape(out.shape[0], -1)
#         out = F.relu(self.dropout(self.fc1(out)))
#         out = F.relu(self.dropout(self.fc2(out)))
#         out = self.fc3(out)
#         return out

class Attention_Block(nn.Module):

    def __init__(self, planes, size):
        super(Attention_Block, self).__init__()

        self.globalAvgPool = nn.AvgPool2d(size, stride=1)

        self.fc1 = nn.Linear(planes, round(planes / 16))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(round(planes / 16), planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.globalAvgPool(x)
        out = out.view(out.shape[0], out.shape[1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        out = out.view(out.shape[0], out.shape[1], 1, 1)
        out = out * residual

        return out


class HybridSN(nn.Module):

    def __init__(self):
        super(HybridSN, self).__init__()
        # 3个3D卷积
        # conv1：（1, 30, 25, 25）， 8个 7x3x3 的卷积核 ==> （8, 24, 23, 23）
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        # conv2：（8, 24, 23, 23）， 16个 5x3x3 的卷积核 ==>（16, 20, 21, 21）
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        # conv3：（16, 20, 21, 21），32个 3x3x3 的卷积核 ==>（32, 18, 19, 19）
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        # 二维卷积：（576, 19, 19） 64个 3x3 的卷积核，得到 （64, 17, 17）
        self.conv4d_2 = nn.Sequential(
            nn.Conv2d(576, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 注意力机制部分
        self.layer1 = self.make_layer(Attention_Block, planes=576, size=19)
        self.layer2 = self.make_layer(Attention_Block, planes=64, size=17)

        # 接下来依次为256，128节点的全连接层，都使用比例为0.1的 Dropout
        self.fn1 = nn.Linear(18496, 256)
        self.fn2 = nn.Linear(256, 128)

        self.fn_out = nn.Linear(128, 16)

        self.drop = nn.Dropout(p=0.1)

    def make_layer(self, block, planes, size):
        layers = []
        layers.append(block(planes, size))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
        out = out.view(out.shape[0], out.shape[1] * out.shape[2], out.shape[3], out.shape[4])

        # 在二维卷积部分引入注意力机制
        out = self.layer1(out)
        out = self.conv4d_2(out)
        out = self.layer2(out)
        # 接下来是一个 flatten 操作，变为 18496 维的向量
        # 进行重组，以b行，d列的形式存放（d自动计算）
        out = out.view(out.shape[0], -1)

        out = self.fn1(out)
        out = self.drop(out)
        out = self.fn2(out)
        out = self.drop(out)

        out = self.fn_out(out)

        # out = self.soft(out)

        return out


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


def train(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    print(device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
