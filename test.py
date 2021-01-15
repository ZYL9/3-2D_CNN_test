import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数
pca_components = 30
patch_size = 25
path = './model.pth'
lr = 0.001

# 读数据
X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
# 网络
net = HybridSN().to(device)  # 网络放到GPU上
optimizer = optim.Adam(net.parameters(), lr=lr)
checkpoint = torch.load(path)
net.load_state_dict(checkpoint['net'])

height = y.shape[0]
width = y.shape[1]

X = applyPCA(X, pca_components)
X = padWithZeros(X, patch_size // 2)

outputs = np.zeros((height, width))

# 逐像素
for i in range(height):
    for j in range(width):
        if int(y[i, j]) == 0:
            continue
        else:
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            prediction = net(X_test_image)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
    print('... ... row ', i, ' handling ... ...')

np.savetxt("new.csv", outputs, delimiter=',')
predict_image = spectral.imshow(classes=y.astype(int),figsize=(10,10))
predict_image = spectral.imshow(classes=outputs.astype(int),figsize=(10,10))
plt.show()
