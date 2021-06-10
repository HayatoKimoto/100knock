from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(300,4,bias = False)

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()



X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32)
Y_train = torch.tensor(Y_train, dtype = torch.int64)


net = Net()
ds = TensorDataset(X_train, Y_train)
loss_fn = nn.CrossEntropyLoss()
# DataLoaderを作成

bs_list = [2**i for i in range(10)]
time_list = []

for bs in tqdm(bs_list):
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    optimizer = optim.SGD(net.parameters(), lr=1e-1)
    for epoch in range(1):
        start_time = time()
        for xx, yy in loader:
            xx = xx
            yy = yy
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    time_list.append(time()-start_time)

print(time_list)

"""
[プログラムの結果]
%python 78.py
[3.1344478130340576, 1.8021328449249268, 0.9995532035827637, 0.4774291515350342, 0.2679271697998047, 0.1744701862335205, 0.12577176094055176, 0.2232370376586914, 0.13744044303894043, 0.09383249282836914]
"""


    
        


    
torch.save(net, 'model.pth')