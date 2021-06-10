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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device('cpu')

X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32)
Y_train = torch.tensor(Y_train, dtype = torch.int64)

net = Net().to(device)
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
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    time_list.append(time()-start_time)

print(time_list)

"""
[プログラムの結果]
%python 77.py
[6.518698215484619, 3.2509524822235107, 1.7100205421447754, 0.7222609519958496, 0.5127522945404053, 0.29308319091796875, 0.17279505729675293, 0.12406730651855469, 0.10103416442871094, 0.09032535552978516]
"""


    
        


    
torch.save(net, 'model.pth')