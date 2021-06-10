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
        self.fc = nn.Linear(300,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,4)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)

        return x

        
def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype = torch.int64).to(device)

X_valid = np.loadtxt('X_valid.txt')
Y_valid = np.loadtxt('Y_valid.txt')
X_valid = torch.tensor(X_valid, dtype = torch.float32).to(device)
Y_valid = torch.tensor(Y_valid, dtype = torch.int64).to(device)

fig = plt.figure()
ax= fig.subplots(2)
train_acc_list = []
train_loss_list =[]

valid_acc_list = []
valid_loss_list = []


net = Net().to(device)
ds = TensorDataset(X_train, Y_train)
loss_fn = nn.CrossEntropyLoss()
# DataLoaderを作成

loader = DataLoader(ds, batch_size=1, shuffle=True)
optimizer = optim.SGD(net.parameters(), lr=1e-1)


for epoch in tqdm(range(10)):
    for xx, yy in loader:
        xx = xx.to(device)
        yy = yy.to(device)
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = net(X_train)
        loss = loss_fn(y_pred,Y_train)
        y_pred = y_pred.cpu()
        y_train = Y_train.cpu()
        score=accuracy(y_pred,y_train)

        loss_value = loss.cpu()
        train_acc_list.append(score)
        train_loss_list.append(loss_value)
        

        y_pred = net(X_valid)
        Y_valid = Y_valid.to(device)
        loss = loss_fn(y_pred,Y_valid)
        y_pred = y_pred.cpu()
        y_valid = Y_valid.cpu()
        score = accuracy(y_pred,y_valid)

        loss_value = loss.cpu()
        valid_acc_list.append(score)
        valid_loss_list.append(loss_value)

fig = plt.figure()
ax= fig.subplots(2)
ax[0].plot(train_loss_list,label='train')
ax[1].plot(train_acc_list,label='train')
ax[0].plot(valid_loss_list,label='valid')
ax[1].plot(valid_acc_list,label='valid')

ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')

ax[0].legend()
ax[1].legend()
fig.savefig('79.png')


train_pred = net(X_train).to(device)
valid_pred = net(X_valid).to(device)

train_pred = train_pred.cpu()
valid_pred = valid_pred.cpu()
Y_train = Y_train.cpu()
Y_valid = Y_valid.cpu()
print('学習データ:',accuracy(train_pred,Y_train))
print('評価データ:',accuracy(valid_pred,Y_valid))



"""
[プログラムの結果]
%python 79.py
学習データ: 0.8779047976011994
評価データ: 0.8643178410794603
"""