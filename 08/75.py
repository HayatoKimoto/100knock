from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

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

X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype = torch.int64).to(device)

X_valid = np.loadtxt('X_valid.txt')
Y_valid = np.loadtxt('Y_valid.txt')
X_valid = torch.tensor(X_valid, dtype = torch.float32).to(device)
Y_valid = torch.tensor(Y_valid, dtype = torch.int64).to(device)

net = Net().to(device)
ds = TensorDataset(X_train, Y_train)
# DataLoaderを作成
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-1)

fig = plt.figure()
ax= fig.subplots(2)
train_acc_list = []
train_loss_list =[]

valid_acc_list = []
valid_loss_list = []


for epoch in tqdm(range(10)):
    for xx, yy in loader:
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Y_pred = net(X_train)
        loss = loss_fn(Y_pred,Y_train)
        Y_pred = Y_pred.cpu()
        Y_train = Y_train.cpu()
        score = accuracy(Y_pred,Y_train)

        loss_value = loss.cpu()
        train_acc_list.append(score)
        train_loss_list.append(loss_value)


        Y_pred = net(X_valid)
        Y_valid = Y_valid.to(device)
        loss = loss_fn(Y_pred,Y_valid)
        Y_pred = Y_pred.cpu()
        Y_valid = Y_valid.cpu()
        score = accuracy(Y_pred,Y_valid)

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
fig.savefig('75.png')
    