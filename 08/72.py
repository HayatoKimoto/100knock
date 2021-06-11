import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(300,4,bias = False)

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype = torch.int64).to(device)
loss_fn = nn.CrossEntropyLoss()

net = Net().to(device)

y_1 =net(X_train[:1])
loss = loss_fn(y_1,Y_train[:1])
loss.backward()
print(loss)
print(net.fc.weight.grad)

net.zero_grad()
Y = net(X_train[:4])
loss = loss_fn(Y,Y_train[:4])
loss.backward()
print(loss)
print(net.fc.weight.grad)


"""
[プログラムの結果]
%python 72.py
tensor(1.3896, device='cuda:0', grad_fn=<NllLossBackward>)
tensor([[ 0.0098, -0.0029,  0.0017,  ..., -0.0030,  0.0068,  0.0017],
        [ 0.0105, -0.0031,  0.0019,  ..., -0.0032,  0.0073,  0.0018],
        [-0.0302,  0.0090, -0.0053,  ...,  0.0092, -0.0209, -0.0051],
        [ 0.0098, -0.0029,  0.0017,  ..., -0.0030,  0.0068,  0.0017]],
       device='cuda:0')
tensor(1.3863, device='cuda:0', grad_fn=<NllLossBackward>)
tensor([[ 0.0119, -0.0052,  0.0039,  ..., -0.0072,  0.0108,  0.0010],
        [ 0.0129, -0.0050,  0.0039,  ..., -0.0072,  0.0110,  0.0004],
        [-0.0365,  0.0155, -0.0118,  ...,  0.0216, -0.0328, -0.0026],
        [ 0.0117, -0.0053,  0.0040,  ..., -0.0072,  0.0109,  0.0012]],
       device='cuda:0')
"""



