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
Y = net(X_train[:4])

print(loss_fn(y_1,Y_train[:1]))
print(loss_fn(Y,Y_train[:4]))

"""
[プログラムの結果]
%python 72.py
tensor(1.3981, device='cuda:0', grad_fn=<NllLossBackward>)
tensor(1.4020, device='cuda:0', grad_fn=<NllLossBackward>)
"""



