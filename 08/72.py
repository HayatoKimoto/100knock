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

X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32)
Y_train = torch.tensor(Y_train, dtype = torch.int64)
loss_fn = nn.CrossEntropyLoss()

net = Net()
y_1 =net(X_train[:1])
Y = net(X_train[:4])

print(loss_fn(y_1,Y_train[:1]))
print(loss_fn(Y,Y_train[:4]))

"""
[プログラムの結果]
tensor(1.3960, grad_fn=<NllLossBackward>)
tensor(1.3943, grad_fn=<NllLossBackward>)
"""



