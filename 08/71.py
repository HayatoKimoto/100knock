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
X_train = torch.tensor(X_train, dtype = torch.float32).to(device)

net = Net().to(device)
y_1 =net(X_train[:1])
Y = net(X_train[:4])

print(y_1)
print(Y)

"""
[プログラムの結果]
%python 71.py
tensor([[0.2475, 0.2632, 0.2377, 0.2516]], device='cuda:0',
       grad_fn=<SoftmaxBackward>)
tensor([[0.2475, 0.2632, 0.2377, 0.2516],
        [0.2425, 0.2549, 0.2518, 0.2508],
        [0.2519, 0.2597, 0.2413, 0.2471],
        [0.2475, 0.2538, 0.2335, 0.2652]], device='cuda:0',
       grad_fn=<SoftmaxBackward>)
"""

