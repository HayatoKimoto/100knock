from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
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


X_test = np.loadtxt('X_test.txt')
Y_test = np.loadtxt('Y_test.txt')
X_test = torch.tensor(X_test, dtype = torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype = torch.int64)

X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
X_train = torch.tensor(X_train, dtype = torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype = torch.int64)

model = Net().to(device)
model.load_state_dict(torch.load("model.pth"))

train_pred = model(X_train)
test_pred = model(X_test)

train_pred = train_pred.cpu()
test_pred = test_pred.cpu()


print('学習データ:',accuracy(train_pred,Y_train))
print('評価データ:',accuracy(test_pred,Y_test))

"""
[プログラムの結果]
学習データ: 0.8958958020989505
評価データ: 0.8958020989505248
"""