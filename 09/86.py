import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from matplotlib import pyplot as plt
from tqdm import tqdm
from word2id import get_id
from word2id import get_len

class MyDataset(Dataset):
  def __init__(self, padded_packed_data, target):
    self.padded_data, self.len_list = padded_packed_data
    self.target = target
    self.len = len(target)
  
  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    packed_data = (self.padded_data[index], self.len_list[index])
    label = self.target[index]
    return packed_data, label

class CNN(nn.Module):
  def __init__(self,vocab_size,emb_size,output_size,hidden_size,filter_size):
    super(CNN,self).__init__()

    self.emb =nn.Embedding(vocab_size,emb_size)
    # kernel_sizeが(3, emb_size)なのでパッディングを(1, 0)にして畳み込み後のサイズが変わらないようにする
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size,kernel_size=(filter_size, emb_size),stride=1,padding=(1,0))
    self.relu = nn.ReLU()
    self.linear = nn.Linear(hidden_size,output_size,bias=True)
      

  def forward(self,padded_packed_input):
    x ,len_list = padded_packed_input
    x = self.emb(x).unsqueeze(1)
    x = self.conv1(x)
    x=x.squeeze(-1)
    x = self.relu(x)
    x = F.max_pool1d(x,x.size(2))
    x = x.squeeze(2)
    y = self.linear(x)
    y = F.softmax(y,dim=1)
    return y

def list2tensor(data):
  new = []
  for s in data:
    new.append(torch.tensor(s))
  
  packed_inputs= pack_sequence(new,enforce_sorted=False)
  padded_packed_inputs = pad_packed_sequence(packed_inputs, batch_first=True)
  return padded_packed_inputs

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


#データの読み込み
X_train = get_id('ans50/train.tsv')
X_valid = get_id('ans50/valid.tsv')
X_test  = get_id('ans50/test.tsv')
Y_train = np.loadtxt('ans50/Y_train.txt')
Y_valid = np.loadtxt('ans50/Y_valid.txt')

#パラメータの設定
V=get_len()+1
dw = 300
dh = 50
output_size =4
filter_size=3

model=CNN(V,dw,output_size,dh,filter_size)
X_test= list2tensor(X_test)
y_pred = model(X_test)

print(y_pred)
"""
tensor([[0.4660, 0.1788, 0.2841, 0.0711],
        [0.4902, 0.1657, 0.2635, 0.0806],
        [0.4138, 0.2305, 0.2709, 0.0848],
        ...,
        [0.3595, 0.1799, 0.3613, 0.0993],
        [0.4729, 0.1870, 0.2550, 0.0852],
        [0.4113, 0.1558, 0.3633, 0.0696]], grad_fn=<SoftmaxBackward>)
"""