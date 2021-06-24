import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from word2id import get_id
from word2id import get_len

class RNN(nn.Module):
    def __init__(self,vocab_size,emb_size,output_size,hidden_size):
        super(RNN,self).__init__()
        self.emb = nn.Embedding(vocab_size,emb_size)
        self.rnn = nn.RNN(emb_size,hidden_size,batch_first =True)
        self.fc = nn.Linear(hidden_size,output_size,bias=True)

    def forward(self,padded_packed_input):
        x ,len_list = padded_packed_input
        x = self.emb(x)
        x = pack_padded_sequence(x, len_list, batch_first=True, enforce_sorted=False)
        x,h = self.rnn(x)
        y = self.fc(h)
        y = y.squeeze(0)
        y = F.softmax(y,dim=1)
        return y

def list2tensor(data):
  new = []
  for s in data:
    new.append(torch.tensor(s))
  
  packed_inputs= pack_sequence(new,enforce_sorted=False)
  padded_packed_inputs = pad_packed_sequence(packed_inputs, batch_first=True)
  return padded_packed_inputs



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


model=RNN(V,dw,4,dh)
X_test= list2tensor(X_test)
y_pred = model(X_test)

print(y_pred)

"""
tensor([[0.2116, 0.2378, 0.3269, 0.2238],
        [0.1469, 0.3318, 0.2608, 0.2605],
        [0.1510, 0.3017, 0.3326, 0.2146],
        ...,
        [0.2803, 0.1648, 0.2736, 0.2813],
        [0.2022, 0.2300, 0.2815, 0.2863],
        [0.1738, 0.5530, 0.1434, 0.1297]], grad_fn=<SoftmaxBackward>)
"""