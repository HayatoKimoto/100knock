import optuna
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
from word2id import add_id
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

class RNN(nn.Module):
    def __init__(self,vocab_size,emb_size,output_size,hidden_size,n_layer):
        super(RNN,self).__init__()
        self.emb = nn.Embedding(vocab_size,emb_size)
        self.rnn = nn.RNN(emb_size,hidden_size,bidirectional=True,batch_first =True,num_layers=n_layer)
        self.fc = nn.Linear(2*hidden_size,output_size,bias=True)

    def forward(self,padded_packed_input):
        x ,len_list = padded_packed_input
        x = self.emb(x)
        x = pack_padded_sequence(x, len_list, batch_first=True, enforce_sorted=False)
        x,h = self.rnn(x)
        h = torch.cat([h[-1],h[-2]],dim=1)
        y = self.fc(h)
        # = F.softmax(y,dim=1)
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

def objective(trial):
    # Int parameter
    num_layers = trial.suggest_int('num_layers', 1, 3)

    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Discrete-uniform parameter
    batchSize = int(trial.suggest_discrete_uniform('batchSize', 16, 128, 16))

    #データの読み込み
    X_train = add_id('ans50/train.tsv')
    X_valid = add_id('ans50/valid.tsv')
    X_test  = add_id('ans50/test.tsv')
    Y_train = np.loadtxt('ans50/Y_train.txt')
    Y_valid = np.loadtxt('ans50/Y_valid.txt')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #固定パラメータの設定
    V=get_len() + 1
    dw = 300
    dh = 50
    output_size = 4

    model=RNN(V,dw,4,dh,num_layers).to(device)
    X_train = list2tensor(X_train)
    Y_train = torch.tensor(Y_train, dtype = torch.int64)

    X_valid = list2tensor(X_valid)
    Y_valid = torch.tensor(Y_valid, dtype = torch.int64)


    dataset = MyDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    valid_data=(X_valid[0].to(device),X_valid[1].to(device))

    for epoch in tqdm(range(100)):
        model.train()
        for xx, yy in loader:
            xx=(xx[0].to(device),xx[1].to(device))
            yy = yy.to(device)
            y_pred = model(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    model.eval()
    with torch.no_grad():
        Y_pred = model(valid_data)
        Y_valid = Y_valid.to(device)
        loss = loss_fn(Y_pred,Y_valid)

    return loss

study = optuna.create_study()
study.optimize(objective, timeout=7200)
