import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from matplotlib import pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


#データの読み込み
X_train = pd.read_table('ans50/train.tsv', header=None)[1].tolist()
X_valid = pd.read_table('ans50/valid.tsv', header=None)[1].tolist()
Y_train = np.loadtxt('ans50/Y_train.txt')
Y_valid = np.loadtxt('ans50/Y_valid.txt')


X_train = tokenizer(X_train,padding="max_length",return_tensors='pt',truncation=True)['input_ids']

Y_train = torch.tensor(Y_train, dtype = torch.int64)

X_valid = tokenizer(X_valid,padding="max_length",return_tensors='pt',truncation=True)['input_ids']
Y_valid = torch.tensor(Y_valid, dtype = torch.int64)

# モデルのロード
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4).to(device)


for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

for name, param in model.classifier.named_parameters():
    param.requires_grad = True


dataset = TensorDataset(X_train, Y_train)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(10)):
  model.train()
  for xx, yy in loader:
    xx = xx.to(device)
    yy = yy.to(device)
    #outputs=[loss,logits,hidden_states,attentions]
    outputs = model(xx,labels=yy)
    loss, logits = outputs[:2]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(net.state_dict(), 'model.pth')

model.eval()
with torch.no_grad(): 
    outputs = model(X_valid.to(device),labels=Y_valid.to(device))
    loss, logits = outputs[:2]
    Y_pred = logits.cpu()
    Y_valid = Y_valid.cpu()
    score = accuracy(Y_pred,Y_valid)
    print(score)
        
        
