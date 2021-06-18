import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence

class RNN(nn.Module):
    def __init__(self,vocab_size,emb_size,padding_idx,output_size,hidden_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size,emb_size,padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size,hidden_size,batch_first =True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,h=None):
        x = self.emb(x)
        y,h = self.rnn(x,h)
        y=y[:,-1,:]
        y = self.fc(y)
        #y = F.softmax(y,dim=1)
        return y

def remove_mark(sentence):
    specialChars = "!?#$%^&*().\"'" 
    for specialChar in specialChars:
        sentence = sentence.replace(specialChar, '')
    return sentence
    
def get_id(sentence):
  r = []
  for word in sentence:
    r.append(d.get(word,0))
  return r

def df2id(df):
  ids = []
  for s in df.iloc[:,1].str.lower():
    s=remove_mark(s)
    ids.append(get_id(s.split()))
  return ids

def accuracy(pred, label):
  pred = np.argmax(pred.data.numpy(), axis=1)
  label = label.data.numpy()
  return (pred == label).mean()

def list2tensor(data,padding_id):
  new = []
  for s in data:
    new.append(torch.tensor(s))

  return pad_sequence(new,padding_value=padding,batch_first=True)

#GPUが使用可能か
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

#データの呼び出し
train_df = pd.read_table('ans50/train.tsv', header=None)
val_df   = pd.read_table('ans50/valid.tsv', header=None)
test_df  = pd.read_table('ans50/test.tsv', header=None)


vectorizer = CountVectorizer(min_df=2)
#大文字の単語を小文字にして抽出
train_title = train_df.iloc[:,1].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]
d = dict()
for i in range(len(words)):
  d[words[i]] = i+1

X_train=df2id(train_df)
X_valid=df2id(val_df)
X_test=df2id(test_df)
Y_train = np.loadtxt('ans50/Y_train.txt')
Y_valid = np.loadtxt('ans50/Y_valid.txt')

V=len(d)+1
padding=len(d)
dw = 300
dh = 50
output_size =4

model=RNN(V,dw,padding,output_size,dh).to(device)

X_train = list2tensor(X_train,padding)
Y_train = torch.tensor(Y_train, dtype = torch.int64)

X_valid = list2tensor(X_valid,padding).to(device)
Y_valid = torch.tensor(Y_valid, dtype = torch.int64)

ds = TensorDataset(X_train.to(device), Y_train.to(device))
loader = DataLoader(ds, batch_size=1, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

fig = plt.figure()
ax= fig.subplots(2)
train_acc_list = []
train_loss_list =[]

valid_acc_list = []
valid_loss_list = []

for epoch in tqdm(range(10)):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        Y_pred = model(X_train.to(device))
        loss = loss_fn(Y_pred,Y_train.to(device))
        
        Y_pred = Y_pred.cpu()
        Y_train = Y_train.cpu()
        score = accuracy(Y_pred,Y_train)

        loss_value = loss.cpu()
        train_acc_list.append(score)
        train_loss_list.append(loss_value)


        Y_pred = model(X_valid.to(device))
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
fig.savefig('82.png')