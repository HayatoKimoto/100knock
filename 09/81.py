import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class RNN(nn.Module):
    def __init__(self,vocab_size,emb_size,padding_idx,output_size,hidden_size):
        super(RNN,self).__init__()
        self.emb = nn.Embedding(vocab_size,emb_size,padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size,hidden_size,batch_first =True)
        self.fc = nn.Linear(hidden_size,output_size)
        

    def forward(self,x,h=None):
        x = self.emb(x)
        y,h = self.rnn(x,h)
        y=y[:,-1,:]
        y = self.fc(y)
        y = F.softmax(y,dim=1)
        return y

def remove_mark(sentence):
    specialChars = "!?#$%^&*().\"'" 
    sentence = sentence.replace('.','')
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

def list2tensor(data,padding_id):
  new = []
  for s in data:
    new.append(torch.tensor(s))

  return pad_sequence(new,padding_value=padding,batch_first=True)

#データの呼び出し
train_df = pd.read_table('ans50/train.tsv', header=None)
val_df   = pd.read_table('ans50/valid.tsv', header=None)
test_df  = pd.read_table('ans50/test.tsv', header=None)

#ID番号への変換
vectorizer = CountVectorizer(min_df=2)
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

V=len(d)+1
padding=len(d)
dw = 300
dh = 50
output_size =4


model=RNN(V,dw,padding,4,dh)
X_test = list2tensor(X_test,padding)
y_pred = model(X_test)

print(y_pred)