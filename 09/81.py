import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,vocab_size,emb_size,padding_idx,output_size,hidden_size):
        super(RNN,self).__init__()
        self.emb = nn.Embedding(vocab_size,emb_size,padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size,hidden_size,batch_first =True)
        self.fc = nn.Linear(hidden_size,output_size)
        

    def forward(self,x,h=None):
        x = self.emb(x)
        y,h = self.rnn(x,h)
        y = y[:,-1,:] # 最後のステップ
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

def list2tensor(data, max_len):
  new = []
  for d in data:
    if len(d) > max_len:
      d = d[:max_len]
    else:
      d += [len(d)+1] * (max_len - len(d))
    new.append(d)
  return torch.tensor(new, dtype=torch.int64)

#データの呼び出し
train_df = pd.read_table('ans50/train.tsv', header=None)
val_df   = pd.read_table('ans50/valid.tsv', header=None)
test_df  = pd.read_table('ans50/test.tsv', header=None)

vectorizer = CountVectorizer(min_df=2)
#大文字の単語を小文字にして抽出
train_title = train_df.iloc[:,1].str.lower()
cnt = vectorizer.fit_transform(train_title).toarray()
sm = cnt.sum(axis=0)
#print(sm)
idx = np.argsort(sm)[::-1]
words = np.array(vectorizer.get_feature_names())[idx]
d = dict()
for i in range(len(words)):
  d[words[i]] = i+1

X_train=df2id(train_df)
X_valid=df2id(val_df)
X_test=df2id(test_df)

V=len(d)+2
padding=len(d)+1
dw = 300
dh = 50
output_size =4


model=RNN(V,dw,padding,4,dh)
X_train = list2tensor(X_train,10)
y_pred = model(X_train)

print(y_pred)