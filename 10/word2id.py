import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import pickle

#id番号を付与するクラス
def get_dict_ja():
    d=dict()
    counter=Counter()
    with open('kftt-data-1.0/data/tok/kyoto-train.cln.ja') as f_j:
        for line in f_j:
            line=line.split(' ')
            counter.update(line)
        
    d['<s>']  =1
    d['</s>'] =2
    d['[PAD]']=0

    for i,word in enumerate(counter.most_common()):
        d[word[0]]=i+4
        if word[1]<2:
            d[word[0]] = 3

    with open('word2id_ja','wb')as f:
        pickle.dump(d,f)

def get_dict_en():
    d=dict()
    counter=Counter()
    with open('kftt-data-1.0/data/tok/kyoto-train.cln.en') as f_j:
        for line in f_j:
            line=line.split(' ')
            counter.update(line)
        
    d['<s>']  =1
    d['</s>'] =2
    d['[PAD]']=0

    for i,word in enumerate(counter.most_common()):
        d[word[0]]=i+4
        if word[1]<2:
            d[word[0]] = 3

    with open('word2id_en','wb')as f:
        pickle.dump(d,f)

    
def get_ID(sentence,d):
  r = []
  sentence.append('</s>')
  sentence.insert(0,'<s>')
  for word in sentence:
    r.append(d.get(word,0))
  return r

def df2id(lines,d):
  ids = []
  for s in lines:
    ids.append(get_ID(s.split(),d))
  return ids

def get_id_ja(path):
    with open('word2id_ja','rb') as f:
        d=pickle.load(f)

    with open(path) as f:
        lines=f.readlines()
    
    return df2id(lines,d)

def get_id_en(path):
    with open('word2id_en','rb') as f:
        d=pickle.load(f)

    with open(path) as f:
        lines=f.readlines()
    
    return df2id(lines,d)


def get_vocab_ja():
    with open('word2id_ja','rb') as f:
        d=pickle.load(f)

    v_size=len(d)

    
    return v_size

def get_vocab_en():
    with open('word2id_en','rb') as f:
        d=pickle.load(f)

    v_size=len(d)
   

    return v_size

def get_id2word_dict_en():
    with open('word2id_en','rb') as f:
        d=pickle.load(f)
    
    id2word_dict=dict()
    for word,id in d.items():
        id2word_dict[id]=word

    with open('id2word_en','wb')as f:
        pickle.dump(id2word_dict,f)

def get_id2word_dict_ja():
    with open('word2id_ja','rb') as f:
        d=pickle.load(f)
    
    id2word_dict=dict()
    for word,id in d.items():
        id2word_dict[id]=word

    with open('id2word_ja','wb')as f:
        pickle.dump(id2word_dict,f)

    
"""

if __name__ == "__main__":
    get_id2word_dict_en()
    get_id2word_dict_ja()
"""