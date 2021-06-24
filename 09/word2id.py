import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

#id番号を付与するクラス
def remove_mark(sentence):
    specialChars = "!?#$%^&*().\"'" 
    sentence = sentence.replace('.','')
    for specialChar in specialChars:
        sentence = sentence.replace(specialChar, '')
    return sentence
    

def get_id(sentence,d):
  r = []
  for word in sentence:
    r.append(d.get(word,0))
  return r

def df2id(df,d):
  ids = []
  for s in df.iloc[:,1].str.lower():
    s=remove_mark(s)
    ids.append(get_id(s.split(),d))
  return ids

def get_id(path):
    #データの呼び出し
    df = pd.read_table(path, header=None)
    train_df = pd.read_table('ans50/train.tsv', header=None)
    #ID番号への変換
    vectorizer = CountVectorizer(min_df=2)
    train_title =train_df.iloc[:,1].str.lower()
    cnt = vectorizer.fit_transform(train_title).toarray()
    sm = cnt.sum(axis=0)
    idx = np.argsort(sm)[::-1]
    words = np.array(vectorizer.get_feature_names())[idx]
    d = dict()

    for i in range(len(words)):
        d[words[i]] = i+1

    return df2id(df,d)

def get_len():
    #データの呼び出し
    train_df = pd.read_table('ans50/train.tsv', header=None)
    #ID番号への変換
    vectorizer = CountVectorizer(min_df=2)
    df_title =train_df.iloc[:,1].str.lower()
    cnt = vectorizer.fit_transform(df_title).toarray()
    sm = cnt.sum(axis=0)
    idx = np.argsort(sm)[::-1]
    words = np.array(vectorizer.get_feature_names())[idx]
    d = dict()

    for i in range(len(words)):
        d[words[i]] = i+1

    return len(d)
