import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


x_train = pd.read_pickle('ans50/train.feature.pkl')
x_test  = pd.read_pickle('ans50/test.feature.pkl')
x_val  = pd.read_pickle('ans50/valid.feature.pkl')
train_df = pd.read_table('ans50/train.tsv',header = None)[0]
test_df  = pd.read_table('ans50/test.tsv',header = None)[0]
val_df  = pd.read_table('ans50/valid.tsv',header = None)[0]

y_train = []
cate2num = {"b": 0, "t": 1, "e": 2, "m": 3}
for category in train_df:
    y_train.append(cate2num[category])

y_test = []
for category in test_df:
    y_test.append(cate2num[category])

y_val = []
for category in val_df:
    y_val.append(cate2num[category])
    
#学習アルゴリズム
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#学習パラメータ
C = np.logspace(-10, 10, 21)
max_score = 0
for c in C:
  for s in solver:
    clf = LogisticRegression(solver=s,C=c)
    clf.fit(x_train, y_train)
    #それぞれの予測値  
    y_val_pred   = clf.predict(x_val)
    #正解率の計算
    val_acc = accuracy_score(y_val,y_val_pred)
    #resultに格納
    if max_score < val_acc:
        best_clf = clf
        max_score = val_acc
        best_params = {"solver":s ,"C":c}

print(best_params)
y_test_pred   = best_clf.predict(x_test)
test_acc = accuracy_score(y_test,y_test_pred)
print(test_acc)