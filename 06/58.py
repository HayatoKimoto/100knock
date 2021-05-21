import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

result = []
C = np.logspace(-10, 10, 21)
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
for c in tqdm(C):
  #モデルの学習

  clf = LogisticRegression(solver='liblinear',C=c)
  clf.fit(x_train, y_train)
  #それぞれの予測値
  y_train_pred = clf.predict(x_train) 
  y_test_pred  = clf.predict(x_test)
  y_val_pred   = clf.predict(x_val)
  #正解率の計算
  train_acc = accuracy_score(y_train,y_train_pred)
  val_acc = accuracy_score(y_val,y_val_pred)
  test_acc = accuracy_score(y_test,y_test_pred)
  #resultに格納
  result.append([c, train_acc, val_acc, test_acc])
result = np.array(result).T

plt.plot(result[0], result[1], label='train')
plt.plot(result[0], result[2], label='val')
plt.plot(result[0], result[3], label='test')
#plt.ylim(0.5, 1.1)
plt.ylabel('Accuracy')
plt.xscale('log')
plt.xlabel('C')
plt.legend()
plt.show()