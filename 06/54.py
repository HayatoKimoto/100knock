import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

x_train = pd.read_pickle('ans50/train.feature.pkl')
x_test  = pd.read_pickle('ans50/test.feature.pkl')
train_df = pd.read_table('ans50/train.tsv',header = None)[0]
test_df  = pd.read_table('ans50/test.tsv',header = None)[0]

#正解ラベルの生成
cate2num = {"b": 0, "t": 1, "e": 2, "m": 3}
y_train = []
for category in train_df:
    y_train.append(cate2num[category])

y_test = []
for category in test_df:
    y_test.append(cate2num[category])

#予測した結果
y_train_pred = clf.predict(x_train) 
y_test_pred  = clf.predict(x_test) 

#学習データと評価データの正解率
print(accuracy_score(y_train,y_train_pred))
print(accuracy_score(y_test,y_test_pred))

"""
[プログラムの結果]
%python 54.py
0.9672038980509745
0.8943028485757122
"""
