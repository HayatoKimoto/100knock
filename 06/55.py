from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


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


print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test,y_test_pred))

"""
%python 55.py
[[4408   40   36    8]
 [  90 1089   50    2]
 [  27    9 4179    0]
 [  42    4   42  646]]
[[530   9  23   5]
 [ 32 106  17   0]
 [ 14   2 504   1]
 [ 10   6  22  53]]
"""