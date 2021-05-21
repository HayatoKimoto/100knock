from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

x_test  = pd.read_pickle('ans50/test.feature.pkl')
test_df  = pd.read_table('ans50/test.tsv',header = None)[0]

cate2num = {"b": 0, "t": 1, "e": 2, "m": 3}

#正解ラベルの生成
y_test = []
for category in test_df:
    y_test.append(cate2num[category])

#予測した結果
y_test_pred  = clf.predict(x_test) 

print(classification_report(y_test, y_test_pred,target_names=['b','t','e','m']))

print('micro ave')
print(f"precision:{precision_score(y_test, y_test_pred, average= 'micro')}")
print(f"recall:{recall_score(y_test, y_test_pred, average= 'micro')}")
print(f"f1_score:{f1_score(y_test, y_test_pred, average= 'micro')}")


"""
[プログラムの結果]
%python 56.py
              precision    recall  f1-score   support

           b       0.90      0.93      0.92       567
           t       0.86      0.68      0.76       155
           e       0.89      0.97      0.93       521
           m       0.90      0.58      0.71        91

    accuracy                           0.89      1334
   macro avg       0.89      0.79      0.83      1334
weighted avg       0.89      0.89      0.89      1334

micro ave
precision:0.8943028485757122
recall:0.8943028485757122
f1_score:0.8943028485757122

"""