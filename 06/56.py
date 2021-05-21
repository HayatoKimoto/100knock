from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

x_test  = pd.read_pickle('ans50/test.feature.pkl')
test_df  = pd.read_table('ans50/test.tsv',header = None)[0]

cate2num = {"b": 0, "t": 1, "e": 2, "m": 3}

y_test = []
for category in test_df:
    y_test.append(cate2num[category])

x_test_pred  = clf.predict(x_test) 

#precision_score,recall_score,f1_scoreを用いた。
#引数averageとしてNoneを指定した場合はカテゴリ毎にリストで返却
#micro,macroを指定した場合はそれぞれマイクロ平均とマクロ平均が返却

print(classification_report(y_test, x_test_pred,target_names=['b','t','e','m']))

print('micro ave')
print(f"precision:{precision_score(y_test, x_test_pred, average= 'micro')}")
print(f"recall:{recall_score(y_test, x_test_pred, average= 'micro')}")
print(f"f1_score:{f1_score(y_test, x_test_pred, average= 'micro')}")

