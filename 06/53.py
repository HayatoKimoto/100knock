import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

title = pd.read_table('ans50/test.tsv', header=None)[1][0]

title_cv = cv.transform([title])

#数値を文字に対応させるための辞書
num2cate = {0:"b", 1: "t", 2: "e",3: "m"}

predict_category = clf.predict(title_cv)[0]

#print(clf.predict(title_cv)) -> [0]
#print(clf.predict_proba(title_cv))  -> [[0.87482419 0.06634552 0.02551963 0.03331066]]

print(clf.predict_proba(title_cv)[0][predict_category],num2cate[predict_category])

#clf.predict_proba(title_cv)[0][predict_category]が予測確立
#num2cate[predict_category]はカテゴリの予測


"""
[プログラムの結果]
% python 53.py  
0.9581742940786622 t
"""