import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))
title = pd.read_csv('ans50/test.tsv', sep='\t', header=None)[1][0]
title_cv = cv.transform([title])
num2cate = {0:"b", 1: "t", 2: "e",3: "m"}

predict_category=clf.predict(title_cv)[0]

print(clf.predict_proba(title_cv)[0][predict_category],num2cate[predict_category])

