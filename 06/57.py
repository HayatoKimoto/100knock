import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

x_train = pd.read_pickle('ans50/train.feature.pkl')
clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

num2cate = {0:"b", 1: "t", 2: "e",3: "m"}
names = np.array(cv.get_feature_names())
#インスタンス名.coef_とすることで、パラメータ（重み）を取得することができます
for c,coef in zip(clf.classes_,clf.coef_):
    print(num2cate[c])
    idx = np.argsort(coef)[::-1]
    print('重みの高い特徴量トップ10:',names[idx][:10])
    print('重みの低い特徴量トップ10:',names[idx][-10:][::-1])