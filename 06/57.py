import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

x_train = pd.read_pickle('ans50/train.feature.pkl')
clf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

num2cate = {0:"b", 1: "t", 2: "e",3: "m"}
names = np.array(cv.get_feature_names())

#インスタンス名.coef_とすることで、パラメータ（重み）を取得することができる
for c,coef in zip(clf.classes_,clf.coef_):
    print(num2cate[c])
    idx = np.argsort(coef)[::-1]
    print('重みの高い特徴量トップ10:',names[idx][:10])
    print('重みの低い特徴量トップ10:',names[idx][-10:][::-1])

"""
[プログラムの結果]
b
重みの高い特徴量トップ10: ['bank' 'fed' 'ecb' 'euro' 'oil' 'ukraine' 'china' 'stocks' 'yellen'
 'dollar']
重みの低い特徴量トップ10: ['ebola' 'she' 'video' 'aereo' 'her' 'virus' 'star' 'kardashian' 'drug'
 'fda']
t
重みの高い特徴量トップ10: ['google' 'facebook' 'apple' 'microsoft' 'climate' 'nasa' 'tesla' 'fcc'
 'gm' 'heartbleed']
重みの低い特徴量トップ10: ['stocks' 'american' 'her' 'percent' 'fed' 'drug' 'cancer' 'still' 'fda'
 'ecb']
e
重みの高い特徴量トップ10: ['kardashian' 'chris' 'movie' 'miley' 'film' 'she' 'paul' 'cyrus' 'jay'
 'thrones']
重みの低い特徴量トップ10: ['google' 'study' 'billion' 'china' 'gm' 'ceo' 'facebook' 'risk'
 'microsoft' 'apple']
m
重みの高い特徴量トップ10: ['ebola' 'cancer' 'fda' 'drug' 'study' 'mers' 'health' 'cases' 'cdc'
 'heart']
重みの低い特徴量トップ10: ['gm' 'facebook' 'apple' 'bank' 'google' 'ceo' 'buy' 'twitter' 'deal'
 'climate']

 """