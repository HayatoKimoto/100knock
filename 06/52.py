import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

x_train = pd.read_pickle('ans50/train.feature.pkl')
train_df = pd.read_table('ans50/train.tsv',header = None)[0]

cate2num = {"b": 0, "t": 1, "e": 2, "m": 3}

#文字を数値に対応
y_train = []
for category in train_df:
    y_train.append(cate2num[category])

clf = LogisticRegression(solver='sag',random_state=0)
clf.fit(x_train, y_train)

#学習モデルを保存
pickle.dump(clf,open("model.pkl","wb"))