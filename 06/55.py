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

cate2num = {"b": 0, "t": 1, "e": 2, "m": 3}
y_train = []
for category in train_df:
    y_train.append(cate2num[category])

y_test = []
for category in test_df:
    y_test.append(cate2num[category])

x_train_pred = clf.predict(x_train) 
x_test_pred  = clf.predict(x_test) 

print(confusion_matrix(y_train,x_train_pred))
print(confusion_matrix(y_test,x_test_pred))
