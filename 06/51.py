import joblib
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#データの読み込み
train_df = pd.read_table('ans50/train.tsv', header=None)
val_df = pd.read_table('ans50/valid.tsv', header=None)
test_df = pd.read_table('ans50/test.tsv', header=None)

#
use_cols = ['CATEGORY','TITLE']
train_df.columns = use_cols
val_df.columns   = use_cols
test_df.columns  = use_cols


#CountVectorizerクラスを利用した特徴抽出
cv = CountVectorizer()
train_cv = cv.fit_transform(train_df['TITLE'])
test_cv  = cv.transform(test_df['TITLE'])
val_cv   = cv.transform(val_df['TITLE'])

#pickleはバイト列に変換してファイルに保存する
pickle.dump(cv,open("cv.pkl","wb"))

#データフレームに変換
x_train = pd.DataFrame(train_cv.toarray(), columns=cv.get_feature_names())
x_test  = pd.DataFrame(test_cv.toarray(), columns=cv.get_feature_names())
x_val   = pd.DataFrame(val_cv.toarray(), columns=cv.get_feature_names())

#保存
x_train.to_pickle('ans50/train.feature.pkl')
x_val.to_pickle('ans50/valid.feature.pkl')
x_test.to_pickle('ans50/test.feature.pkl')

