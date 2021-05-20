import joblib
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
#CountVectorizerクラスを利用したBoW形式の特徴抽出(train_valのみ)
from sklearn.feature_extraction.text import CountVectorizer
#データの読み込み
train_df = pd.read_csv('ans50/train.tsv', sep='\t', header=None)
val_df = pd.read_csv('ans50/valid.tsv', sep='\t', header=None)
test_df = pd.read_csv('ans50/test.tsv', sep='\t', header=None)
#前処理を一括で行うためのデータの統合

use_cols = ['CATEGORY','TITLE']

train_df.columns = use_cols
val_df.columns   = use_cols
test_df.columns  = use_cols

data = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

cv = CountVectorizer()
train_cv = cv.fit_transform(data['TITLE'])
test_cv  = cv.transform(test_df['TITLE'])
val_cv   = cv.transform(val_df['TITLE'])

#データフレームに変換
x_train = pd.DataFrame(train_cv.toarray(), columns=cv.get_feature_names())
x_test  = pd.DataFrame(test_cv.toarray(), columns=cv.get_feature_names())
x_val   = pd.DataFrame(val_cv.toarray(), columns=cv.get_feature_names())

x_train = x_train[len(x_test)+len(x_val):]
#保存
x_train.to_csv('ans50/train.feature.tsv', sep='\t',header=False,index=False)
x_val.to_csv('ans50/valid.feature.tsv', sep='\t',header=False, index=False)
x_test.to_csv('ans50/test.feature.tsv', sep='\t',header=False, index=False)

