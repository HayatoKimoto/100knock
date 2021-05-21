#p51 コードを段落に分ける
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split

with zipfile.ZipFile('NewsAggregatorDataset.zip') as zf:
  with zf.open("newsCorpora.csv") as myfile:
     newsCorpora = pd.read_table(myfile, header=None)

newsCorpora.columns = ['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP']


#条件2つ目：指定の情報源で絞り込み
df = newsCorpora[newsCorpora['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
#条件3つ目：ランダムに並べ替え
df = df.sample(frac=1)
"""flac=1で100%の指定"""
#条件4つ目：データの切り分け
train, val_test = train_test_split(df, test_size=0.2)
val, test = train_test_split(val_test, test_size=0.5)

"""
train_test_split()関数でリストを分割
test_sizeは2つめのリストの大きさの割合を指定
train_sizeは1つめのリストの大きさの割合を指定
個数でも割合でも指定可能
"""

#保存
train.to_csv('ans50/train.tsv', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
val.to_csv('ans50/valid.tsv', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test.to_csv('ans50/test.tsv', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)

print("学習データ")
print(train['CATEGORY'].value_counts())

print("検証データ")
print(val['CATEGORY'].value_counts())

print("評価データ")
print(test['CATEGORY'].value_counts())



"""
[プログラムの結果]
%python 50.p
学習データ
b    4462
e    4242
t    1242
m     726
Name: CATEGORY, dtype: int64
検証データ
b    571
e    525
t    146
m     92
Name: CATEGORY, dtype: int64
評価データ
b    594
e    512
t    136
m     92
Name: CATEGORY, dtype: int64
"""