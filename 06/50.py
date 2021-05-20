import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split

with zipfile.ZipFile('NewsAggregatorDataset.zip') as zf:
  with zf.open("newsCorpora.csv") as myfile:
     newsCorpora = pd.read_csv(myfile, sep='\t', header=None)

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
test_sizeは2つめのリストの大きさを指定
train_sizeは1つめのリストの大きさを指定
個数でも割合でも指定可能
"""
#保存
train.to_csv('ans50/train.tsv', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
val.to_csv('ans50/valid.tsv', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test.to_csv('ans50/test.tsv', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)

print(len(train[['CATEGORY','TITLE']]),len(test[['CATEGORY','TITLE']]))