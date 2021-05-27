import gensim
import pickle
import pandas as pd
import scipy.stats

df = pd.read_csv('wordsim353/combined.csv')
model = pickle.load(open("kv.pkl","rb"))

sim = []
for i in range(len(df)):
    line = df.iloc[i]
    sim.append(model.similarity(line['Word 1'],line['Word 2']))

df['w2v'] = sim 
print(scipy.stats.spearmanr(df['Human (mean)'], df['w2v']))

"""
[プログラムの結果]
%python 66.py
SpearmanrResult(correlation=0.7000166486272194, pvalue=2.86866666051422e-53)
"""
