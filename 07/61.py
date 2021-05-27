import gensim
import pickle

model = pickle.load(open("kv.pkl","rb"))

print(model.similarity('United_States','U.S.'))

"""
[プログラムの結果]
%python 61.py
0.73107743
"""