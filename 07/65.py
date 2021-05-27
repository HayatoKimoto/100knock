import gensim
import pickle
import re
from sklearn.metrics import accuracy_score

with open('ans64.txt') as f:
    lines = f.readlines()

label_true = []
pred = []
for line in lines:
    if re.match(': ',line):continue

    line = line.split()

    
    label_true.append(line[3])
    pred.append(line[4])

print(accuracy_score(label_true,pred))

"""
[プログラムの結果]
%python 65.py
0.7358780188293083
"""


