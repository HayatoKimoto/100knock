import pandas as pd
import sys
f = open("popular-names.txt", "r")
l_list = []

for l in f:
    tmplist = []
    tmplist = l.split("\t")
    tmplist[2]=int(tmplist[2])
    l_list.append(tmplist)

df = pd.DataFrame(l_list, columns=['j0', 'j1', 'j2', 'j3'])
df = (df.sort_values('j2', ascending=False))
df=df.values.tolist()

for l in df:
    l[2]=str(l[2])
    print('\t'.join(l),end="")

#sort -n -k 3 -r popular-names.txt

