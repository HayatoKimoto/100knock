import pandas as pd
import sys
with open("popular-names.txt", "r")as f:
    l_list = []

    for l in f:
        tmp_list = []
        tmp_list = l.split("\t")
        tmp_list[2]=int(tmp_list[2])
        l_list.append(tmp_list)

    df = pd.DataFrame(l_list, columns=['j0', 'j1', 'j2', 'j3'])
    df = (df.sort_values('j2', ascending=False))
    df=df.values.tolist()

    for l in df:
        l[2]=str(l[2])
        print('\t'.join(l),end="")


"""
[UNIXコマンド](長いので結果省略)
%sort -n -k 3 -r popular-names.txt
Linda	F	99689	1947
Linda	F	96211	1948
James	M	94757	1947
Michael	M	92704	1957
Robert	M	91640	1947
Linda	F	91016	1949
Michael	M	90656	1956
Michael	M	90517	1958
James	M	88584	1948

[プログラムの結果](長いので結果省略)
%python 18.py
Linda	F	99689	1947
Linda	F	96211	1948
James	M	94757	1947
Michael	M	92704	1957
Robert	M	91640	1947
Linda	F	91016	1949
Michael	M	90656	1956
Michael	M	90517	1958
James	M	88584	1948

"""       


        
