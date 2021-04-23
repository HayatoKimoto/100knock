with open("col1.txt", 'r')as c1, open("col2.txt", 'r')as c2, open("mergefile.txt", 'w')as mf:

    for l1, l2 in zip(c1, c2):
        l1 = l1.replace('\n','')
        l2 = l2.replace('\n','')
        mf.write(l1+'\t'+l2+'\n')
"""
[UNIXコマンド]
%paste col1.txt col2.txt
Mary	F
Anna	F
Emma	F
Elizabeth	F
Minnie	F
Margaret	F
Ida	F
Alice	F
Bertha	F
Sarah	F

[プログラムの結果]   
%open mergefile.txt
Mary	F
Anna	F
Emma	F
Elizabeth	F
Minnie	F
Margaret	F
Ida	F
Alice	F
Bertha	F
Sarah	F
"""