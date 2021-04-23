with open('popular-names.txt', 'r') as f,open("col1.txt", 'w')as c1,open("col2.txt", 'w') as c2:
    for l in f:
        c1.write(l.split('\t')[0]+'\n')
        c2.write(l.split('\t')[1]+'\n')
"""
[UNIXコマンド]
%cut -f 1 popular-names.txt
Mary
Anna
Emma
Elizabeth
Minnie
Margaret
Ida
Alice
Bertha
Sarah

%cut -f 2 popular-names.txt
F
F
F
F
F
F
F
F
F
F

[プログラムの結果]
%open col1.txt
Mary
Anna
Emma
Elizabeth
Minnie
Margaret
Ida
Alice
Bertha
Sarah

%open col2.txt
F
F
F
F
F
F
F
F
F
F

"""