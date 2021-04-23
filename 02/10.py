with open('popular-names.txt', 'r')as f:

    count=0
    for l in f:
        count += 1
        
    print(count)
    
"""
[UNIXコマンド]
% wc -l popular-names.txt
   2780 popular-names.txt

[プログラムの実行結果]
%python 10.py 
2780
"""

