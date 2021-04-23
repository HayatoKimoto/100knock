import sys

n = int(sys.argv[1])
with open("popular-names.txt", "r")as f:
    count = 0

    lines=f.readlines()
    if n < 0:
        sys.exit(1)


    for i in range(n):
        print("".join(lines[-1*(n-i)]),end="")
                

"""
[UNIXコマンド]ƒ
%tail -n 10 popular-names.txt   

Liam	M	19837	2018
Noah	M	18267	2018
William	M	14516	2018
James	M	13525	2018
Oliver	M	13389	2018
Benjamin	M	13381	2018
Elijah	M	12886	2018
Lucas	M	12585	2018
Mason	M	12435	2018
Logan	M	12352	2018
[プログラムの結果]   
%python 15.py 10
Liam	M	19837	2018
Noah	M	18267	2018
William	M	14516	2018
James	M	13525	2018
Oliver	M	13389	2018
Benjamin	M	13381	2018
Elijah	M	12886	2018
Lucas	M	12585	2018
Mason	M	12435	2018
Logan	M	12352	2018
"""       

