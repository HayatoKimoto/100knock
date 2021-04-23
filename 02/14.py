import sys

n = int(sys.argv[1])

with open("popular-names.txt", "r")as f:
    count = 0

    if n < 0:
        sys.exit(1)
    for l in f:
        if n == count:
            break
        else:
            print(l, end="")
            count += 1

 """
[UNIXコマンド]
%head -n 10 popular-names.txt

Mary	F	7065	1880
Anna	F	2604	1880
Emma	F	2003	1880
Elizabeth	F	1939	1880
Minnie	F	1746	1880
Margaret	F	1578	1880
Ida	F	1472	1880
Alice	F	1414	1880
Bertha	F	1320	1880
Sarah	F	1288	1880

[プログラムの結果]   
%python 14.py 10
Mary	F	7065	1880
Anna	F	2604	1880
Emma	F	2003	1880
Elizabeth	F	1939	1880
Minnie	F	1746	1880
Margaret	F	1578	1880
Ida	F	1472	1880
Alice	F	1414	1880
Bertha	F	1320	1880
Sarah	F	1288	1880

"""      

        

