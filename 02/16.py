import sys

n = int(sys.argv[1])

with open("popular-names.txt", "r") as f:
    file_num = 1
    file = open('file' + str(file_num) + '.txt', 'w')

    for l in f:
        if file_num % n == 0:
            file = open('file' + str(int(file_num / n + 1)) + '.txt', 'w')
        
        file.write(l)
        file_num+=1
    
"""
[UNIXコマンド]
% split -l 500 popular-names.txt file-
% ls
10.py				col2.txt
11.py				file-aa
12.py				file-ab
13.py				file-ac
14.py				file-ad
15.py				file-ae
16.py				file-af
17.py				mergefile.txt
18.py				popular-names-copy11.txt
19.py				popular-names.txt
col1.txt

[プログラムの結果]   
%python 16.py 500
%ls
10.py				file-ac
11.py				file-ad
12.py				file-ae
13.py				file-af
14.py				file1.txt
15.py				file2.txt
16.py				file3.txt
17.py				file4.txt
18.py				file5.txt
19.py				file6.txt
col1.txt			mergefile.txt
col2.txt			popular-names-copy11.txt
file-aa				popular-names.txt
file-ab


"""       


        
    