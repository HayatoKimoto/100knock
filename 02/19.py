import collections
#
f = open("popular-names.txt", "r")
l1_list=[]
for l in f:
    tmp_list = []
    tmp_list = l.split("\t")
    l1_list.append(tmp_list[0])

l1_list.sort()
c=collections.Counter(l1_list)
for i in c.most_common():
    print(i[0])


"""
[UNIXコマンド](長いので結果省略)
%cut -f 1 popular-names.txt | sort | uniq -c|sort -nr 
 118 James
 111 William
 108 Robert
 108 John
  92 Mary
  75 Charles
  74 Michael
  73 Elizabeth
  70 Joseph
  60 Margaret
  58 Thomas

[プログラムの結果](長いので省略)  
%python 19.py
James
William
John
Robert
Mary
Charles
Michael
Elizabeth
Joseph


"""       


        
