import collections
f = open("popular-names.txt", "r")
l1_list=[]
for l in f:
    tmplist = []
    tmplist = l.split("\t")
    l1_list.append(tmplist[0])

l1_list.sort()
c=collections.Counter(l1_list)
for i in c.most_common():
    print(i[0])

#cut -f 1 popular-names.txt | sort | uniq -c|sort -nr         
