f = open("popular-names.txt", "r")
l1_list=[]
for l in f:
    tmplist = []
    tmplist = l.split("\t")
    l1_list.append(tmplist[0])

l1_unique_list = list(dict.fromkeys(l1_list))

print(len(l1_unique_list))

#cut -f 1 popular-names.txt | sort | uniq | wc -l


