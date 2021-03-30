f  = open('popular-names.txt', 'r')
c1 = open("col1.txt", 'w')
c2 = open("col2.txt", 'w')

for l in f:
    tmplist = []
    tmplist = l.split("\t")
    c1.write(tmplist[0]+"\n")
    c2.write(tmplist[1]+"\n")

f.close()
c1.close()
c2.close()
    