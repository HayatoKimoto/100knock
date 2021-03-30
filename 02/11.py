f  = open('popular-names.txt', 'r')
f1 = open("popular-names-copy11.txt", 'w')


count=0
for l in f:
    l=l.replace("\t"," ")
    f1.write(l)

f.close()
f1.close()
    
