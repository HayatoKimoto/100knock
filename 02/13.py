c1 = open("col1.txt", 'r')
c2 = open("col2.txt", 'r')
mf = open("mergefile.txt", 'w')

for l1, l2 in zip(c1, c2):
    l1 = l1.replace("\n","")
    l2 = l2.replace("\n","")
    mf.write(l1+"\t"+l2+"\n")
    
c1.close()
c2.close()
mf.close()
    


    
