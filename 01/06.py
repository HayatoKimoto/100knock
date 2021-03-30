import copy 
strX = "paraparaparadise"
strY = "paragraph"

def MakeNgram(sequence, N):
    ngramlist = []
    for i in range(len(sequence)):
        ngramlist.append(sequence[i:i+N])

    return ngramlist

biagram_X = MakeNgram(strX, 2)
biagram_Y = MakeNgram(strY, 2)

#print(biagram_X)
#print(biagram_Y)

biagram_X = list(set(biagram_X))
biagram_Y = list(set(biagram_Y))

#print(biagram_X)
#print(biagram_Y)

wa = []#xUy
seki = []#x∩y
sa = []  #x/y
wa=biagram_X.copy()
sa=biagram_X.copy()
for i in biagram_Y:
    if (i in biagram_X):
        sa.remove(i)
        seki.append(i)
    else:
        wa.append(i)

print("和集合:") 
print(wa)
print("積集合:") 
print(seki)
print("差集合:") 
print(sa)

if ('se' in biagram_X):
    print("seは集合Xに含まれる.")
else:
    print("seは集合Xに含まれない.")

    
if ('se' in biagram_Y):
    print("seは集合Yに含まれる.")
else:
    print("seは集合Yに含まれない.")

