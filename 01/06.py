import copy 
strX = "paraparaparadise"
strY = "paragraph"

def makeNgram(sequence, N):
    ngramlist = []
    for i in range(len(sequence)):
        ngramlist.append(sequence[i:i+N])

    return ngramlist

biagram_X = makeNgram(strX, 2)
biagram_Y = makeNgram(strY, 2)

biagram_X = list(set(biagram_X))
biagram_Y = list(set(biagram_Y))

#print(biagram_X)
#print(biagram_Y)

wa = biagram_X.copy()  #xUy 和集合
seki = []              #x∩y 積集合
sa = biagram_X.copy()  #x/y 差集合

for d in biagram_Y:
    if (d in biagram_X):
        sa.remove(d)
        seki.append(d)
    else:
        wa.append(d)

#確認
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

