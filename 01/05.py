str = "I am an NLPer"

def MakeNgram(sequence, N):
    ngramlist = []
    for i in range(len(sequence)):
        ngramlist.append(sequence[i:i+N])

    return ngramlist
#空白文字を消去
bigram_c = str.replace(' ', '')
#単語を保つためにリストに変更
bigram_w = str.split(' ')

print(MakeNgram(bigram_c, 2))
print(MakeNgram(bigram_w, 2))