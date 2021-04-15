s = "I am an NLPer"
#
def makeNgram(sequence, N):
    ngramlist = []

    for i in range(len(sequence)):
        ngramlist.append(sequence[i:i+N])

    return ngramlist

#bigram_cが文字bi-gram、bigram_wが単語bi-gramを得るための変数

#空白文字を消去
bigram_c = s.replace(' ', '')
#単語を保つためにリストに変更
bigram_w = s.split(' ')

print(makeNgram(bigram_c, 2))
print(makeNgram(bigram_w, 2))