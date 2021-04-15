s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

#アルファベット以外の文字を取り除く
s = s.replace(',', '').replace('.', '')

#print(str)

wordlist = s.split(' ')

anslist=[]
for d in wordlist:
    anslist.append(len(d))

print(anslist)
