str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

#アルファベット以外の文字を取り除く
str = str.replace(',', '')
str = str.replace('.', '')

#print(str)

wordlist = str.split(' ')

countlist=[]
for i in wordlist:
    countlist.append(len(i))

print(countlist)
