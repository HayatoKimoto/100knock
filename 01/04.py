s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
#先頭1文字を取り出す単語
indexlist = [1, 5, 6, 7, 8, 9, 15, 16, 19]
elemlist={}

#アルファベット以外の文字を取り除く
s = s.replace(',', '').replace('.', '')

print(s)

wordlist = s.split(' ')
for d,i in zip(wordlist,range(1,len(wordlist)+1)):
        if (i in indexlist):
            element = {d[0]:i}
        else:
            element = {d[0:2]:i}
        
        elemlist.update(element)

#確認
print(elemlist)       
    