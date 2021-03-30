str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
#先頭1文字を取り出す単語
indexlist = [1, 5, 6, 7, 8, 9, 15, 16, 19]
elemlist={}

#アルファベット以外の文字を取り除く
str = str.replace(',', '')
str = str.replace('.', '')

#print(str)

wordlist = str.split(' ')
for i in range(len(wordlist)):
        if ((i + 1) in indexlist):
            tmp=wordlist[i]
            element = {tmp[0]:i+1}
            elemlist.update(element)
        else:
            tmp=wordlist[i]
            element = {tmp[0:2]:i+1}
            elemlist.update(element)
print(elemlist["Na"])
        
    