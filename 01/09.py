import random
str = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

wordlist = str.split(' ')

ans =[]
for s in wordlist:
    if (len(s) <= 4):
        ans.append(s)
    else:
        tmp = s[1:-1]
        word = s[0] + "".join(random.sample(tmp, len(tmp))) + s[-1]
        ans.append(word)

print(ans)



        