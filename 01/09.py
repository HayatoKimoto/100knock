import random
s = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
#
wordlist = s.split(' ')

ans =[]
for d in wordlist:
    if (len(d) <= 4):
        ans.append(d)
    else:
        tmp = d[1:-1]
        word = d[0] + "".join(random.sample(tmp, len(tmp))) + d[-1]
        ans.append(word)


print(ans)



        