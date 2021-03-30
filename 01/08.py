s = "KimotoHayato"
s1="Dentsudai18"
def cipher(str):
    ans=""
    for i in str:
        if (i.islower()):
            ans += chr(219 - ord(i))
        else:
            ans += i
            
    return ans



print(cipher(s))
print(cipher(s1))




