s1 = "KimotoHayato"
s2="Dentsudai18"

def cipher(s):
    ans=""
    for i in s:
        if (i.islower()): #islower関数は全ての文字が小文字かどうかを判断する関数
            ans += chr(219 - ord(i)) #ord関数は文字を文字コードに変更する関数
        else:
            ans += i
            
    return ans


#結果
print(cipher(s1))
print(cipher(s2))




