#1章の問題は不用意な変数を減らすようにしてコードを書きました.

s = "stressed"
ans = ""

for i in range(len(s)):
    ans+=s[-(i+1)]

print(ans)

