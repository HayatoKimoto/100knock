str = "パタトクカシーー"
ans = ""
'''
for i in range(int(len(str)/2)):
    ans += str[2 * i]
    
print(ans)
'''
ans = str[::2]
print(ans)