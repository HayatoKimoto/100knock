import sys

n = int(sys.argv[1])

f=open("popular-names.txt", "r")
count = 0

lines=f.readlines()
if n < 0:
    sys.exit(1)


for i in range(n):
    print("".join(lines[-1*(n-i)]),end="")
        
        

        

