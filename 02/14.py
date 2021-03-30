import sys

n = int(sys.argv[1])

f=open("popular-names.txt", "r")
count = 0

if n < 0:
    sys.exit(1)
for l in f:
    if n == count:
        break
    else:
        print(l, end="")
        count += 1
        
        

        

