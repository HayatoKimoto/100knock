import sys

n = int(sys.argv[1])

f=open("popular-names.txt", "r")
file_num = 1
file = open('file' + str(file_num) + '.txt', 'w')

for l in f:
    if (file_num % n == 0):
        file = open('file' + str(int(file_num / n + 1)) + '.txt', 'w')
    

    file.write(l)
    file_num+=1
    

        
    