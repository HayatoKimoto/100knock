with open("popular-names.txt", "r")as f:
    l1_list=[]
    for l in f:
        l1_list.append(l.split("\t")[0])

    #辞書のキーは重複した要素を持たないので辞書に変更した後，リストに変えている
    l1_unique_list = list(dict.fromkeys(l1_list))

    #unixコマンドの結果と一致させるためにソートした
    l1_unique_list.sort()
    print(l1_unique_list)



"""
[UNIXコマンド](長いので結果省略)
%cut -f 1 popular-names.txt | sort | uniq 
Abigail
Aiden
Alexander
Alexis
Alice
Amanda
Amelia
Amy

[プログラムの結果](長いので結果省略)  
%python 17.py 
['Abigail', 'Aiden', 'Alexander', 'Alexis', 'Alice', 'Amanda', 'Amelia', 'Amy', 'Andrew', 'Angela', 'Anna', 'Annie',
"""      