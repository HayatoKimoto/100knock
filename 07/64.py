import gensim
import pickle
import re
from tqdm import tqdm

model = pickle.load(open("kv.pkl","rb"))
with open('questions-words.txt') as f:
    questions = f.readlines()
with open('ans64.txt','w')as f:
    for line in tqdm(questions):
        if re.match(': ',line):
            f.write(line)
            continue
        line=line.replace('\n',' ')
        data_list=line.split()

        ans = model.most_similar(positive=[data_list[1],data_list[2]], negative=[data_list[0]],topn=1)[0]

        line += ans[0]+' '+str(ans[1])

        f.write(line+'\n')




