import pandas as pd
import numpy as np
import gensim
import pickle

train_df = pd.read_table('ans50/train.tsv', header=None)
val_df   = pd.read_table('ans50/valid.tsv', header=None)
test_df  = pd.read_table('ans50/test.tsv', header=None)

model = pickle.load(open("kv.pkl","rb"))

d = {'b':0, 't':1, 'e':2, 'm':3}
y_train = train_df.iloc[:,0].replace(d)
y_train.to_csv('y_train.txt',header=False, index=False)
y_valid = val_df.iloc[:,0].replace(d)
y_valid.to_csv('y_valid.txt',header=False, index=False)
y_test  = test_df.iloc[:,0].replace(d)
y_test.to_csv('y_test.txt',header=False, index=False)

def write_X(file_name, df):
    with open(file_name,'w') as f:
        for text in df.iloc[:,1]:
            vectors = []
            for word in text.split():
                if word in model.key_to_index.keys():
                    vectors.append(model[word])
            if len(vectors) == 0:
                vector = np.zeros(300)
            else:
                vectors = np.array(vectors)
                vector  = vectors.mean(axis=0)
            vector = vector.astype(np.str_).tolist()
            output = ' '.join(vector)+'\n'
            f.write(output)


write_X('X_train.txt',train_df)
write_X('X_valid.txt',val_df)
write_X('X_test.txt',test_df)

"""
[プログラムの結果]

"""