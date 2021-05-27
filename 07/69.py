import gensim
import pickle
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = pickle.load(open("kv.pkl","rb"))

def make_countries_list(filename):
    with open(filename) as f:
        questions = f.readlines()

    countries = set()
    for line in questions:
        #: capital-common-countriesと: capital-worldに記載されている国名をリストとする
        #: currency以降は除外。
        if re.match(': ',line):
            if re.match(': currency',line):break
            continue

        data_list=line.split()
        countries.add(data_list[1])
    countries = list(countries)

    return countries

countries = make_countries_list('questions-words.txt')
#print(countries)

countries_vec_list = model[countries]
#print(countries_vec_list)

tsne_model = TSNE(n_components=2)
vec_embedded = tsne_model.fit_transform(countries_vec_list)
vec_embedded_t = list(zip(*vec_embedded)) # 転置
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(*vec_embedded_t)
for i, c in enumerate(countries):
    ax.annotate(c, (vec_embedded[i][0],vec_embedded[i][1]))

fig.savefig("69.png")
plt.show()

