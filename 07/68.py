import gensim
import pickle
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

countries_vec_list =model[countries]
#print(countries_vec_list)


fig=plt.figure(figsize=(32.0, 24.0))
link = linkage(countries_vec_list, method='ward')
dendrogram(link, labels=countries,leaf_rotation=90,leaf_font_size=10)
fig.savefig("68.png")
plt.show()