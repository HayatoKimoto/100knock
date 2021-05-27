import gensim
import pickle
import re
from sklearn.cluster import KMeans

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

kmeans_model = KMeans(n_clusters=5, random_state=0)
kmeans_model.fit(countries_vec_list)

predicted_list = list(kmeans_model.predict(countries_vec_list))

for i in range(5):
    print(i,' '.join(sorted([countries[num] for num in range(len(predicted_list)) if predicted_list[num] == i])))
    

"""
[プログラムの結果]
0 Albania Austria Belgium Bulgaria Canada Croatia Cyprus Denmark England Estonia Finland France Georgia Germany Greece Greenland Hungary Ireland Italy Latvia Liechtenstein Lithuania Macedonia Malta Montenegro Norway Poland Portugal Romania Serbia Slovakia Slovenia Spain Sweden Switzerland Turkey
1 Algeria Angola Botswana Burundi Eritrea Gabon Gambia Ghana Guinea Kenya Liberia Libya Madagascar Malawi Mali Mauritania Morocco Mozambique Namibia Niger Nigeria Rwanda Senegal Somalia Sudan Tunisia Uganda Zambia Zimbabwe
2 Afghanistan Australia Bahrain Bangladesh Bhutan China Egypt Indonesia Iran Iraq Japan Jordan Laos Lebanon Nepal Oman Pakistan Qatar Syria Taiwan Thailand Vietnam
3 Armenia Azerbaijan Belarus Kazakhstan Kyrgyzstan Moldova Russia Tajikistan Turkmenistan Ukraine Uzbekistan
4 Bahamas Belize Chile Cuba Dominica Ecuador Fiji Guyana Honduras Jamaica Nicaragua Peru Philippines Samoa Suriname Tuvalu Uruguay Venezuela
"""


