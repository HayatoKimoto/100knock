import MeCab
from collections import Counter

fname='neko.txt.mecab'
def make_morphemes(filename):
    with open(filename) as data_file:
        #形態素解析の辞書
        morphemes=[]
        sentence=[]
        
        for line in data_file:
            #文章の最後(EOS)まできたらfor文から抜ける
            if len(line.split('\t')) < 2:
                break

            words=line.split('\t')[1].split(',')
            morpheme={
                'surface':line.split('\t')[0],
                'base'   :words[6],
                'pos'    :words[0],
                'pos1'   :words[1]
            }
            sentence.append(morpheme)

            if words[1]=='句点' :
                #print(sentence)
                morphemes.append(sentence)
                sentence=[]

    return morphemes

def make_freq_counter_list(morphemes):
    word_list=[]
    for line in morphemes:
        for d in line:
            word_list.append(d['base'])

    c=Counter(word_list)

    word_most_common_list=c.most_common()
    return word_most_common_list

            
Morphemes=make_morphemes(fname)

print(make_freq_counter_list(Morphemes))

"""
[プログラムの結果](長いので省略)
% python 35.py
[('の', 9194), ('。', 7486), ('て', 6853), ('、', 6772), ('は', 6422), ('に', 6268), ('を', 6071), ('だ', 5978), ('と', 5515), 
"""