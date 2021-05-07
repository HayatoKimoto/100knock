import MeCab
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

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

def make_double_log_graph(counter_list):
    left=[]
    height=[]
    for i,v in enumerate(counter_list):
        left.append(i+1)
        height.append(v[1])


    height=np.array(height)
    left=np.array(left)

    plt.plot(left,height)
    ax=plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.savefig('ans39.png')
    plt.show()
            
Morphemes=make_morphemes(fname)

make_double_log_graph(make_freq_counter_list(Morphemes))
"""
[プログラムの結果]
出力結果のグラフはans39.pngに保存
"""