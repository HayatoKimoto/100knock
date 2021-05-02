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

def make_n_freq_counter_list_with_X(morphemes,n,x):
    word_list=[]
    for line in morphemes:
        if any(d['base']== x for d in line):
            for w in line:

                #名詞のみ
                if w['base']!= x and w['base'] != '*\n' and w['pos'] == '名詞': word_list.append(w['base'])

                #全ての品詞
                #if w['base']!= x and w['base'] != '*\n': word_list.append(w['base'])

    c=Counter(word_list)
    word_most_common_list=c.most_common(n)
    return word_most_common_list

def make_bar_chart(counter_list):
    left=[]
    height=[]
    for v in counter_list:
        left.append(v[0])
        height.append(v[1])


    height=np.array(height)
    left=np.array(left)


    plt.bar(left, height)
    plt.savefig('ans37.png')
    plt.show()
            
Morphemes=make_morphemes(fname)

ten_word_list=make_n_freq_counter_list_with_X(Morphemes,10,'猫')
#print(make_n_freq_counter_list_with_X(Morphemes,10,'猫'))
make_bar_chart(ten_word_list)
"""
[プログラムの結果]
出力結果のグラフはans37.pngに保存
"""