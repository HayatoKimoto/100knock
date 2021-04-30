import MeCab

fname='neko.txt.mecab'
def make_morphemes(filename):
    with open(filename) as data_file:
        #形態素解析の辞書
        morphemes=[]
        sentence=[]
        
        for line in data_file:
            #文章の最後(EOS)まできたらfor文から抜ける
            if len(line.split('\t')) < 2 : break

            words=line.split('\t')[1].split(',')
            morpheme={
                'surface':line.split('\t')[0],
                'base'   :words[6],
                'pos'    :words[0],
                'pos1'   :words[1]
            }
            sentence.append(morpheme)

            if words[1]=='句点' :
                morphemes.append(sentence)
                sentence=[]

    return morphemes

def extract_verbs_sur(morphemes):
    verbs_list=[]
    for line in morphemes:
        for morpheme in line:
            if morpheme['pos']=='動詞' :
                verbs_list.append(morpheme['surface'])
    
    return verbs_list


Morphemes=make_morphemes(fname)

print(extract_verbs_sur(Morphemes))


"""
[プログラムの結果](長いので省略)
% python 31.py
['生れ', 'つか', 'し', '泣い', 'し', 'いる', '始め', '見', '聞く', '捕え', '煮', '食う'(省略)
"""
    



          