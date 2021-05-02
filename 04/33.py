import MeCab

fname='neko.txt.mecab'
def make_morphemes(filename):
    with open(filename) as data_file:
        #形態素解析の辞書
        morphemes=[]
        sentence=[]
        
        for line in data_file:
            #文章の最後(EOS)まできたらfor文から抜ける
            if len(line.split('\t'))<2:
                break

            words=line.split('\t')[1].split(',')
            morpheme={
                'surface':line.split('\t')[0],
                'base'   :words[6],
                'pos'    :words[0],
                'pos1'   :words[1]
            }
            sentence.append(morpheme)

            if words[1]=='句点':
                #print(sentence)
                morphemes.append(sentence)
                sentence=[]

    return morphemes

def extractA_no_B(morphemes):
    nplist=[]
    for line in morphemes:
        for i in range(1,len(line)-1):
            if line[i-1]['pos']=='名詞' and line[i]['base']=='の' and line[i+1]['pos']=='名詞':
                nplist.append(line[i-1]['surface']+line[i]['surface']+line[i+1]['surface'])

    return nplist

Morphemes=make_morphemes(fname)

print(extractA_no_B(Morphemes))

"""
実行結果(長いので先頭部分のみ)
% python 34.py
['彼の掌', '掌の上', '書生の顔', 'はずの顔', '顔の真中', '穴の中', '書生の掌', '掌の裏', '何の事',

"""