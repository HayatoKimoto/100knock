import MeCab

fname='neko.txt.mecab'
def makeMorphemes(filename):
    with open(filename) as data_file:
        #形態素解析の辞書
        Morphemes=[]
        sentence=[]
        
        for line in data_file:
            #文章の最後(EOS)まできたらfor文から抜ける
            if(len(line.split('\t'))<2):
                break

            words=line.split('\t')[1].split(',')
            morpheme={
                'surface':line.split('\t')[0],
                'base'   :words[6],
                'pos'    :words[0],
                'pos1'   :words[1]
            }
            sentence.append(morpheme)

            if(words[1]=='句点'):
                #print(sentence)
                Morphemes.append(sentence)
                sentence=[]

    return Morphemes

def extractVerbs_sur(morphemes):
    verbslist=[]
    for line in morphemes:
        for morpheme in line:
            if(morpheme['pos']=='動詞'):
                #print(morpheme['pos']+' '+morpheme['surface'])
                verbslist.append(morpheme['surface'])
    
    print(verbslist)



extractVerbs_sur(makeMorphemes(fname))





          