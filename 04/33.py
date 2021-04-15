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

def extractA_to_B(morphemes):
    nplist=[]
    for line in morphemes:
        for i in range(1,len(line)-1):
            if(line[i-1]['pos']=='名詞' and line[i]['base']=='の' and line[i+1]['pos']=='名詞'):
                #print(line[i-1]['base']+line[i]['base']+line[i+1]['base'])
                nplist.append(line[i-1]['surface']+line[i]['surface']+line[i+1]['surface'])

    return nplist




            




print(extractA_to_B(makeMorphemes(fname)))