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

            if words[1]=='句点' :
                #print(sentence)
                Morphemes.append(sentence)
                sentence=[]

    return Morphemes

def extract_noun_noun(morphemes):
    len_words=[]
    for line in morphemes:
        tmp=[]
        for d in line:
            if d['pos']=='名詞':
                tmp.append(d['surface'])
            else:
                if len(tmp)>1:
                    len_words.append(''.join(tmp))

                tmp=[]

    return len_words

            



print(extract_noun_noun(makeMorphemes(fname)))


#実行結果(長いので先頭部分のみ)

#% python 34.py
#['人間中', '一番獰悪', '時妙', '一毛', 'その後猫', '一度', 'ぷうぷうと煙', '邸内', '三毛', '書生以外', '四五遍', 'この間おさん', 
# '三馬', '御台所', 'まま奥', '住家', '終日書斎', '勉強家', '勉強家', '勤勉家', '二三ページ', '主人以外', '限り吾輩', '朝主人',