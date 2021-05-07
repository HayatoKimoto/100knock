import MeCab

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

def extract_noun_noun(morphemes):
    words_list=[]
    for line in morphemes:
        tmp=[]
        for d in line:
            if d['pos']=='名詞':
                tmp.append(d['surface'])
            else:
                if len(tmp) > 1:
                    words_list.append(''.join(tmp))

                tmp=[]

    return words_list

            
Morphemes=make_morphemes(fname)

print(extract_noun_noun(Morphemes))

"""
実行結果(長いので先頭部分のみ)
% python 34.py
['人間中', '一番獰悪', '時妙', '一毛', 'その後猫', '一度', 'ぷうぷうと煙', '邸内', '三毛', '書生以外', '四五遍', 'この間おさん', 
 '三馬', '御台所', 'まま奥', '住家', '終日書斎', '勉強家', '勉強家', '勤勉家', '二三ページ', '主人以外', '限り吾輩', '朝主人',

"""