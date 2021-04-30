import MeCab

fname='neko.txt.mecab'
def make_morphemes(filename):
    with open(filename) as data_file:
        #形態素解析の辞書
        morphemes=[]
        sentence=[]
        
        for line in data_file:
            #文章の最後(EOS)まできたらfor文から抜ける
            if len(line.split('\t')) < 2: break

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

print(make_morphemes(fname))

"""
[プログラムの結果](長いので省略)
% python 30.py
[[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}, {'surface': '\u3000'(省略)
"""
    




    

        



