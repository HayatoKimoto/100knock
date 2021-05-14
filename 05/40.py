class Morph:
    def __init__(self,surface,base,pos,pos1):
        self.surface=surface
        self.base=base
        self.pos=pos
        self.pos1=pos1


    def show(self):
        print(self.surface,self.base,self.pos,self.pos1)


with open('ai.ja.txt.parsed') as f:
    morphs=[]
    morphs_line_list=[]
    for line in f:
        line=line.replace('\n','')
        if line[0] == '*':
            continue
        elif line.split('\t')[0] == 'EOS':
            morphs_line_list.append(morphs)
            morphs=[]
        else:

            words=line.split('\t')[1].split(',')
            morpheme=Morph(line.split('\t')[0],words[6],words[0],words[1])
            morphs.append(morpheme)
            

for morph in morphs_line_list[2]:#morphs_line_listは冒頭の説明文の形態素列
    morph.show()

'''
[プログラムの結果](長いので一部省略)
人工 人工 名詞 一般
知能 知能 名詞 一般
（ （ 記号 括弧開
じん じん 名詞 一般
こうち こうち 名詞 一般
のう のう 助詞 終助詞
、 、 記号 読点
、 、 記号 読点
AI * 名詞 一般
〈 〈 記号 括弧開
エーアイ * 名詞 固有名詞
〉 〉 記号 括弧閉
） ） 記号 括弧閉

'''