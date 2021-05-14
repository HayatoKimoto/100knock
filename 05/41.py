class Morph:
    def __init__(self,surface,base,pos,pos1):
        self.surface=surface
        self.base=base
        self.pos=pos
        self.pos1=pos1


    def show(self):
        print(self.surface,self.base,self.pos,self.pos1)

class Chunk:
    def __init__(self,dst):
        self.morphs=[]
        self.dst=dst
        self.srcs=[]

    def show(self):
        phrase=''
        for x in self.morphs:
            #print(x.surface,x.base,x.pos,x.pos1)
            phrase+=x.surface

        print('文節の文字列:',phrase)
        print('係り先文節インデックス番号(dst):',self.dst)
        print('係り元文節インデックス番号のリスト(srcs):',self.srcs)

"""    
ai.ja.txt.parsedは　cabocha ai.ja.txt -f1で生成
"""

with open('ai.ja.txt.parsed') as f:
    morphs=[]
    chunk = None
    chunks=[]
    sentence=[]
    for line in f:
        line=line.replace('\n','')
        if line[0] == '*':
            inf=line.split(' ')
            dst=int(inf[2][:-1])
            chunk=Chunk(dst)
            chunks.append(chunk)
        elif line == 'EOS':
            if chunks :
                #出てくる順番がインデックス番号と一致
                for i,c in enumerate(chunks,0):
                    chunks[c.dst].srcs.append(i)
                sentence.append(chunks)

            morphs=[]
            chunks=[]
        else:
            words=line.split('\t')[1].split(',')
            morpheme=Morph(line.split('\t')[0],words[6],words[0],words[1])
            morphs.append(morpheme)
            chunk.morphs.append(morpheme)

for i,c in enumerate(sentence[1],0):
    print(i)
    c.show()

'''
[プログラムの結果](長いので一部省略)
0
文節の文字列: 人工知能
係り先文節インデックス番号(dst): 17
係り元文節インデックス番号のリスト(srcs): []
1
文節の文字列: （じんこうちのう、、
係り先文節インデックス番号(dst): 17
係り元文節インデックス番号のリスト(srcs): []
2
文節の文字列: AI
係り先文節インデックス番号(dst): 3
係り元文節インデックス番号のリスト(srcs): []
3
文節の文字列: 〈エーアイ〉）とは、
係り先文節インデックス番号(dst): 17
係り元文節インデックス番号のリスト(srcs): [2]
4

'''