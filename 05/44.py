from graphviz import Digraph

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
        #print('係り元文節インデックス番号のリスト(srcs):',self.srcs)

    def get_phrase(self):
        phrase=''
        for x in self.morphs:
            if x.pos !='記号':phrase+=x.surface
        
        return phrase

    def is_verb(self):
        for x in self.morphs:
            if x.pos =='動詞':return True

        return False

    def is_noon(self):
        for x in self.morphs:
            if x.pos =='名詞':return True

        return False

def get_chunk_list(filename):
    with open(filename) as f:
        morphs=[]
        morphs_line_list=[]
        chunk = None
        chunks=[]
        sentence=[]
        for line in f:
            line=line.replace('\n','')
            if line[0] == '*':
                inf=line.split(' ')
                dst_num=int(inf[2][:-1])
                chunk=Chunk(dst_num)
                chunks.append(chunk)
            elif line == 'EOS':
                if chunks :
                    for i,c in enumerate(chunks,0):
                        chunks[c.dst].srcs.append(i)
                    sentence.append(chunks)

                morphs_line_list.append(morphs)
                morphs=[]
                chunks=[]
            else:
                words=line.split('\t')[1].split(',')
                morpheme=Morph(line.split('\t')[0],words[6],words[0],words[1])
                morphs.append(morpheme)
                chunk.morphs.append(morpheme)

        return sentence

def make_pair_phrase_graph(sentence,path):
    pair_phrase_list=[]
    dot = Digraph(comment='ans44')
    for i,chunk in enumerate(sentence,0):
        if chunk.dst == -1 :continue

        pair_phrase=chunk.get_phrase()

        if pair_phrase != '':
            x=str(i)
            y=str(chunk.dst)
            dot.node(x,pair_phrase)
            dot.node(y,sentence[chunk.dst].get_phrase())
            dot.edge(x, y)

    dot.render(path, view=False)
    

fname='ai.ja.txt.parsed'
article=get_chunk_list(fname)



for i,s in enumerate(article,1):   
    file_path = 'test-output/ans44-' + str(i) + '.gv'
    make_pair_phrase_graph(s,file_path)



