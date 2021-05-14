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

    def get_base_of_verb(self):
        for x in self.morphs:
            if x.pos =='動詞':return x.base

        return ''    

    def is_noun(self):
        for x in self.morphs:
            if x.pos =='名詞':return True

        return False

    def get_particle(self):
        for x in self.morphs:
            if x.pos == '助詞':return x.surface

        return ''

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

def extract_case_patterns_of_verbs(sentence):
    patterns_of_verbs_list=[]
    for chunk in sentence:
        if chunk.dst == -1 :continue

        if chunk.get_phrase() != '' and chunk.is_verb():
            verb=chunk.get_base_of_verb()
            particles_list=[]
            for i in chunk.srcs:
                particle=sentence[i].get_particle()
                if particle!='' :particles_list.append(particle)

            particles_list.sort()
            if particles_list:#listが空かどうかを判定
                s=verb+('\t')+' '.join(particles_list)
                patterns_of_verbs_list.append(s)

    return patterns_of_verbs_list


    

fname='ai.ja.txt.parsed'
article=get_chunk_list(fname)


answer_list=[]
for s in article:
    patterns_of_verbs_list=extract_case_patterns_of_verbs(s)
    answer_list.extend(patterns_of_verbs_list)

s='\n'.join(answer_list)
with open('ans45.txt','w') as f:
    f.write(s)

print(s)

'''
[プログラムの結果]
ans45.txtファイルに保存
unixコード
sort ans45.txt | uniq -c | sort -n -r | head
  48 する       を
  16 する       に
  15 する       が
  14 する       と
  10 する       は を
  10 する       に を
   9 する       で を
   9 よる       に
   8 する       が に
   8 行う       を
grep "^行う\t" ans45.txt | sort | uniq -c | sort -n -r | head
   8 行う       を
   1 行う       まで を
   1 行う       から
   1 行う       に まで を
   1 行う       は を をめぐって
   1 行う       に に により を
   1 行う       て に は は
   1 行う       が て で に
   1 行う       に を を
   1 行う       で に を        
grep "^なる\t" ans45.txt | sort | uniq -c | sort -n -r | head
   3 なる       に は
   2 なる       が と
   2 なる       に
   2 なる       と
   1 なる       から が て で と
   1 なる       から で と
   1 なる       て として に は
   1 なる       が と にとって
   1 なる       で に に
   1 なる       て に は
grep "^与える\t" ans45.txt | sort | uniq -c | sort -n -r | head
   1 与える     が など
   1 与える     に は を
   1 与える     が に


'''

