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

    def get_base_of_most_left_verb(self):
        for x in self.morphs:
            if x.pos =='動詞':return x.base

        return ''    

    def is_noon(self):
        for x in self.morphs:
            if x.pos =='名詞':return True

        return False

    def get_particle(self):
        for x in self.morphs:
            if x.pos == '助詞':return x.surface

        return ''

    def is_functions_verben(self):
        flag=0
        #s=''
        for x in self.morphs:
            if flag == 1 and x.base =='を':
                return True
            elif x.pos1 == 'サ変接続':
                flag = 1
                #s=x.surface
            else :
                flag = 0
                #s=''


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

def extract_synac_of_functions_verben(sentence):
    functions_verben_list=[]
    for i,chunk in enumerate(sentence,0):
        if chunk.dst == -1 :continue

        if chunk.is_functions_verben():
            phrase=chunk.get_phrase()
            if sentence[chunk.dst].is_verb():
                phrase+=sentence[chunk.dst].get_base_of_most_left_verb()
                pairs_list=[]
                for num in sentence[chunk.dst].srcs:
                    particle=sentence[num].get_particle()
                    if particle != '':
                        pairs=[particle,sentence[num].get_phrase()]
                        pairs_list.append(pairs)

                #print(pairs_list)
                pairs_list=sorted(pairs_list,key=lambda x :x[1])
                pairs_list=sorted(pairs_list,key=lambda x :x[0])
                if pairs_list:
                    s=phrase+('\t')+' '.join(x[0] for x in pairs_list)+(' ')+' '.join(x[1] for x in pairs_list)
                    functions_verben_list.append(s)

    return functions_verben_list


    

fname='ai.ja.txt.parsed'
article=get_chunk_list(fname)


answer_list=[]
for s in article:
    patterns_of_verbs_list=extract_synac_of_functions_verben(s)
    answer_list.extend(patterns_of_verbs_list)

s='\n'.join(answer_list)
with open('ans47.txt','w') as f:
    f.write(s)

print(s)