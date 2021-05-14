
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

def extract_path_of_noun_to_root(sentence):
    roots_list=[]
    for i,chunk in enumerate(sentence,0):

        if chunk.is_noun():
            tmp=[]
            phrase=chunk.get_phrase()
            num=chunk.dst
            tmp.append(phrase)
            while(num != -1):
                phrase=sentence[num].get_phrase()
                num=sentence[num].dst
                tmp.append(phrase)
            
            root=' -> '.join(tmp)
            roots_list.append(root)
            tmp=[]

    return roots_list


    

fname='ai.ja.txt.parsed'
article=get_chunk_list(fname)


answer_list=[]
for s in article:
    roots_list=extract_path_of_noun_to_root(s)
    answer_list.extend(roots_list)

s='\n'.join(answer_list)
with open('ans48.txt','w') as f:
    f.write(s)

print(s)

'''
[プログラムの結果](長いので一部省略)
ans48.txtファイルに結果を保存
'''