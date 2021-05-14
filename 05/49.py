
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

    def get_phrase_with_mask(self, mask, bl=False):
        #blがTrueの場合は最左の名詞をマスクした以降は切り捨てて返す
        phrase=''
        flag=0
        for x in self.morphs:
            if x.pos !='記号':
                if x.pos =='名詞':
                    if flag == 1:continue
                    phrase+=mask
                    if bl :return phrase
                    flag = 1
                else:
                    phrase+=x.surface

        
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

        for x in self.morphs:
            if flag == 1 and x.base =='を':
                return True
            elif x.pos1 == 'サ変接続':
                flag = 1

            else :
                flag = 0



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

def extract_depency_path(sentence):
    roots_list=[]
    for chunk in sentence:
        index_noun_list=[i for i in range(len(sentence)) if sentence[i].is_noun()]
        #print(index_noun_list)
        if len(index_noun_list) < 2: continue


        for i,index_x in enumerate(index_noun_list[:-1]):
            for index_y in index_noun_list[i+1:]:
                flag=False
                number=-1
                routes_x=set()
                #print(i,index_x,index_y)
                dst= sentence[index_x].dst
                while dst != -1:
                    if dst == index_y:
                        flag=True
                        break
                    routes_x.add(dst)
                    dst= sentence[dst].dst
                
                if not flag:
                    dst = sentence[index_y].dst
                    while dst != -1:
                        if dst in routes_x:
                            number=dst
                            break
                        else:
                            dst = sentence[dst].dst

                if number == -1:
                    s=sentence[index_x].get_phrase_with_mask('X')
                    dst = sentence[index_x].dst
                    while dst != -1:
                        if dst == index_y:
                            s+=' -> '+sentence[dst].get_phrase_with_mask('Y',True)
                            break
                        else:
                            s+=' -> '+sentence[dst].get_phrase()
                        dst=sentence[dst].dst
                    
                    roots_list.append(s)
                else:
                    s = sentence[index_x].get_phrase_with_mask('X')
                    dst = sentence[index_x].dst
                    while dst != number:
                        s += ' -> ' + sentence[dst].get_phrase()
                        dst = sentence[dst].dst
                    s += ' | '

                    s += sentence[index_y].get_phrase_with_mask('Y')
                    dst = sentence[index_y].dst
                    while dst != number:
                        s += ' -> '+sentence[dst].get_phrase()
                        dst = sentence[dst].dst
                    s += ' | '

                    s += sentence[number].get_phrase()
                    roots_list.append(s)

                    


                
        


    return roots_list


    

fname='ai.ja.txt.parsed'
article=get_chunk_list(fname)


answer_list=[]
s=article[1]
roots_list=extract_depency_path(s)
answer_list.extend(roots_list)



s='\n'.join(answer_list)
with open('ans49.txt','w') as f:
    f.write(s)

print(s)

'''
[プログラムの結果](長いので一部省略)
ans49.txtファイルに結果を保存
'''