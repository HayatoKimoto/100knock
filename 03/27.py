import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']

def is_templete(string):
    pattern=re.compile(r'^\{\{基礎情報.*?\n(.*?)^\}\}$',re.MULTILINE+re.DOTALL)
    return re.search(pattern,string)


fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス')

d = dict()
s = is_templete(text).group(1).split('\n')
for line in s:
    #26 強調マークアップの除去
    line = re.sub(r'(\'{2,5})(.+?)(\1)', r'\2', line)
    #27 内部リンクの除去
    line = re.sub(r'\[\[([^|]*?)\|?([^|]*?)\]\]', r'\2', line)   
    
    if (re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$',line)):
        inf = re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$', line)
        d[inf.group(1)]=inf.group(2)
    else:
        d[inf.group(1)] += "\n" + line
        
for i,j in d.items():
    print(i,j)

"""
[プログラムの結果](内部リンクの一部分だけ抜粋)
変化前:
国章リンク （[[イギリスの国章|国章]]）
変化後:
国章リンク （国章


変化前:
元首等肩書 [[イギリスの君主|女王]]
変化後:
元首等肩書 女王

"""