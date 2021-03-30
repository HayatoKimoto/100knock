import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']

def isTemplete(string):
    pattern=re.compile(r'^\{\{基礎情報.*?\n(.*?)^\}\}$',re.MULTILINE+re.VERBOSE+re.DOTALL)
    return re.search(pattern,string)


fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス')

d = dict()
s = isTemplete(text).group(1).split('\n')
for line in s:
    #26 強調マークアップの除去
    line = re.sub(r'(\'{2,5})(.+?)(\1)', r'\2', line, flags=(re.MULTILINE | re.VERBOSE))

    #27 内部リンクの除去
    #print(re.search(r'\[\[([^|]*?)\|*([^|]*?)\]\]', line,flags=(re.MULTILINE | re.VERBOSE)))
    line = re.sub(r'\[\[([^|]*?)\|*([^|]*?)\]\]', r'\2', line, flags=(re.MULTILINE | re.VERBOSE))
    
    #28 MediaWikiマークアップの除去
    #28-1 ファイルの除去
    print(re.search(r'\[\[([^|]*?)\|+([^|]*?)\|+(.*?)\]\]', line,flags=(re.MULTILINE | re.VERBOSE)))
    line=re.sub(r'\[\[ファイル:.*\]\]','',line,flags=(re.MULTILINE | re.VERBOSE))
    #28-2 外部リンクの除去
    #28-3 言語タグの除去
    #28-4 htmlタグの除去
    if (re.match(r'^\|\s*(.+?)=\s*(.+?)$',line)):
        inf = re.match(r'^\|\s*(.+?)\s*=\s*(.+?)$', line)
        d[inf.group(1)]=inf.group(2)
    else:
        d[inf.group(1)] += "\n" + line
        
#for i,j in d.items():
#    print(i,j)

