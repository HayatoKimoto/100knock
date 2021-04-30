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
    line = re.sub(r'(\'{2,5})(.+?)(\1)', r'\2', line)
    if re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$',line):
        inf = re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$', line)
        d[inf.group(1)]=inf.group(2)
    else:
        d[inf.group(1)] += "\n" + line
        
for i,j in d.items():
    print(i,j)

"""
[プログラムの結果](強調マークアップの一部分だけ抜粋)
変化前:
確立形態4 現在の国号「'''グレートブリテン及び北アイルランド連合王国'''」に変更
変化後:
確立形態4 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更


変化前:
国歌 [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}
変化後:
国歌 [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />神よ女王を護り賜え<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}

"""