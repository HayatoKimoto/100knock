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
    #例）'{{基礎情報 国\n|略名 (省略)
    pattern=re.compile(r'^\{\{基礎情報.*?\n(.*?)^\}\}$',re.MULTILINE+re.DOTALL)
    return re.search(pattern,string)



fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス')

d = dict()

s = is_templete(text).group(1).split('\n')
for line in s:
    if re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$',line):
        inf = re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$', line)
        d[inf.group(1)]=inf.group(2)
    else:
        d[inf.group(1)]+="\n"+line


for i,j in d.items():
    print(i,j)

"""
[プログラムの結果](長いので省略)
%python 25.py
略名 イギリス
日本語国名 グレートブリテン及び北アイルランド連合王国
公式国名 {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）

"""