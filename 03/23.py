import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']

def is_section(string):
    return re.match(r'^(={2,})\s*(.+?)\s*\1$',string)

fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス').split('\n')

for line in text:
    if is_section(line):
        ctg = is_section(line).groups()
        num=len(ctg[0])-1
        s='セクション名:'+ctg[1]+',レベル:'+str(num)
        print(s)

"""
[プログラムの結果]
 % python 23.py
セクション名:国名,レベル:1
セクション名:歴史,レベル:1
セクション名:地理,レベル:1
セクション名:主要都市,レベル:2
セクション名:気候,レベル:2
(途中省略)
セクション名:野球,レベル:3
セクション名:カーリング,レベル:3
セクション名:自転車競技,レベル:3
セクション名:脚注,レベル:1
セクション名:関連項目,レベル:1
セクション名:外部リンク,レベル:1

"""
