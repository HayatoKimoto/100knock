import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']

def isSection(string):
    return re.match(r'^(={2,}) *(.+?) *\1$',string)

fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス').split('\n')

for line in text:
    if isSection(line):
        #print(isSection(line).groups())
        ctg = isSection(line).groups()
        num=len(ctg[0])-1
        s='セクション名:'+ctg[1]+',レベル:'+str(num)
        print(s)