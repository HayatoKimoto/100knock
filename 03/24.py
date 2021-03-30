import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']

def isFile(string):
    return re.match(r'^\[\[ファイル:(.+?)\|', string)

fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス').split('\n')

for line in text:
    if isFile(line):
        print(isFile(line).group(1))
        
