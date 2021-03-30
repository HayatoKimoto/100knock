import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']


def isCategory(string):
    # カテゴリー行を正規表現で判定.
    return re.match(r'^\[\[Category:.+\]\]$', string)

#rはraw文字列でエスケープシーケンス(\nや\t)を無視できる.
#\[\[の部分はメタ文字である[を\でエスケープシーケンスしている.
#^は先頭から一致しているかの判定.
fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス').split('\n')

# カテゴリの行を表示する
for line in text:
    if isCategory(line):
        print(line)
