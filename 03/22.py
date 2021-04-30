import json
import gzip
import re

def find_title(fname, title):
    with gzip.open(fname) as f:
        for line in f:
            data = json.loads(line)
            if data["title"] != title: continue
            return data['text']


def is_category(string):
    # カテゴリー行を正規表現で判定.
    # 1つめのグループにはカテゴリ名が入っている
    # (\|?)\*?の部分は[[Category:(カテゴリ名)|*]]の|*の部分である.
    return re.match(r'^\[\[Category:([^|]*?)(\|?)\*?\]\]', string)


fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス').split('\n')

# カテゴリの行を表示する
for line in text:
    if is_category(line):
        print(is_category(line).group(1))

"""
[プログラムの結果]
% python 22.py 
イギリス
イギリス連邦加盟国
英連邦王国
G8加盟国
海洋国家
現存する君主国
島国
1801年に成立した国家・領域

"""