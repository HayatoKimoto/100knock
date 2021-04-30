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
    return re.match(r'^\[\[Category:.+\]\]$', string)


#rはraw文字列でエスケープシーケンス(\nや\t)を無視できる.
#[はメタ文字であるので\[で表す.
#^は先頭，$は末尾から一致しているかの判定.
fname = 'jawiki-country.json.gz'
text = find_title(fname, 'イギリス').split('\n')

# カテゴリの行を表示する
for line in text:
    if is_category(line):
        print(line)

"""
[プログラムの結果]
%python 21.py
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
"""