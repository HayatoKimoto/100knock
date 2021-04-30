import json
import gzip

with gzip.open('jawiki-country.json.gz') as f:
    for line in f:
        data = json.loads(line)
        if data["title"] != "イギリス": continue
        print(data)

"""
[プログラムの結果](長いので省略)
%python 20.py
{'title': 'イギリス', 'text': '{{redirect|UK}}\n{{redirect|英国|春秋時代の諸侯国|英 (春秋)}}\n(省略)
"""