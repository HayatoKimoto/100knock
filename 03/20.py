import json
import gzip

with gzip.open('jawiki-country.json.gz') as f:
    for line in f:
        data = json.loads(line)
        if data["title"] != "イギリス": continue
        print(data)

