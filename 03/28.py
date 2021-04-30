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
    #26 強調マークアップの除去
    line = re.sub(r'(\'{2,5})(.+?)(\1)', r'\2', line)

    #27 内部リンクの除去
    line = re.sub(r'\[\[([^|]*?)\|?([^|]*?)\]\]', r'\2', line)
    
    #28 MediaWikiマークアップの除去
    #28-1 ファイルの除去
    line = re.sub(r'\[\[ファイル:([^|]*)\|?([^|]*)\|?([^|]*)\]\]',r'\3',line)
    #28-2 外部リンクの除去
    line = re.sub(r'\[http:.+\]','',line)
    #28-3 言語タグの除去
    line = re.sub(r'\{\{lang\|[^|]*?\|([^|]*?)\}\}',r'\1',line)
    #28-4 htmlタグの除去
    line = re.sub(r'<\/?(ref|br)[^>]*?>','',line)
    #28-5 箇条書き
    line = re.sub(r'^(\*{1,2})(.+?)', r'\2', line)
    #28-6 仮リンク
    line = re.sub(r'\{\{仮リンク\|([^|]*?)\|(.*?)\}\}',r'\1',line)
    #28-7 その他
    line = re.sub(r'\{\{(.*?)\}\}','',line)

    if re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$',line):
        inf = re.match(r'^\|\s?(.+?)\s?=\s?(.+?)$', line)
        d[inf.group(1)] = inf.group(2)
    else:
        d[inf.group(1)] += "\n" + line
        
for i,j in d.items():
    print(i,j)

"""
[プログラムの結果]
#28-1 ファイルの除去
変化前:
国章画像 [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
変化後:
国章画像 イギリスの国章

#28-2 外部リンクの除去
変化前:
GDP値元 1兆5478億[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]

変化後:
GDP値元 1兆5478億

#28-3 言語タグ
変化前:
*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]}
変化後:
Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon（ウェールズ語）

#28-4 htmlタグの除去
変化前:
確立形態3 グレートブリテン及びアイルランド連合王国成立<br />（1800年合同法）
変化後:
確立形態3 グレートブリテン及びアイルランド連合王国成立（1800年合同法）

#28-5 箇条書き
変化前:
*{{lang|sco|Unitit Kinrick o Great Breetain an Northren Ireland}}（[[スコットランド語]]）
変化後:
Unitit Kinrick o Great Breetain an Northren Ireland（スコットランド語）

#28-6 仮リンク
変化前:
他元首等氏名2 {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
変化後:
他元首等氏名2 リンゼイ・ホイル

#28-7 その他
変化前:
人口値 6643万5600{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}
変化後:
人口値 6643万5600

"""