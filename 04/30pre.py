import MeCab

fname='neko.txt'
fname_parsed='neko.txt.mecab'

with open(fname) as data_file,\
    open(fname_parsed,mode='w') as out_file:

    mecab=MeCab.Tagger()
    out_file.write(mecab.parse(data_file.read()))

