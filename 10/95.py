import sentencepiece as spm


spm.SentencePieceTrainer.Train(
    '--input=kftt-data-1.0/data/tok/kyoto-train.cln.ja, --model_prefix=sentencepiece-kftt-16000-ja --model_type=word --character_coverage=0.9995 --vocab_size=16000 --pad_id=0 --unk_id=3'
)


spm.SentencePieceTrainer.Train(
    '--input=kftt-data-1.0/data/tok/kyoto-train.cln.en, --model_prefix=sentencepiece-kftt-16000-en --model_type=word --character_coverage=1.0 --vocab_size=16000 --pad_id=0 --unk_id=3'
)





#日本語
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece-kftt-16000-ja.model")

with open('kftt-data-1.0/data/tok/kyoto-train.cln.ja') as f:
    lines=f.readlines()

line=lines[0]
print(line)
#日本 の 水墨 画 を 一変 さ せ た 。
tokens=line.split()
print(tokens)
#['日本', 'の', '水墨', '画', 'を', '一変', 'さ', 'せ', 'た', '。']
print(sp.DecodePieces(tokens))
#日本の水墨画を一変させた。

ids=sp.EncodeAsIds(line)
print(ids)
#[42, 4, 6446, 510, 11, 56, 4341, 26, 117, 10, 5]
print(sp.DecodeIds(ids))
#日本 の 水墨 画 を 一変 さ せ た 。
print(sp.GetPieceSize())
#16000

print(sp.IdToPiece(10))
#1


#英語
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece-kftt-16000-en.model")

with open('kftt-data-1.0/data/tok/kyoto-train.cln.en') as f:
    lines=f.readlines()

line=lines[0]
print(line)
#He revolutionized the Japanese ink painting .

tokens=line.split()
print(tokens)
#['He', 'revolutionized', 'the', 'Japanese', 'ink', 'painting', '.']
print(sp.DecodePieces(tokens))
#HerevolutionizedtheJapaneseinkpainting.
ids=sp.EncodeAsIds(line)
print(ids)
#[31, 0, 4, 60, 1576, 540, 6]
print(sp.DecodeIds(ids))
#He ⁇  the Japanese ink painting .
print(sp.GetPieceSize())
#16000
print(sp.IdToPiece(10))
