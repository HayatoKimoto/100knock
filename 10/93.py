import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from word2id import get_id_ja, get_vocab_ja
from word2id import get_id_en, get_vocab_en
import torch.tensor as Tensor
from torch.nn.init import xavier_uniform_
import math
from tqdm import tqdm
import MeCab
import pickle
import nltk

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=42):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class MyTransformer(nn.Module):
    # def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
    #              num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
    #              activation: str = "relu",source_vocab_length: int = 60000,target_vocab_length: int = 60000) -> None:
    def __init__(self, device, max_len, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", source_vocab_length: int = 60000, target_vocab_length: int = 60000) -> None:
        super(MyTransformer, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_length, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.target_embedding = nn.Embedding(target_vocab_length, d_model, padding_idx=pad_id)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.out = nn.Linear(d_model, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.device = device


    # def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    def forward(self, src, tgt, src_mask = None, tgt_mask = None,
                memory_mask = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None) -> Tensor:


        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src_key_padding_mask != None:
            src_key_padding_mask = src_key_padding_mask.transpose(0,1)
        if tgt_key_padding_mask != None:
            tgt_key_padding_mask = tgt_key_padding_mask.transpose(0,1)

        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)

        # ?????????????????????????????????????????????????????????mask
        size = tgt.size(0)
        #tgt_mask = ~ torch.triu(torch.ones(size, size)==1).transpose(0,1)
        tgt_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(self.device)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.out(output)
        return output


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

def greeedy_decode_sentence(model,sentence):
    model.eval()
    indexed = []
    mecab = MeCab.Tagger ("-Owakati")
    text = mecab.parse(sentence)
    text=text.split(' ')
    text.insert(0,'<s>')
    text.append('</s>')
    with open('word2id_ja','rb') as f:
        word2id_dict=pickle.load(f)

    with open('id2word_en','rb') as f:
        id2word_dict=pickle.load(f)

    for tok in text:
        if tok in word2id_dict:
            indexed.append(word2id_dict[tok])
        else:
            indexed.append(3)

    sentence=torch.tensor([indexed[:42]]).to(device)
    trg_init_tok = 1
    trg = torch.LongTensor([[trg_init_tok]]).to(device)
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.to(device)
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = id2word_dict[pred.argmax(dim=2)[-1].item()]
        translated_sentence+=" "+add_word
        if add_word=='</s>':
            break
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).to(device)))
        #print(trg)
    return translated_sentence     

batch_size=16
max_len=42

pad_id = 0
bos_id = 1
eos_id = 2

with open('kftt-data-1.0/data/orig/kyoto-dev.ja') as f:
    X_valid=f.readlines()

with open('kftt-data-1.0/data/tok/kyoto-dev.en') as f:
    Y_valid=f.readlines()


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

V_ja = get_vocab_ja()
V_en = get_vocab_en()

model = MyTransformer(device=device, max_len=max_len, source_vocab_length=V_ja, target_vocab_length=V_en).to(device)
model.load_state_dict(torch.load('best_model.pt'))




total_score=0
for xx,yy in zip(X_valid,Y_valid):
    references=[yy.split(' ')]
    hypothesis= greeedy_decode_sentence(model,xx).split(' ')
    total_score+=nltk.translate.bleu_score.sentence_bleu(references,hypothesis)
    

BLEUscore=total_score/len(X_valid)
print(BLEUscore)

"""
[????????????????????????]
0.00966520457079559
"""