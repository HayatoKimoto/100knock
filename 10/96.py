import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch.tensor as Tensor
from torch.nn.init import xavier_uniform_
import math
from tqdm import tqdm
import MeCab
import pickle
import sentencepiece as spm
from torch.utils.tensorboard import SummaryWriter
import sacrebleu


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
    def __init__(self, device, max_len, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", source_vocab_length: int = 16000, target_vocab_length: int = 16000) -> None:
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

        # 未来の情報を参照しないようにするためのmask
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


def word2id_tensor(model,data):
    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    with open(data) as f:
        lines=f.readlines()

    list_ja = []
    for line in lines:
        ids=sp.EncodeAsIds(line)
        ids.insert(0,1)
        ids.append(2)
        list_ja.append(torch.tensor(ids[:max_len]))

    packed_inputs= pack_sequence(list_ja,enforce_sorted=False)
    padded_packed_inputs,_ = pad_packed_sequence(packed_inputs, batch_first=True,total_length=max_len)

    return padded_packed_inputs
        

def greeedy_decode_sentence(model,sentence):
    model.eval()
    mecab = MeCab.Tagger ("-Owakati")
    #日本語
    sp_ja = spm.SentencePieceProcessor()
    sp_ja.Load("sentencepiece-kftt-16000-ja.model")
    #英語
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load("sentencepiece-kftt-16000-en.model")
    
    text = mecab.parse(sentence)
    ids=sp_ja.EncodeAsIds(text)
    ids=ids[:40]
    ids.insert(0,bos_id)
    ids.append(eos_id)
    sentence=torch.tensor([ids]).to(device)
    trg_init_tok = bos_id
    trg = torch.LongTensor([[trg_init_tok]]).to(device)
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.to(device)
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = sp_en.DecodeIds([pred.argmax(dim=2)[-1].item()])
        if add_word=='</s>':
            break
        translated_sentence+=" "+add_word
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).to(device)))
        #print(trg)
    return translated_sentence
        



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

pad_id = 3
bos_id = 1
eos_id = 2

batch_size=512
max_len=42

#データの読み込み
X_train = word2id_tensor('sentencepiece-kftt-16000-ja.model','kftt-data-1.0/data/tok/kyoto-train.cln.ja')
Y_train = word2id_tensor('sentencepiece-kftt-16000-en.model','kftt-data-1.0/data/tok/kyoto-train.cln.en')

X_valid = word2id_tensor('sentencepiece-kftt-16000-ja.model','kftt-data-1.0/data/tok/kyoto-dev.ja')
Y_valid = word2id_tensor('sentencepiece-kftt-16000-en.model','kftt-data-1.0/data/tok/kyoto-dev.en')

print(X_train,X_train.size())
print(Y_train,Y_train.size())


print(X_valid,X_valid.size())
print(Y_valid,Y_valid.size())



train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(X_valid, Y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

V_ja = 16000
V_en = 16000
model = MyTransformer(device=device, max_len=max_len, source_vocab_length=V_ja, target_vocab_length=V_en).to(device)


optim = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

writer = SummaryWriter(log_dir="./logs")


def evaluation(valid_loader, model):
    model.eval()

    len_loader = len(valid_loader)
    total_loss = 0
    total_score = 0
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load("sentencepiece-kftt-16000-en.model")

    sp_ja = spm.SentencePieceProcessor()
    sp_ja.Load("sentencepiece-kftt-16000-ja.model")

    with torch.no_grad():
        for i, batch in enumerate(valid_loader, 1):
            src = batch[0].to(device, non_blocking=True)
            tgt = batch[1].to(device, non_blocking=True)
            
            tgt_input = tgt[:, :-1]
            targets = tgt[:, 1:].contiguous().view(-1)

            # paddingを無視するためのmask
            src_padding_mask = (src == pad_id).transpose(0,1).to(device, non_blocking=True)
            tgt_padding_mask = (tgt_input == pad_id).transpose(0,1).to(device, non_blocking=True)

            preds = model(src.transpose(0,1), tgt_input.transpose(0,1), src_key_padding_mask = src_padding_mask, tgt_key_padding_mask = tgt_padding_mask)
            preds_logits = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))         
            loss = criterion(preds_logits, targets)
            total_loss += loss.item() / batch_size
            
            for s,t in zip(src,tgt):
                s=s.cpu().tolist()
                t=t.cpu().tolist()
                references=[]
                references.append(sp_en.DecodeIds(t))
                xx = sp_ja.DecodeIds(s)
                xx = xx.split()
                xx = sp_ja.DecodePieces(xx)
                hypothesis= greeedy_decode_sentence(model,xx)
                bleu=sacrebleu.corpus_bleu(references,hypothesis)
                total_score+=bleu.score

        


    

        
    return total_loss / len_loader , total_score / len(valid_dataset)

def train(train_loader, valid_loader, model, optim, epoch):
    model.train()

    len_loader = len(train_loader)
    total_loss = 0

    global best_valid_loss, best_epoch, best_batch

    for i, batch in enumerate(train_loader, 1):
        src = batch[0].to(device, non_blocking=True)
        tgt = batch[1].to(device, non_blocking=True)
        tgt_input = tgt[:, :-1]
        targets = tgt[:, 1:].contiguous().view(-1)

        # paddingを無視するためのmask
        src_padding_mask = (src == pad_id).transpose(0,1).to(device, non_blocking=True)
        tgt_padding_mask = (tgt_input == pad_id).transpose(0,1).to(device, non_blocking=True)


        for param in model.parameters():
            param.grad = None
        preds = model(src.transpose(0,1), tgt_input.transpose(0,1), src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))

        loss = criterion(preds, targets)

        del src, tgt, tgt_input, targets, src_padding_mask, tgt_padding_mask, preds
        torch.cuda.empty_cache()

        loss.backward()
        optim.step()
        total_loss += loss.item() / batch_size


        if i % 10 == 0:
            train_loss = total_loss / i
            print(f"Epoch [{epoch[0]}/{epoch[1]}] Batch[{i}/{len_loader}] complete. train_loss = {train_loss}")
            writer.add_scalar("train_loss", train_loss, i)
        
        
        if i % 10 == 0:
            valid_loss,bleu_score= evaluation(valid_loader, model)
            print_list = [epoch[0], i, valid_loss]
            print(f"Epoch [{epoch[0]}/{epoch[1]}] Batch[{i}/{len_loader}] complete. valid_loss = {valid_loss}")
            writer.add_scalar("valid_loss", valid_loss, i)
            writer.add_scalar("valid_bleu_score", bleu_score, i)

            # bestモデル保存
            if valid_loss < best_valid_loss:
                best_valid_loss, best_epoch, best_batch = valid_loss, epoch[0], i
                torch.save(model.state_dict(), "best_model_96.pt")
                print("best models saved")
            print(f"Best Valid Loss = {best_valid_loss} (Epoch: {best_epoch}, Batch: {best_batch})\n\n")
            model.train()
        
        
    


EPOCH_NUM = 2
best_valid_loss, best_epoch, best_batch = 1e7, 0, 0

for epoch in range(1, EPOCH_NUM):
    train(train_loader, valid_loader, model, optim, [epoch, EPOCH_NUM]) 


# writerを閉じる
writer.close()