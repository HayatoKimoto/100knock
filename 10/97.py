import optuna
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
import torch.tensor as Tensor
from torch.nn.init import xavier_uniform_
import math
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from matplotlib import pyplot as plt
from tqdm import tqdm
import sentencepiece as spm


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
        
def evaluation(valid_loader, model):
    model.eval()

    len_loader = len(valid_loader)
    total_loss = 0

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


        
    return total_loss / len_loader

pad_id = 3
bos_id = 1
eos_id = 2

batch_size=512
max_len=42
V_ja = 16000
V_en = 16000

EPOCH_NUM = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
best_valid_loss, best_epoch, best_batch = 1e7, 0, 0

def objective(trial):
    global best_valid_loss, best_epoch, best_batch
    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # optimizer_name
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #固定パラメータの設定


    #データの読み込み
    X_train = word2id_tensor('sentencepiece-kftt-ja.model','kftt-data-1.0/data/tok/kyoto-train.cln.ja')
    Y_train = word2id_tensor('sentencepiece-kftt-en.model','kftt-data-1.0/data/tok/kyoto-train.cln.en')

    X_valid = word2id_tensor('sentencepiece-kftt-ja.model','kftt-data-1.0/data/tok/kyoto-dev.ja')
    Y_valid = word2id_tensor('sentencepiece-kftt-en.model','kftt-data-1.0/data/tok/kyoto-dev.en')

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(X_valid, Y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    model = MyTransformer(device=device, max_len=max_len, source_vocab_length=V_ja, target_vocab_length=V_en).to(device)
    
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)

    
    for epoch in range(1, EPOCH_NUM):
        model.train()
        total_loss=0
        for i, batch in tqdm(enumerate(train_loader, 1)):
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
            optimizer.step()
            total_loss += loss.item() / batch_size


            if i % 50 == 0:
                train_loss = total_loss / i
                print(f"Epoch [{epoch}] Batch[{i}] complete. train_loss = {train_loss}")
            
            
            if i % 50 == 0:
                valid_loss= evaluation(valid_loader, model)
                print(f"Epoch [{epoch}] Batch[{i}] complete. valid_loss = {valid_loss}")

                if valid_loss < best_valid_loss:
                    best_valid_loss, best_epoch, best_batch = valid_loss, epoch, i
            
        if epoch>5: return best_valid_loss


study = optuna.create_study()
study.optimize(objective, timeout=3600)

"""
[プログラムの結果]
'learning_rate': 4.543832189446735e-05, 'optimizer': 'AdamW'
"""