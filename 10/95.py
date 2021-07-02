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