from accelerate import Accelerator, ProfileKwargs
from io import open
import unicodedata
import re
import random
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import math
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

SOS_token = 0
EOS_token = 1

# ---------- Perfilado ----------
def trace_handler(p):
    print("\n--- GPU Profiling ---")
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print("\n--- CPU Profiling ---")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    os.makedirs("/tmp/accelerate_traces", exist_ok=True)
    p.export_chrome_trace(f"/tmp/accelerate_traces/trace_{p.step_num}.json")

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    schedule_option={"wait": 5, "warmup": 1, "active": 3, "repeat": 2, "skip_first": 1},
    on_trace_ready=trace_handler
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
device = accelerator.device

# ---------- Procesamiento del lenguaje ----------
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    lines = open(f'data/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split("CC-BY")[0].split('\t')[:-1]] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        return Lang(lang2), Lang(lang1), pairs
    return Lang(lang1), Lang(lang2), pairs

MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

# ---------- Redes ----------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        return self.gru(embedded)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs, attentions = [], []
        for i in range(MAX_LENGTH):
            output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(output)
            attentions.append(attn)
            decoder_input = target_tensor[:, i].unsqueeze(1) if target_tensor is not None else output.topk(1)[1].squeeze(-1).detach()
        return F.log_softmax(torch.cat(decoder_outputs, dim=1), dim=-1), decoder_hidden, torch.cat(attentions, dim=1)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        context, attn_weights = self.attention(hidden.permute(1, 0, 2), encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        return self.out(output), hidden, attn_weights

# ---------- Dataset ----------
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def get_dataloader(batch_size):
    global input_lang, output_lang, pairs
    input_lang, output_lang, pairs = prepareData('eng', 'spa', True)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp) + [EOS_token]
        tgt_ids = indexesFromSentence(output_lang, tgt) + [EOS_token]
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
    data = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))
    return DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)

# ---------- Entrenamiento ----------
def train_epoch(dataloader, encoder, decoder, encoder_opt, decoder_opt, criterion):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        enc_out, enc_hidden = encoder(input_tensor)
        dec_out, _, _ = decoder(enc_out, enc_hidden, target_tensor)
        loss = criterion(dec_out.view(-1, dec_out.size(-1)), target_tensor.view(-1))
        accelerator.backward(loss)
        encoder_opt.step()
        decoder_opt.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train(train_loader, encoder, decoder, n_epochs, lr=0.001):
    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    encoder, decoder, encoder_opt, decoder_opt, train_loader, criterion = accelerator.prepare(
        encoder, decoder, encoder_opt, decoder_opt, train_loader, criterion)
    for epoch in range(1, n_epochs + 1):
        with accelerator.profile():
            loss = train_epoch(train_loader, encoder, decoder, encoder_opt, decoder_opt, criterion)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---------- Evaluación ----------
def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = torch.tensor(indexesFromSentence(input_lang, sentence) + [EOS_token], dtype=torch.long).unsqueeze(0).to(device)
        enc_out, enc_hidden = encoder(input_tensor)
        dec_out, _, _ = decoder(enc_out, enc_hidden)
        topi = dec_out.topk(1)[1].squeeze()
        decoded = [output_lang.index2word[idx.item()] for idx in topi if idx.item() != EOS_token]
        return decoded

def evaluateRandomly(encoder, decoder, n=5):
    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output = evaluate(encoder, decoder, pair[0])
        print('<', ' '.join(output), '\n')

# ---------- Ejecutar ----------
start_time = time.time()
hidden_size = 128
batch_size = 32
n_epochs = 2
train_loader = get_dataloader(batch_size)
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
train(train_loader, encoder, decoder, n_epochs)
print(f"Training time: {time.time() - start_time:.2f} s")
evaluateRandomly(encoder, decoder)
