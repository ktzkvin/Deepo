"""
Model definitions and text processing utilities for the Seq2Seq LSTM.
Includes Bahdanau Attention mechanism and bidirectional encoding.
"""
import torch
import torch.nn as nn
import re

UNK = '<unk>'

class Attention(nn.Module):
    """
    Bahdanau Attention mechanism to compute attention weights 
    over encoder outputs based on the current decoder hidden state.
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v    = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        B, S, _ = encoder_outputs.shape
        h_top  = hidden[-1].unsqueeze(1).repeat(1, S, 1)
        energy = torch.tanh(self.attn(torch.cat([h_top, encoder_outputs], dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder for processing source sequences.
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_id, num_layers):
        super().__init__()
        self.hid_dim    = hid_dim
        self.num_layers = num_layers
        self.emb        = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn        = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers,
                                  bidirectional=True, batch_first=True)
        self.fc_h       = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c       = nn.Linear(hid_dim * 2, hid_dim)
        
    def forward(self, src):
        B = src.size(0)
        outputs, (h, c) = self.rnn(self.emb(src))
        h = h.view(self.num_layers, 2, B, self.hid_dim)
        c = c.view(self.num_layers, 2, B, self.hid_dim)
        h = torch.tanh(self.fc_h(torch.cat([h[:, 0], h[:, 1]], dim=2)))
        c = torch.tanh(self.fc_c(torch.cat([c[:, 0], c[:, 1]], dim=2)))
        return outputs, h, c

class Decoder(nn.Module):
    """
    LSTM Decoder with attention for generating target sequences.
    """
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, pad_id, num_layers):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attn = Attention(enc_hid_dim, dec_hid_dim)
        self.rnn  = nn.LSTM(emb_dim + enc_hid_dim * 2, dec_hid_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(dec_hid_dim, vocab_size)
        
    def forward(self, inp, h, c, encoder_outputs):
        embedded     = self.emb(inp).unsqueeze(1)
        attn_weights = self.attn(h, encoder_outputs).unsqueeze(1)
        context      = torch.bmm(attn_weights, encoder_outputs)
        rnn_input    = torch.cat([embedded, context], dim=2)
        out, (h, c)  = self.rnn(rnn_input, (h, c))
        return self.fc(out.squeeze(1)), h, c

def tokenize(s: str) -> list:
    """Tokenizes input string, handling CJK characters separately."""
    s = s.strip()
    if any('\u3000' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in s):
        return list(s.replace(' ', ''))
    return s.split()

def preprocess(sentence: str) -> str:
    """Normalizes spacing and removes specific punctuation."""
    sentence = re.sub(r"[?.!,;:]", " ", sentence)
    return re.sub(r" +", " ", sentence).strip()

def postprocess(tokens: list) -> str:
    """Reconstructs a clean string from a list of tokens."""
    tokens = [t for t in tokens if t != UNK]
    text = ' '.join(tokens)
    text = re.sub(r" ' ", "'", text)
    text = re.sub(r" ([.!?,;:])", r"\1", text)
    text = re.sub(r'(?<=[\u4e00-\u9fff\u3040-\u30ff]) (?=[\u4e00-\u9fff\u3040-\u30ff])', '', text)
    return text