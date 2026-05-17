import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    tgt_len = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
    ).bool()
    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, scale: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.scale     = scale

        self.W_q     = nn.Linear(d_model, d_model)
        self.W_k     = nn.Linear(d_model, d_model)
        self.W_v     = nn.Linear(d_model, d_model)
        self.W_o     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask, scale=self.scale)

        attn_output = (
            attn_output.transpose(1, 2)
                       .contiguous()
                       .view(batch_size, -1, self.d_model)
        )
        return self.W_o(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, scale: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale=scale)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, scale: bool = True) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, scale=scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, scale=scale)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, memory, memory, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        d_model     = layer.norm1.normalized_shape[0]
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        d_model     = layer.norm1.normalized_shape[0]
        self.norm   = nn.LayerNorm(d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int   = None,
        tgt_vocab_size: int   = None,
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        pad_idx:        int   = 1,
        pe_type:        str   = 'sinusoidal',
        scale:          bool  = True,
        src_vocab=None,
        tgt_vocab=None,
        checkpoint_path: str  = None,
    ) -> None:
        super().__init__()

        if checkpoint_path is not None:
            gdown.download(id="16-IFTBgSpWAnUetSSZgyhcfLdvihPzj8", output=checkpoint_path, quiet=False)
            ckpt           = torch.load(checkpoint_path, map_location='cpu')
            cfg            = ckpt['model_config']
            src_vocab_size = cfg['src_vocab_size']
            tgt_vocab_size = cfg['tgt_vocab_size']
            d_model        = cfg['d_model']
            N              = cfg['N']
            num_heads      = cfg['num_heads']
            d_ff           = cfg['d_ff']
            dropout        = cfg['dropout']
            pad_idx        = cfg['pad_idx']
            pe_type        = cfg.get('pe_type', 'sinusoidal')
            scale          = cfg.get('scale', True)
            src_vocab      = ckpt.get('src_vocab')
            tgt_vocab      = ckpt.get('tgt_vocab')

        self.d_model   = d_model
        self.pad_idx   = pad_idx
        self.pe_type   = pe_type
        self.scale     = scale
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self._spacy_de = None

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if pe_type == 'learned':
            self.pe = LearnedPositionalEncoding(d_model, dropout)
        else:
            self.pe = PositionalEncoding(d_model, dropout)

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout, scale=scale)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout, scale=scale)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        if checkpoint_path is not None:
            self.load_state_dict(ckpt['model_state_dict'])

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pe(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        if self._spacy_de is None:
            import spacy
            self._spacy_de = spacy.load('de_core_news_sm')

        self.eval()
        device = next(self.parameters()).device

        tokens  = [tok.text.lower() for tok in self._spacy_de.tokenizer(src_sentence)]
        sos_idx = self.src_vocab.stoi['<sos>']
        eos_idx = self.src_vocab.stoi['<eos>']
        indices = [sos_idx] + self.src_vocab.lookup_indices(tokens) + [eos_idx]

        src      = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, self.pad_idx)

        with torch.no_grad():
            memory  = self.encode(src, src_mask)
            tgt_sos = self.tgt_vocab.stoi['<sos>']
            tgt_eos = self.tgt_vocab.stoi['<eos>']
            ys      = torch.tensor([[tgt_sos]], dtype=torch.long).to(device)

            for _ in range(100):
                tgt_mask   = make_tgt_mask(ys, self.pad_idx)
                logits     = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys         = torch.cat([ys, next_token], dim=1)
                if next_token.item() == tgt_eos:
                    break

        generated = ys.squeeze(0).tolist()[1:]
        words = []
        for idx in generated:
            tok = self.tgt_vocab.lookup_token(idx)
            if tok == '<eos>':
                break
            if tok not in ('<sos>', '<pad>', '<unk>'):
                words.append(tok)

        return ' '.join(words)
