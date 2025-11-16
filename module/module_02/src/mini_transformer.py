#!/usr/bin/env python3
# encoding=utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.nn import GELU

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  # Apple GPU acceleration interface, Metal Performance Shaders
else:
    device = "cpu"

"""
'Mini Transformer' Model Structure Description on each Component:
Encoder:
    [ENC-1] Embedding
    [ENC-2] Positional Encoding (currently using "absolute sine"; relative/rotary in attention changes Q/K)
    [ENC-3] Multi-Head Attention (Q/K/V mapping + attention matrix)
    [ENC-4] FFN (two-layer non-linearity; equivalent form implemented with 1x1 Conv)
    [ENC-5] Residual + LayerNorm
Decoder:
    [DEC-1] Masked Self-Attention (PAD + Subsequent upper triangular)
    [DEC-2] Cross-Attention (Q comes from decoder, K/V comes from encoder)
"""

# ---------- model hyperparameters ----------
d_k = 64            # Q/K dimension per head
d_v = 64            # V dimension per head
d_embedding = 512   # Embedding dimension

n_heads = 8         # Number of attention heads
n_layers = 6        # Number of encoder/decoder layers

# Ensure dimensions match, if not, raise an error, to ensure correctness
assert d_k * n_heads == d_embedding and d_v * n_heads == d_embedding
# -------------------------------------------


# ---------- model components ----------
# [ENC-2-ABS] Absolute Position Encoding (Sine) Table Generation
def get_sin_enc_table(n_position, embedding_dim):
    sinusoid_table = np.zeros((n_position, embedding_dim))
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


# [ATTN-core] Scaled Dot-Product Attention (used inside Multi-Head)
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    Q: Queries tensor of shape (batch_size, n_heads, len_q, d_k)
    K: Keys tensor of shape (batch_size, n_heads, len_k, d_k)
    V: Values tensor of shape (batch_size, n_heads, len_k, d_v)
    attn_mask: Attention mask tensor of shape (batch_size, 1, len_q, len_k)
    Returns:
        context: Output tensor after attention of shape (batch_size, n_heads, len_q, d_v)
        attn: Attention weights tensor of shape (batch_size, n_heads, len_q, len_k)
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, attn_mask):
        """
        Attention calculation process
            - Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
            - if attn_mask is given:
                - Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k) + attn_mask) · V
        1. Calculate attention scores: Q · K^T / sqrt(d_k)
        2. Apply attention mask: set masked positions to -inf
        3. Calculate attention weights: softmax(Attention scores)
        4. Calculate context vector: Attention weights · V
        5. Return context and attention weights
        """
        d_k = Q.size(-1) # The last dimension size of Q, i.e., d_k
        # Calculate the attention scores
        # - Attention scores = Q · K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5)
        # Apply the attention mask
        # - set masked positions to -inf, so that after softmax they become 0
        # - should not set to 0 directly, as that would affect the softmax distribution, exp(0)=1
        scores.masked_fill_(attn_mask, float("-inf"))
        # Calculate the attention weights
        # - Attention weights = softmax(Attention scores)
        # - 'dim=-1' means softmax is applied on the last dimension (len_k)
        attn = torch.softmax(scores, dim=-1)
        # Calculate the context vector
        # - Context = Attention weights · V
        context = torch.matmul(attn, V)
        return context, attn


# [ENC-3]/[DEC-1]/[DEC-2] Multi-Head Attention（includes Q/K/V mapping, attention matrix）
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)  # The Q mapping for Multi-Attn
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)  # The K mapping for Multi-Attn
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)  # The V mapping for Multi-Attn
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)       # [ENC-5] Residual+LayerNorm / the same in [DEC-1/2] 
        self.attn = ScaledDotProductAttention()

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
        context, weights = self.attn(q_s, k_s, v_s, attn_mask)  # Scaled Dot-Product Attention
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        output = self.layer_norm(output + residual)  # Residual + LayerNorm
        return output, weights


# PAD(Padding) Masked Code（be used in both self-attention and cross-attention）
def get_attn_pad_mask(seq_q, seq_k):
    B, Lq = seq_q.size()
    _, Lk = seq_k.size()
    pad_attn_mask = seq_k.eq(0).unsqueeze(1).expand(B, Lq, Lk)  # True means to mask
    return pad_attn_mask


# [ENC-4] Feed Forward Network (FFN + Activation); also used in Decoder layers
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # Using Conv1d to implement FFN
        # - First Conv1d layer expands dimension from d_embedding to 2048
        # - Second Conv1d layer reduces dimension back to d_embedding
        # - Kernel size is 1, equivalent to position-wise fully connected layer
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)  # [ENC-5] Residual + LayerNorm
        self.gelu = GELU()  # Using GELU activation function (Research has shown GELU performs better than ReLU in Transformers)
        
    def forward(self, inputs):
        residual = inputs
        # Option 1: Using ReLU activation
        # output = torch.relu(self.conv1(inputs.transpose(1, 2)))  # Activation after first conv
        # Option 2: Using GELU activation (Research has shown GELU performs better than ReLU in Transformers)
        output = self.gelu(self.conv1(inputs.transpose(1, 2)))  # Activation after first conv
        output = self.conv2(output).transpose(1, 2)      
        output = self.layer_norm(output + residual)              # Residual + LayerNorm
        return output
# -------------------------------------------


# ----------------- Encoder -----------------
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # [ENC-3] Self-Attn
        self.pos_ffn = PoswiseFeedForwardNet()     # [ENC-4] FFN
        
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn_weights


class Encoder(nn.Module):
    def __init__(self, corpus, max_len=4096):
        super().__init__()
        self.src_emb = nn.Embedding(corpus.src_vocab, d_embedding, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(max_len, d_embedding), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
    def forward(self, enc_inputs):
        B, L = enc_inputs.size()
        pos_idx = torch.arange(L, device=enc_inputs.device).unsqueeze(0)  # 通常从0开始
        x = self.src_emb(enc_inputs) + self.pos_emb(pos_idx)
        mask = get_attn_pad_mask(enc_inputs, enc_inputs).to(enc_inputs.device)
        attn_ws = []
        for layer in self.layers:
            x, w = layer(x, mask)
            attn_ws.append(w)
        return x, attn_ws
# -------------------------------------------


# ----------------- Decoder -----------------
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # [DEC-1] Masked Self-Attn
        self.dec_enc_attn = MultiHeadAttention()   # [DEC-2] Cross-Attn
        self.pos_ffn = PoswiseFeedForwardNet()     # FFN + Residual + LayerNorm
        
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)  # [DEC-1]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)  # [DEC-2]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# The mask to prevent attention to subsequent positions (The upper triangular part is True)
def get_attn_subsequent_mask(seq):
    B, L = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=seq.device), diagonal=1)
    return subsequent_mask.unsqueeze(0).expand(B, L, L)


class Decoder(nn.Module):
    def __init__(self, corpus, max_len=4096):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(corpus.tgt_vocab, d_embedding, padding_idx=0)  # Decoder Embedding
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(max_len, d_embedding), freeze=True)  # Positional Encoding
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        B, L = dec_inputs.size()
        pos_indices = torch.arange(L, device=dec_inputs.device).unsqueeze(0)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)                   # PAD
        subsequent = get_attn_subsequent_mask(dec_inputs)                      # [DEC-1] Mask
        dec_self_attn_mask = (pad_mask.to(dec_inputs.device) | subsequent)     # 合并成最终 masked self-attn
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs).to(dec_inputs.device)  # 给 cross-attn 用
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
# -------------------------------------------


# ------------ Transformer Model ------------
class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)
        self.decoder = Decoder(corpus)
        self.projection = nn.Linear(d_embedding, corpus.tgt_vocab, bias=False)  # 映射到词表
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # [batch, tgt_len, tgt_vocab]
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
# -------------------------------------------


if __name__ == "__main__":
    print("Select device: " + device)

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    class Corpus:
        src_vocab, tgt_vocab = 1000, 1200
        src_len,  tgt_len  = 20, 22
    model = Transformer(Corpus()).to(device)
    src = torch.randint(1, Corpus.src_vocab, (2, Corpus.src_len)).to(device)
    tgt = torch.randint(1, Corpus.tgt_vocab, (2, Corpus.tgt_len)).to(device)
    print(model(src, tgt)[0].shape)  # Expected Output: [2, 22, 1200]
