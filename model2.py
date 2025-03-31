from torch import nn
import torch


class PosEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        # [max_seq_len, 1]
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        # [d_model / 2]
        mul = 1./10_000 ** (torch.arange(0, d_model, 2) / d_model)
        pe = torch.ones(max_seq_len, d_model)
        # Shape: [max_seq_len, d_model/2]
        tmp = pos * mul
        pe[:, 0::2] = torch.sin(tmp)
        pe[:, 1::2] = torch.cos(tmp)
        # Shape: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MHA(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.qw = nn.Linear(d_model, d_model, bias=False)
        self.kw = nn.Linear(d_model, d_model, bias=False)
        self.vw = nn.Linear(d_model, d_model, bias=False)
        self.ow = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model
        self.scale = self.d_head ** -0.5
        
    def forward(self, q, k, v, mask=None):
        batch = k.size(0)
        ql, kl, vl = q.size(1), k.size(1), v.size(1)
        
        # Project and reshape to [batch, n_head, seq_len, d_head]
        q = self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)
        k = self.kw(k).reshape(batch, kl, self.n_head, self.d_head).transpose(1, 2)
        v = self.vw(v).reshape(batch, vl, self.n_head, self.d_head).transpose(1, 2)
        
        # Calculate attention scores
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # Expand mask to match attention score dimensions
            mask = mask.expand(batch, self.n_head, ql, kl)
            attn_score = attn_score.masked_fill(mask, float('-inf'))
            
        attn = torch.softmax(attn_score, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values and reshape back
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, ql, self.d_model)
        
        # Final projection
        out = self.ow(out)
        return out


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        ])
    
    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_head, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(2)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        mha_out = self.norms[0](x + self.mha(x, x, x, mask))
        ffn_out = self.norms[1](mha_out + self.ffn(mha_out))
        return self.dropout(ffn_out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MHA(d_model, n_head, dropout)
        self.cross_attn = MHA(d_model, n_head, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(3)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, context, tgt_causal_mask=None, tgt_pad_mask=None, context_pad_mask=None):
        att_out = self.self_attn(tgt, tgt, tgt, tgt_causal_mask)
        if tgt_pad_mask is not None:
            # small trick
            att_out = self.self_attn(att_out, att_out, att_out, tgt_pad_mask)
        self_attn_out = self.norms[0](tgt + att_out)
        cross_attn_out = self.norms[1](self_attn_out + self.cross_attn(
            self_attn_out, context, context, context_pad_mask))
        ffn_out = self.norms[2](cross_attn_out + self.ffn(cross_attn_out))
        return self.dropout(ffn_out)


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layer, dropout=0.1):
        super().__init__()
        self.encode_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for i in range(n_layer)
        ])
        
    def forward(self, x, mask=None):
        out = x
        for layer in self.encode_layers:
            out = layer(out, mask)
        return out


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layer, dropout=0.1):
        super().__init__()
        self.decode_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for i in range(n_layer)
        ])
        
    def forward(
        self, x, memory, tgt_causal_mask=None, tgt_pad_mask=None, memory_pad_mask=None):
        out = x
        for layer in self.decode_layers:
            out = layer(out, memory, tgt_causal_mask, tgt_pad_mask, memory_pad_mask)
        return x


import math


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_head, d_ff, n_layer, dropout=0.1, pad_idx=0, max_len=50_000):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos_encoding = PosEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(d_model, n_head, d_ff, n_layer, dropout)
        self.decoder = Decoder(d_model, n_head, d_ff, n_layer, dropout)
        self.ow = nn.Linear(d_model, tgt_vocab)
        self._init_parameters()
    
    def forward(self, src, tgt):
        src_pad_mask = self._pad_mask(src)
        tgt_pad_mask = self._pad_mask(tgt)
        tgt_causal_mask = self._causal_mask(tgt)
        
        src = self.src_emb(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        
        encoder_out = self.encoder(src, src_pad_mask)
        decoder_out = self.decoder(
            tgt, encoder_out, tgt_causal_mask, tgt_pad_mask, src_pad_mask)
        return self.ow(decoder_out)
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _causal_mask(self, x):
        # diagonal = 1, ignore it
        # [[0, 1, 1, 1],    # Can attend to all tokens except itself
        #  [0, 0, 1, 1],    # Can attend to next 2 tokens
        #  [0, 0, 0, 1],    # Can attend to next token
        #  [0, 0, 0, 0]]    # Cannot attend to any tokens
        seq = x.size(1)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        return mask
    
    def _pad_mask(self, x):
        # Example of padding tokens in a sequence:
        # Original sequence: [5, 2, 8, 1]
        # Padded sequence: [5, 2, 8, 1, 0, 0, 0]  # where 0 is the pad_idx
        # The padding mask would be: [0, 0, 0, 0, 1, 1, 1]  # 1 indicates padding position
        # Padding tokens are special tokens used to make all sequences in a batch the same length
        # They are typically represented by a special index (pad_idx) in the vocabulary
        # For example, if pad_idx = 0, then all padding tokens in the sequence will be 0
        # This is necessary because neural networks expect fixed-size inputs
        # The padding mask ensures the model doesn't attend to these padding tokens
        # pad_idx is used to mark padding tokens in sequences
        # pad_mask masks out padding tokens to prevent attention to them
        # x: [batch, seq, dim], pad_idx [dim]
        # [batch, seq] -> [batch_size, 1, seq_len]
        return (x == self.pad_idx).unsqueeze(1)


if __name__ == "__main__":
    # PositionEncoding(512,100)
    att = Transformer(
        src_vocab=100, tgt_vocab=200, pad_idx=0, d_model=512, n_layer=6, n_head=8, 
        d_ff=1024, dropout=0.1)
    x = torch.randint(0, 100, (4, 64))
    y = torch.randint(0, 200, (4, 64))
    out = att(x, y)
    print(out.shape)
