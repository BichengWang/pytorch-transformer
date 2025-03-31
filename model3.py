from torch import nn
import torch
import math


class PosEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        # [max_seq_len, 1]
        """
        bug: no 2d
        """
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        # [d_model / 2]
        mul = 1./10_000 ** (torch.arange(0, d_model, 2) / d_model)
        pe = torch.ones(max_seq_len, d_model)
        # [max_seq_len, 1], [d_model / 2] -> [max_seq_len, d_model/2]
        tmp = pos * mul
        pe[:, 0::2] = torch.sin(tmp)
        pe[:, 1::2] = torch.cos(tmp)
        # [max_seq_len, d_model] -> [1, max_seq_len, d_model]
        """
        bug: no 3d
        """
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        bug: use x.size(1) for 2nd dim
        """
        return x + self.pe[:, :x.size(1), :]


class MHA(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        """bug: check d_head
        """
        assert d_model % n_head == 0
        self.qw = nn.Linear(d_model, d_model, bias=False)
        self.kw = nn.Linear(d_model, d_model, bias=False)
        self.vw = nn.Linear(d_model, d_model, bias=False)
        self.ow = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model
        """check scale
        """
        self.scale = self.d_head ** -0.5
        
    def forward(self, q, k, v, mask=None):
        print("\n==== MHA ====")
        batch, seq, dim = q.size()
        print("batch, seq, dim")
        print(f"q: {q.size()}")
        if dim != self.d_model:
            raise ValueError("dim not match")
        ql, kl, vl = seq, k.size(1), v.size(1)
        
        # Linear projection and reshape for multi-head attention
        # [batch, seq_len, d_model]
        # -> [batch, seq_len, n_head, d_head]
        # -> [batch, n_head, seq_len, d_head]
        q = self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)
        k = self.kw(k).reshape(batch, kl, self.n_head, self.d_head).transpose(1, 2)
        v = self.vw(v).reshape(batch, vl, self.n_head, self.d_head).transpose(1, 2)
        print("self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)")
        print("[batch, n_head, seq_len, d_head]")
        print(f"q: {q.size()}")
        
        # [batch, n_head, seq_len, d_head] @ [batch, n_head, d_head, seq_len] 
        # -> [batch, n_head, seq_len, seq_len]
        attn_score = torch.matmul(q * self.scale, k.transpose(-2, -1))
        print("torch.matmul(q * self.scale, k.transpose(-2, -1))")
        print("[batch, n_head, seq_len, seq_len]")
        print(f"attn_score: {attn_score.size()}")
        
        if mask is not None:
            # if mask.dim() == 2:
            #     mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]
            # elif mask.dim() == 3:
            #     mask = mask.unsqueeze(1) # [batch_size, 1, seq_len, seq_len]
            # if mask.dtype != torch.bool:
            #     mask = mask.bool()
            # Expand mask to match attention score dimensions
            # mask = mask.expand(batch, self.n_head, ql, kl)
            print(f"mask: {mask.size()}")
            attn_score = attn_score.masked_fill(mask, float('-inf'))
        
        # [batch, n_head, seq_len, seq_len] -> [batch, n_head, seq_len, seq_len]
        # Softmax along the last dimension to get attention weights
        attn = torch.softmax(attn_score, dim=-1)
        print("torch.softmax(attn_score, dim=-1)")
        print("[batch, n_head, seq_len, seq_len]")
        print(f"attn: {attn.size()}")
        
        """it's softmax, so no scale
        """
        out = torch.matmul(attn, v)
        print("torch.matmul(attn, v)")
        print("[batch, n_head, seq_len, d_head]")
        print(f"out: {out.size()}")
        # transpose 1, 2, reshape
        out = out.transpose(1, 2).reshape(batch, ql, self.d_model)
        print("out.transpose(1, 2).reshape(batch, ql, self.d_model)")
        print("[batch, seq_len, d_model]")
        print(f"out: {out.size()}")
        
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
        
    def forward(self, x, memory, tgt_causal_mask=None, tgt_pad_mask=None, memory_pad_mask=None):
        out = x
        for layer in self.decode_layers:
            out = layer(out, memory, tgt_causal_mask, tgt_pad_mask, memory_pad_mask)
        return x


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
        print("\n==== start ====")
        src_pad_mask = self._pad_mask(src)
        print("src_pad_mask: [batch_size, 1, 1, seq_len]")
        print(f"src_pad_mask: {src_pad_mask.dim()}, {src_pad_mask.size()}")
        print("tgt_pad_mask: [batch_size, 1, 1, seq_len]")
        tgt_pad_mask = self._pad_mask(tgt)
        print(f"tgt_pad_mask: {tgt_pad_mask.dim()}, {tgt_pad_mask.size()}")
        print("tgt_causal_mask: [1, 1, seq, seq]")
        tgt_causal_mask = self._causal_mask(tgt)
        print(f"tgt_causal_mask: {tgt_causal_mask.dim()}, {tgt_causal_mask.size()}")
        
        src = self.src_emb(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        
        print("\n==== encoder ====")
        encoder_out = self.encoder(src, src_pad_mask)
        print("\n==== decoder ====")
        decoder_out = self.decoder(
            tgt, encoder_out, tgt_causal_mask, tgt_pad_mask, src_pad_mask)
        print("\n==== output ====")
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
        # [1, 1, seq, seq]
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _pad_mask(self, x):
        # Example of padding tokens in a sequence:
        # Original sequence: [5, 2, 8, 1]
        # Padded sequence: [5, 2, 8, 1, 0, 0, 0]  # where 0 is the pad_idx
        # x: [batch, seq], pad_idx [1]
        # [batch, seq] -> [batch_size, 1, 1, seq_len]
        return (x == self.pad_idx).unsqueeze(1).unsqueeze(1)


if __name__ == "__main__":
    """
    batch, seq: (2, 5), (2, 3)

    ==== start ====
    src_pad_mask: [batch_size, 1, 1, seq_len]
    src_pad_mask: 4, torch.Size([2, 1, 1, 5])
    tgt_pad_mask: [batch_size, 1, 1, seq_len]
    tgt_pad_mask: 4, torch.Size([2, 1, 1, 3])
    tgt_causal_mask: [1, 1, seq, seq]
    tgt_causal_mask: 4, torch.Size([1, 1, 3, 3])

    ==== encoder ====

    ==== MHA ====
    batch, seq, dim
    q: torch.Size([2, 5, 32])
    self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)
    [batch, n_head, seq_len, d_head]
    q: torch.Size([2, 2, 5, 16])
    torch.matmul(q * self.scale, k.transpose(-2, -1))
    [batch, n_head, seq_len, seq_len]
    attn_score: torch.Size([2, 2, 5, 5])
    mask: torch.Size([2, 1, 1, 5])
    torch.softmax(attn_score, dim=-1)
    [batch, n_head, seq_len, seq_len]
    attn: torch.Size([2, 2, 5, 5])
    torch.matmul(attn, v)
    [batch, n_head, seq_len, d_head]
    out: torch.Size([2, 2, 5, 16])
    out.transpose(1, 2).reshape(batch, ql, self.d_model)
    [batch, seq_len, d_model]
    out: torch.Size([2, 5, 32])

    ==== decoder ====

    ==== MHA ====
    batch, seq, dim
    q: torch.Size([2, 3, 32])
    self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)
    [batch, n_head, seq_len, d_head]
    q: torch.Size([2, 2, 3, 16])
    torch.matmul(q * self.scale, k.transpose(-2, -1))
    [batch, n_head, seq_len, seq_len]
    attn_score: torch.Size([2, 2, 3, 3])
    mask: torch.Size([1, 1, 3, 3])
    torch.softmax(attn_score, dim=-1)
    [batch, n_head, seq_len, seq_len]
    attn: torch.Size([2, 2, 3, 3])
    torch.matmul(attn, v)
    [batch, n_head, seq_len, d_head]
    out: torch.Size([2, 2, 3, 16])
    out.transpose(1, 2).reshape(batch, ql, self.d_model)
    [batch, seq_len, d_model]
    out: torch.Size([2, 3, 32])

    ==== MHA ====
    batch, seq, dim
    q: torch.Size([2, 3, 32])
    self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)
    [batch, n_head, seq_len, d_head]
    q: torch.Size([2, 2, 3, 16])
    torch.matmul(q * self.scale, k.transpose(-2, -1))
    [batch, n_head, seq_len, seq_len]
    attn_score: torch.Size([2, 2, 3, 3])
    mask: torch.Size([2, 1, 1, 3])
    torch.softmax(attn_score, dim=-1)
    [batch, n_head, seq_len, seq_len]
    attn: torch.Size([2, 2, 3, 3])
    torch.matmul(attn, v)
    [batch, n_head, seq_len, d_head]
    out: torch.Size([2, 2, 3, 16])
    out.transpose(1, 2).reshape(batch, ql, self.d_model)
    [batch, seq_len, d_model]
    out: torch.Size([2, 3, 32])

    ==== MHA ====
    batch, seq, dim
    q: torch.Size([2, 3, 32])
    self.qw(q).reshape(batch, ql, self.n_head, self.d_head).transpose(1, 2)
    [batch, n_head, seq_len, d_head]
    q: torch.Size([2, 2, 3, 16])
    torch.matmul(q * self.scale, k.transpose(-2, -1))
    [batch, n_head, seq_len, seq_len]
    attn_score: torch.Size([2, 2, 3, 5])
    mask: torch.Size([2, 1, 1, 5])
    torch.softmax(attn_score, dim=-1)
    [batch, n_head, seq_len, seq_len]
    attn: torch.Size([2, 2, 3, 5])
    torch.matmul(attn, v)
    [batch, n_head, seq_len, d_head]
    out: torch.Size([2, 2, 3, 16])
    out.transpose(1, 2).reshape(batch, ql, self.d_model)
    [batch, seq_len, d_model]
    out: torch.Size([2, 3, 32])

    ==== output ====
    torch.Size([2, 3, 50])
    x: torch.Size([2, 5])
    y: torch.Size([2, 3])
    out: torch.Size([2, 3, 50])
    argmax: tensor([[13, 16, 48],
            [42, 13, 16]])
    """
    # PositionEncoding(512,100)
    att = Transformer(
        src_vocab=100, tgt_vocab=50, pad_idx=0, d_model=32, n_layer=1, n_head=2, 
        d_ff=32, dropout=0.1)
    # batch, seq
    print("batch, seq: (2, 5), (2, 3)")
    x = torch.randint(0, 100, (2, 5))
    y = torch.randint(0, 50, (2, 3))
    out = att(x, y)
    print(out.size())
    print(f"x: {x.size()}")
    print(f"y: {y.size()}")
    print(f"out: {out.size()}")
    print(f"argmax: {torch.argmax(out, dim=-1)}")
