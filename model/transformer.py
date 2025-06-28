import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.size()
        # Create a lower triangular mask for causal self-attention
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_output, _ = self.attn(x, x, x, attn_mask=~causal_mask)
        x = self.ln(x + self.dropout(attn_output))
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x + self.net(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, num_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        token_emb = self.token_embed(input_ids)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)

        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits
