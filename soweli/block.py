import torch
import torch.nn as nn

class AttentionSubLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

class SelfAttentionSubLayer(AttentionSubLayer):
    def forward(self, x, mask, padding_mask):
        z = self.norm(x)
        z, _ = self.attn(z, z, z, attn_mask=mask, key_padding_mask=padding_mask)
        x = x + self.dropout(z)
        return x

class CrossAttentionSubLayer(AttentionSubLayer):
    def forward(self, x, mem, mask, padding_mask):
        z = self.norm(x)
        z, _ = self.attn(z, mem, mem, attn_mask=mask, key_padding_mask=padding_mask)
        x = x + self.dropout(z)
        return x

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        z = self.norm(x)
        z = self.linear1(z)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.linear2(z)
        x = x + self.dropout(z)
        return x

