import torch
import torch.nn as nn
from .block import SelfAttentionSubLayer, FeedForwardSubLayer

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attention_dropout, dropout):
        super().__init__()
        self.self_attn_layer = SelfAttentionSubLayer(d_model, nhead, attention_dropout)
        self.feed_forward_layer = FeedForwardSubLayer(d_model, dim_feedforward, dropout)

    def forward(self, x, padding_mask=None):
        x = self.self_attn_layer(x, None, padding_mask)
        x = self.feed_forward_layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, attention_dropout, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, attention_dropout, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        return x

