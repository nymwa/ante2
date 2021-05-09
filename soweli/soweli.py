import torch
import torch.nn as nn
from .embedding import TransformerEmbeddingWithoutPositionalEmbedding, TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder

class Soweli(nn.Module):
    def __init__(self, d_vocab, d_model, nhead, dim_feedforward,
            num_encoder_layers, num_decoder_layers, attention_dropout, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, nhead, dim_feedforward, num_encoder_layers, attention_dropout, dropout)
        self.decoder = Decoder(d_model, nhead, dim_feedforward, num_decoder_layers, attention_dropout, dropout)
        self.encoder_embedding, self.decoder_embedding, self.projection = self.make_embeddings(d_vocab, d_model, dropout)

    def make_embeddings(self, d_vocab, d_model, dropout):
        encoder_embedding = TransformerEmbeddingWithoutPositionalEmbedding(d_vocab, d_model, dropout)
        decoder_embedding = TransformerEmbedding(d_vocab, d_model, dropout)
        projection = nn.Linear(d_model, d_vocab)
        decoder_embedding.token_embedding = encoder_embedding.token_embedding
        projection.weight = encoder_embedding.token_embedding.weight
        return encoder_embedding, decoder_embedding, projection

    def encode(self, x, padding_mask=None):
        x = self.encoder_embedding(x)
        x = self.encoder(x, padding_mask = padding_mask)
        return x

    def decode(self, x, mem,
            attention_mask = None,
            encoder_padding_mask = None,
            decoder_padding_mask = None):
        x = self.decoder_embedding(x)
        x = self.decoder(x, mem,
                attention_mask = attention_mask,
                encoder_padding_mask = encoder_padding_mask,
                decoder_padding_mask = decoder_padding_mask)
        x = self.projection(x)
        return x

    def forward(self, batch):
        mem = self.encode(batch.encoder_inputs, padding_mask = batch.encoder_padding_mask)
        x = self.decode(batch.decoder_inputs, mem,
                attention_mask = batch.attention_mask,
                encoder_padding_mask = batch.encoder_padding_mask,
                decoder_padding_mask = batch.decoder_padding_mask)
        return x

