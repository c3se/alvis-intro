# Adapted from https://www.tensorflow.org/text/tutorials/transformer
import torch
from torch import nn


class DecoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.multi_head_attention = nn.MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feedforward_block = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, input):
        x = self.layer_norm1(input)
        x = input + self.multi_head_attention(x, x, x, need_weights=False)[0]
        x = x + self.feedforward_block(self.layer_norm2(x))
        return x


class GPT(nn.Module):
    '''An approximate implementation of a GPT model.
    
    Details regarding weight initialisation, which version of
    GPT and other things are not exact.
    '''

    def __init__(
            self,
            vocab_size,
            context_size=1024,
            n_layers=12,
            d_model=768,
            n_heads=12,
            dropout=0.1,
            verbose=False,
        ):
        super().__init__()
        
        self.verbose = verbose

        self.word_token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.word_position_embedding = nn.Embedding(context_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.decoder_blocks = nn.Sequential(*[
            DecoderBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.linear_to_logits = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input):
        if self.verbose:
            print("Hello from", input.device, "with input size", input.size())

        positions = torch.arange(0, input.size(-1)).unsqueeze(0).to(input.device)
        x = self.embedding_dropout(
            self.word_token_embedding(input)
            + self.word_position_embedding(positions)
        )
        x = self.final_layer_norm(x)
        x = self.decoder_blocks(x)
        x = self.linear_to_logits(x)
        return x


class Model(nn.Module):

    def __init__(self, input_size, output_size, verbose=False):
        super(Model, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, 5),
            nn.Linear(5, 5),
            nn.Linear(5, output_size),
        ])
        self.verbose = verbose

    def forward(self, input):
        if self.verbose:
            print("Hello from", input.device, "with input size", input.size())

        x = input
        for layer in self.layers:
            x = layer(x)
        return x
