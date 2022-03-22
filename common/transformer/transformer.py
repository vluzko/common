import numpy as np
import torch
from torch import nn
from typing import Tuple, Optional


class Transformer(nn.Module):
    """
    Attributes:
        n_token: The number of tokens in the encoding
        n_head: The number of heads for MHA
        d_model: The dimension of each model
        d_hidden: The width of the hidden layers in the decoder and encoder
    """

    def __init__(self, n_token: int, n_head:int, d_model: int, d_hidden: int, num_encoder_layers: int=6, num_decoder_layers: int=6) -> None:
        super().__init__()
        self.n_token = n_token
        self.n_head = n_head
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder = Encoder(n_head, d_model, num_encoder_layers)
        self.decoder = Decoder(n_head, d_model, num_decoder_layers)
        self.token_embed = nn.Embedding(n_token, d_model)
        self.pos_embed = PositionalEncoding(d_model)

    def forward(self, inputs: torch.Tensor):
        pos_emb = self.pos_embed(inputs)
        embedding = self.token_embed(inputs)
        embedded = embedding + pos_emb
        raise NotImplementedError


# TODO: LayerNorm
class Encoder(nn.Module):
    """A transformer encoder.
    Consists of a number of identical layers, each of which is a combination of multi-head attention and a feedforward neural network

    Attributes:
        n_head: Number of heads in each layer
        d_model: The size of the model
        num_layers: The number of layers of attention
    """
    def __init__(self, n_head: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = (EncoderLayer(n_head, d_model) for _ in range(self.num_layers))

    def forward(self, inputs: torch.Tensor):

        val = inputs
        for layer in self.layers:
            val = layer(val)
        return val


# TODO: Normalization
class EncoderLayer(nn.Module):
    """A single layer in a transformer encoder.
    Attributes:
        input_size: The size of the input sequence
        n_head: The number of heads in the multi-head attention sublayer
    """

    def __init__(self, n_head: int, d_model: int, dim_linear: int=256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attention = MultiHeadedAttention(n_head, d_model)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_linear),
            nn.ReLU(),
            nn.Linear(dim_linear, d_model)
        )
        self.norm2 = LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor):
        attn = self.attention(inputs)
        norm_attn = self.norm1(inputs + attn)

        ff = self.feed_forward(norm_attn)
        return self.norm1(inputs + ff)


class Decoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError


class DecoderLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor):
        # self attention
        # encoder-decoder attention
        # feed forward
        raise NotImplementedError


class MultiHeadedAttention(nn.Module):
    """Multi headed attention layer

    Attributes:
        d_model: The incoming dimension of the query, key, and value
        n_head: The number of heads
        head_dimension: The output dimension of each head
    """

    def __init__(self, d_model: int, n_head: int, k_dim: Optional[int]=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0

        self.head_dimension = self.d_model // self.n_head
        # The output dimension is d_model, but really it's self.head_dimension * self.n_head
        self.query_lin = nn.Linear(d_model, d_model)
        self.key_lin = nn.Linear(d_model, d_model)
        self.value_lin = nn.Linear(d_model, d_model)
        self.scaled_dot_prod = ScaledDotProductAttention(self.head_dimension)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Transform query, key, and value
        q, k, v = inputs
        q_t = self.query_lin(q)
        k_t = self.key_lin(k)
        v_t = self.value_lin(v)

        # TODO: Masking
        dot_prod = self.scaled_dot_prod(q_t, k_t, v_t)

        return self.linear(dot_prod)


class ScaledDotProductAttention(nn.Module):
    """Compute QK^T / sqrt(d_k)"""
    def __init__(self, head_dimension: int) -> None:
        super().__init__()
        self.head_dimension = head_dimension

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scaled = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.head_dimension)

        attn = torch.softmax(scaled, dim=-1)
        context = torch.bmm(attn, v)
        return context


class ResNorm(nn.Module):
    """Wrap a sublayer with a residual connection and a normalization.
    Attributes:
        sublayer: The layer to wrap. Really it's MHA or linear.
    """

    def __init__(self, sublayer: nn.Module) -> None:
        super().__init__()
        self.sublayer = sublayer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """Encode sequence positions"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LayerNorm(nn.Module):

    def __init__(self, d_model: int) -> None:
        self.d_model = d_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


