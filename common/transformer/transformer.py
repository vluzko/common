import numpy as np
import torch
from torch import nn
from typing import Tuple, Optional


class Transformer(nn.Module):
    """An encoder-decoder transformer
    Attributes:
        n_token: The number of tokens in the encoding
        n_head: The number of heads for MHA
        d_model: The dimension of each model / the size of the embedding.
        d_hidden: The width of the hidden layers in the decoder and encoder
    """

    def __init__(self, n_token: int, n_head: int, d_model: int, d_hidden: int, num_encoder_layers: int=6, num_decoder_layers: int=6, dropout: float=0.2) -> None:
        super().__init__()
        self.n_token = n_token
        self.n_head = n_head
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder = Encoder(n_head, d_model, num_encoder_layers)
        self.decoder = Decoder(n_head, d_model, num_decoder_layers)
        self.token_embed = nn.Embedding(n_token, d_model)
        self.pos_embed = PositionalEncoding(d_model)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the whole transformer

        Args:
            inputs (torch.Tensor): The input sequence. Each element is a token. Shape: [seq_len, batch_size]

        Returns:
            torch.Tensor: The decoded string
        """
        # pos_emb = self.pos_embed(inputs)
        embedding = self.token_embed(inputs)
        # embedded = embedding + pos_emb
        encoded = self.encoder(embedding)
        decoded = self.decoder(encoded)
        return decoded


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
        self.layers = nn.Sequential(*[EncoderLayer(n_head, d_model) for _ in range(self.num_layers)])

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn = self.attention(inputs, inputs, inputs)
        norm_attn = self.norm1(inputs + attn)
        import pdb
        pdb.set_trace()
        ff = self.feed_forward(norm_attn)
        return self.norm1(inputs + ff)


class Decoder(nn.Module):
    """
    Attributes:
        n_head: Number of heads for multi headed attention
        d_model: Dimension of the model
        num_layers: The number of individual decoder layers
    """
    def __init__(self, n_head: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = [DecoderLayer(n_head, d_model) for _ in range(self.num_layers)]

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError


class DecoderLayer(nn.Module):

    def __init__(self, n_head: int, d_model: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

    def forward(self, inputs: torch.Tensor):
        # self attention
        # encoder-decoder attention
        # feed forward
        raise NotImplementedError


class MultiHeadedAttention(nn.Module):
    """Multi headed attention layer

    Attributes:
        d_model:            The incoming dimension of the query, key, and value
        n_head:             The number of heads
        head_dimension:     The output dimension of each head
    """

    def __init__(self, n_head: int, d_model: int) -> None:
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run MHA

        Args:
            q (torch.Tensor): Query tensor. Shape: [seq_len, batch_size, d_model]
            k (torch.Tensor): Key tensor. Shape: [seq_len, batch_size, d_model]
            v (torch.Tensor): Value tensor. Shape: [seq_len, batch_size, d_model]

        Returns:
            torch.Tensor: The context tensor. Shape: [seq_len, batch_size, d_model]
        """
        # Transform query, key, and value
        q_t = self.query_lin(q)
        k_t = self.key_lin(k)
        v_t = self.value_lin(v)

        # Reshape all to [batch_size * n_head, seq_len, d_head]

        # TODO: Masking
        dot_prod = self.scaled_dot_prod(q_t, k_t, v_t)

        return self.linear(dot_prod)


class ScaledDotProductAttention(nn.Module):
    """Compute softmax(QK^T / sqrt(d_k)) V"""
    def __init__(self, head_dimension: int) -> None:
        super().__init__()
        self.head_dimension = head_dimension

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute attention

        Args:
            q (torch.Tensor): Query tensor. Shape: [seq_len, batch_size, d_model]
            k (torch.Tensor): Key tensor. Shape: [seq_len, batch_size, d_model]
            v (torch.Tensor): Value tensor. Shape: [seq_len, batch_size, d_model]

        Returns:
            torch.Tensor: The context tensor. Shape: [seq_len, batch_size, d_model]
        """
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

    def __init__(self, d_model: int, max_len: int=5000) -> None:
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LayerNorm(nn.Module):

    def __init__(self, d_model: int, eps: float=1e-8) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Layer normalization

        Args:
            inputs (torch.Tensor): Activation to normalize

        Returns:
            torch.Tensor: Normalized activation
        """
        mean = inputs.mean(-1, keepdim=True)
        var = torch.square(inputs - mean).mean(-1, keepdim=True)
        val = (inputs - mean) / torch.sqrt(var + self.eps)
        return val

