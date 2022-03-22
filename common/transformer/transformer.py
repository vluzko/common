import torch
from torch import nn
from typing import Tuple, Optional


class Transformer(nn.Module):
<<<<<<< HEAD
    """A multi-headed attention transformer.

    Attributes:
        n_token (int):              The number of tokens in the vocabulary.
        d_model (int):              The dimension of the model (i.e. the dimension of the space we embed tokens into).
        num_encoder_layers (int):   The number of encoder layers.
        num_decoder_layers (int):   The number of decoder layers.
    """

    def __init__(self, n_token: int, d_model: int, num_encoder_layers: int=6, num_decoder_layers: int=6, n_head: int=8) -> None:
        super().__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.encoder = Encoder(d_model, n_head, num_encoder_layers)
        self.decoder = Decoder(d_model, n_head, num_decoder_layers)
        self.positional_embedding = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.embedding = nn.Embedding(self.n_token, self.d_model)
=======
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
>>>>>>> cedd3d89ae016ff6472ef9e84b7fe32730d014f0

    def forward(self, inputs: torch.Tensor):
<<<<<<< HEAD
        pos_emb = self.positional_embedding(inputs)
        embedding = self.embedding(inputs)
        embedded = embedding + pos_emb
        raise NotImplementedError
=======
        embedded = self.token_embed(inputs) + self.pos_embed(inputs)
        encoded = self.encoder(embedded)
        decoded = self.decoder(encoded)
        return decoded
>>>>>>> 5c8ac6666c7b7d4817f582eb45a5b26b66fe94ec


# TODO: LayerNorm
class Encoder(nn.Module):
    """A transformer encoder.
    Consists of a number of identical layers, each of which is a combination of multi-head attention and a feedforward neural network

    Attributes:
        n_head: Number of heads in each layer
        d_model: The size of the model
        num_layers: The number of layers of attention
    """

<<<<<<< HEAD
    def __init__(self, input_size: int, layer_count: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.layer_count = layer_count
        layers = (EncoderLayer(input_size, n_head) for _ in range(self.layer_count))
=======
    def __init__(self, n_head: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = (EncoderLayer(n_head, d_model) for _ in range(self.num_layers))
>>>>>>> 5c8ac6666c7b7d4817f582eb45a5b26b66fe94ec

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
        head_dimension: The dimension of each head
    """

    def __init__(self, d_model: int, n_head: int, k_dim: Optional[int]=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.k_dim = k_dim if k_dim is not None else d_model
        assert self.d_model % self.n_head == 0

        self.head_dimension = self.d_model // self.n_head
        self.query_lin = nn.Linear(d_model, d_model)
        self.key_lin = nn.Linear(d_model, d_model)
        self.value_lin = nn.Linear(d_model, d_model)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Transform query, key, and value
        q, k, v = inputs
        q_t = self.query_lin(q)
        k_t = self.key_lin(k)
        v_t = self.value_lin(v)

        # Mask if applicable
        # dot product
        raise NotImplementedError


# class MHALinear(nn.Module):

#     def __init__(self, d_input: int, n_heads: int) -> None:
#         super().__init__()
#         self.d_input = d_input
#         self.n_heads = n_heads
#         self.linear = nn.Linear

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         transformed = self.linear(inputs)
#         # Reshape
#         reshaped = rearrange(transformed)
#         return reshaped
#         raise NotImplementedError



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


