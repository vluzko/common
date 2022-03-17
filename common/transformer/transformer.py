import torch
from torch import nn
from typing import Tuple


class Transformer(nn.Module):
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

    def forward(self, inputs: torch.Tensor):
        pos_emb = self.positional_embedding(inputs)
        embedding = self.embedding(inputs)
        embedded = embedding + pos_emb
        raise NotImplementedError


class Encoder(nn.Module):
    """A transformer encoder.
    Consists of a number of identical layers, each of which is a combination of multi-head attention and a feedforward neural network

    Attributes:
        layer_count: The number of layers of attention
    """

    def __init__(self, input_size: int, layer_count: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.layer_count = layer_count
        layers = (EncoderLayer(input_size, n_head) for _ in range(self.layer_count))

    def forward(self, inputs: torch.Tensor):
        # Call each layer
        # Normalize each layer
        raise NotImplementedError


class EncoderLayer(nn.Module):
    """A single layer in a transformer encoder.
    Attributes:
        input_size: The size of the input sequence
        n_head: The number of heads in the multi-head attention sublayer
    """

    def __init__(self, input_size: int, n_head: int=8) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.attention = MultiHeadedAttention()
        self.feed_forward = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear()
        )

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError


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
        heads: The number of heads
        dimension: The incoming dimension of the query, key, and value
    """

    def __init__(self, heads: int=8, dimension: int=512) -> None:
        super().__init__()
        self.heads = heads
        self.dimension = dimension

        assert self.dimension % self.heads == 0
        self.head_dimension = self.dimension // self.heads
        self.query_lin = nn.Linear()
        self.key_lin = nn.Linear()
        self.value_lin = nn.Linear()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Transform query, key, and value
        q, k, v = inputs
        q_t = self.query_lin(q)
        k_t = self.key_lin(k)
        v_t = self.value_lin(v)

        # Mask if applicable
        # dot product
        raise NotImplementedError


class MHALinear(nn.Module):

    def __init__(self, d_input: int, n_heads: int) -> None:
        super().__init__()
        self.d_input = d_input
        self.n_heads = n_heads
        self.linear = nn.Linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.linear(inputs)
        # Reshape
        reshaped = rearrange(transformed)
        return reshaped
        raise NotImplementedError



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


