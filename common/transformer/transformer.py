import torch
from torch import nn
from typing import Tuple


class Encoder(nn.Module):

    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layer_count = layer_count
        layers = (EncoderLayer() for _ in range(self.layer_count))

    def forward(self, inputs: torch.Tensor):
        # Call each layer
        # Normalize each layer
        raise NotImplementedError


class EncoderLayer(nn.Module):

    def __init__(self, input_size: int, nhead: int=8) -> None:
        super().__init__()
        self.input_size = input_size
        self.nhead = nhead
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


class Transformer(nn.Module):

    def __init__(self, num_encoder_layers: int=6, num_decoder_layers: int=6) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor):
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
