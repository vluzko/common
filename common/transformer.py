import torch
from torch import nn


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

    def __init__(self, heads: int=8) -> None:
        super().__init__()
        self.heads = heads

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


