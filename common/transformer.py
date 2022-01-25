import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layer_count = layer_count
        layers = (EncoderLayer() for _ in range(self.layer_count))

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError


class EncoderLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
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


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError





class MultiHeadedAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


