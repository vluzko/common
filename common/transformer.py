from torch import nn


class Transformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        raise NotImplementedError


class Encoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


class MultiHeadedAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        raise NotImplementedError


