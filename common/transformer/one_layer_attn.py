import math
import numpy as np
import torch
import torchtext
from einops import rearrange
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType


class OneLayerAttn(nn.Module):
    """A single attention only layer module"""

    def __init__(self, d_model: int, n_head: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PosEncode(d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.pos_embed(self.embed(inputs))
        q, k, v = self.w_k(embedded), self.w_q(embedded), self.w_v(embedded)
        # Split into heads
        # attention
        # Mask
        # mat mul
        # linear out
        raise NotImplementedError


class PosEncode(nn.Module):
    """Positional encoding"""
    pe: torch.Tensor

    def __init__(self, d_model:int, max_len: int=2000) -> None:
        super().__init__()
        self.max_len = max_len
        numerator = torch.arange(self.max_len, dtype=torch.float32).view(-1, 1)
        exponent = torch.arange(0, self.max_len * 2, 2, dtype=torch.float32) * (-math.log(10000) / d_model)
        denominator = torch.exp(exponent).view(1, -1)
        val = numerator @ denominator
        pe = torch.empty((max_len, 1, d_model))
        pe[:, 0, 0::2] = torch.sin(val)
        pe[:, 0, 1::2] = torch.cos(val)
        self.register_buffer('pe', pe)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pe[inputs.shape[0]:]


class Model(nn.Module):
    """Wrapper around one layer attention"""

    def __init__(self, d_model: int, n_head: int, vocab_size: int) -> None:
        super().__init__()
        self.one_layer = OneLayerAttn(d_model, n_head, vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PosEncode(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(inputs)
        # Positional encoding
        encoded = self.pos(embedded) + embedded
        return self.one_layer(encoded)


def load_data():
    # Load some text dataset
    data_iter = torchtext.datasets.IMDB(split='train')
    return data_iter


def build_vocab(data):
    tokenizer = torchtext.data.get_tokenizer('spacy', language='en_core_web_sm')
    iterator = (tokenizer(x) for pair in data for x in pair)
    vocab = torchtext.vocab.build_vocab_from_iterator(iterator)
    return vocab


def train(d_model: int=256, n_head: int=8, lr: float=1e-5):
    data = load_data()
    vocab = build_vocab(data)
    vocab_size = len(vocab)
    model = Model(d_model, n_head, vocab_size)
    opt = optim.Adam(model.parameters(), lr=lr)

    for i, (data, target) in enumerate(data):
        emb_data, emb_target = vocab(data), vocab(target)
        # Masking
        output = model(data)
        loss = functional.binary_cross_entropy(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        import pdb
        pdb.set_trace()

    return model


def analyze():
    # Run models
    # Freeze attention layer
    raise NotImplementedError


def nat_contrib():
    """Calculate the information contribution of each head and circuit"""
    raise NotImplementedError


def main():
    train()


if __name__ == "__main__":
    main()
