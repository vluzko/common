import math
import torch

from einops import rearrange
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional


class DecoderOnly(nn.Module):
    """GPT"""

    def __init__(self, d_model: int, n_token: int, n_head: int, n_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_token = n_token
        self.n_head = n_head
        self.n_layers = n_layers
        self.token_embed = nn.Embedding(n_token, d_model)
        self.pos_embed = PositionalEmbedding(d_model)
        self.layers = nn.Sequential(*[DecoderLayer(d_model, n_head) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, n_token)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = self.pos_embed(self.token_embed(inputs))
        return self.linear(self.layers(inputs, mask))


class DecoderLayer(nn.Module):
    """A single decoder layer"""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return inputs + self.self_attn(inputs, inputs, inputs)


class MultiHeadAttention(nn.Module):
    """Multi headed attention"""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.q_w = nn.Linear(d_model, d_model)
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.sdp_attn = ScaledDotProductAttn(d_model)
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [seq_len, batch_size, d_model]
        """
        seq_len, batch_size, d_model = q.shape
        def head_split(x):
            return rearrange(x, 's b (nh dh) -> (b nh) s dh', nh=self.n_head, dh=self.d_head)
        # Split by heads, rearrange so batch and n_head are on the outside
        q_t = head_split(self.q_w(q))
        k_t = head_split(self.k_w(k))
        v_t = head_split(self.v_w(v))

        # Apply scaled dot product attention and concatenate heads
        return rearrange(self.sdp_attn(q_t, k_t, v_t, mask), '(b nh) s dh -> s b (nh dh)', nh=self.n_head, dh=self.d_head)



class ScaledDotProductAttn(nn.Module):
    """Scaled dot product attention"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.div = math.sqrt(d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        qk_t = torch.bmm(q, k.transpose(-2, -1)) / self.div

        if mask is not None:
            qk_t = qk_t.masked_fill(mask, -torch.inf)

        return torch.bmm(functional.softmax(qk_t), v)


class PositionalEmbedding(nn.Module):
    """Sin/Cos positional embedding"""
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int=27) -> None:
        super().__init__()
        numerator = torch.arange(max_len, dtype=torch.float32).view(-1, 1)
        denominator = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000) / d_model)).view(1, -1)
        val = numerator @ denominator
        pe = torch.empty(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(val)
        pe[:, 0, 1::2] = torch.cos(val)
        self.register_buffer('pe', pe)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        return inputs + self.pe[:inputs.shape[0]]


def train(batch_size: int, d_model: int, n_token: int, n_head: int, n_layers: int=2) -> DecoderOnly:
    # load data
    data_loader = make_arithmetic_loader(batch_size)
    model = DecoderOnly(d_model, n_token, n_head, n_layers)
    opt = optim.Adam(model.parameters(), lr=0.001)

    for i, (seq, padding, output_len) in enumerate(data_loader):
        # Construct mask
        pred = model(seq[:-1], mask)
        # TODO: Can't index like this. Needs to mask things so the padding just isn't used in the loss computation
        loss = loss_fn(pred[-(output_len + padding): -padding], seq[-(output_len + padding): -padding])

        loss.backward()
        opt.step()

    return model


def make_arithmetic_loader(batch_size: int, num_samples: int) -> DataLoader:
    for i in range(2, 6):
        lhs = torch.randint(0, 1000, (num_samples, i))
        rhs = torch.sum(lhs, dim=-1)
    raise NotImplementedError