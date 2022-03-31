from itertools import product
import torch
from torch import nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DTransformer(nn.Module):

    def __init__(self, d_model: int, n_head: int, n_layers: int, n_tokens) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_model)
        self.pos_embed = PositionalEncoding()
        self.decoder = Decoder(d_model, n_head, n_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        embedded = self.embed(x) + self.pos_embed(x)
        decoded = self.decoder(x)
        return decoded


class Decoder(nn.Module):
    """The entire decoder"""

    def __init__(self, d_model: int, n_head: int, n_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.layers = nn.Sequential(*[DecoderLayer(d_model, n_head) for _ in range(n_layers)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        self.attention = MHAttention(d_model, n_head)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Attention
        attn = self.attention(inputs)
        output = self.mlp(inputs + attn)
        return output


class MHAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scaled_dp = ScaledDotProductAttention(d_model, n_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_p = self.q_proj(q)
        k_p = self.k_proj(k)
        v_p = self.v_proj(v)
        scaled = self.scaled_dp(q_p, k_p, v_p)
        return scaled


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

    def forward(self, q, k, v) -> torch.Tensor:

        raise NotImplementedError


class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

def train(n_layers: int = 1, d_model: int=256, n_head: int=2):
    dataset = generate_dataset()
    n_tokens = dataset.shape[-1]
    model = DTransformer(d_model, n_head, n_layers, n_tokens)


def generate_dataset(upper: int=100) -> torch.Tensor:
    left = torch.arange(upper).to(DEVICE).repeat(upper).view(upper * upper, 1)
    right = torch.arange(upper).to(DEVICE).repeat_interleave(upper).view(upper * upper, 1)
    sums = left + right
    vals = torch.cat((left, right, left + right), dim=1)
    import pdb
    pdb.set_trace()

generate_dataset()


