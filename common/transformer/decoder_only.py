import math
import pdb
import torch

from einops import rearrange
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from typing import Optional


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, n_token)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(self.token_embed(inputs))
        for l in self.layers:
            x = l(x, mask)
        return functional.softmax(self.linear(x), dim=2)


class DecoderLayer(nn.Module):
    """A single decoder layer"""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return inputs + self.self_attn(inputs, inputs, inputs, mask)


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

        sft = functional.softmax(qk_t, dim=-1)
        return torch.bmm(sft, v)


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


def train(batch_size: int, d_model: int, n_head: int, n_layers: int=2, num_samples: int=200) -> DecoderOnly:
    N_TOKEN = 17
    SEQ_LEN = 27
    # load data
    data_loader = make_arithmetic_loader(batch_size, num_samples)
    model = DecoderOnly(d_model, N_TOKEN, n_head, n_layers).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-5)
    attn_mask = torch.triu(torch.ones((SEQ_LEN - 1, SEQ_LEN - 1)), diagonal=1).to(DEVICE)
    range_tensor = torch.arange(7, 0, -1).view(1, -1).to(DEVICE)
    for i, (seq, padding) in enumerate(data_loader):
        cur_batch = seq.shape[0]

        # Construct the padding mask
        seq_mask = torch.ones((cur_batch, SEQ_LEN-1, SEQ_LEN-1)).to(DEVICE)
        seq_mask.mul_(seq.view(batch_size, 1, SEQ_LEN)[:, :, :-1] == 16)
        seq_mask = seq_mask.logical_or(attn_mask).repeat(n_head, 1, 1)

        seq = rearrange(seq, 'b s -> s b')
        pred = model(seq[:-1], seq_mask)

        # Annoying complicated indexing to avoid dealing with one-hot encoding
        index_tensor = (((SEQ_LEN - 1) - padding) - range_tensor).transpose(0, 1)
        gathered_pred = torch.gather(pred, 0, index_tensor.view(7, -1, 1).expand(-1, -1, 17))
        gathered_tgt = torch.gather(seq, 0, index_tensor + 1)
        cat_probs = torch.gather(gathered_pred, 2, gathered_tgt.long().view(7, cur_batch, 1))
        print(cat_probs.mean())
        loss = -torch.log(cat_probs).sum(dim=0).mean()

        print(loss)

        loss.backward()
        opt.step()
    import pdb
    pdb.set_trace()
    return model


def make_arithmetic_loader(batch_size: int, num_samples: int) -> DataLoader:
    MAX_LEN = 3 * 5 + 5 + 6 + 1  # 5 * 3 (digits) + 5 (operators) + 4 (solution) + stop
    OP_MAP = {
        torch.sum: 11,
        torch.sub: 12,
        torch.mul: 13,
        torch.div: 14
    }
    # NUM_TOKENS = 16
    all_tokens = torch.empty((4*num_samples, MAX_LEN), dtype=torch.uint8)
    all_padding = torch.empty((4*num_samples, 1), dtype=torch.uint8)
    for i in range(2, 6):
        op = torch.sum
        tokenized = all_tokens[(i-2) * num_samples: (i-1) * num_samples]
        lhs = torch.randint(0, 1000, (num_samples, i), dtype=torch.int16)
        rhs = op(lhs, dim=-1)
        # Hundreds
        tokenized[:, 0:4*i:4] = torch.div(lhs, 100)
        # Tens
        tokenized[:, 1:4*i:4] = torch.div((lhs % 100), 10)
        # Ones
        tokenized[:, 2:4*i:4] = lhs % 10
        # Operators
        tokenized[:, 3:4*i - 1:4] = OP_MAP[op]
        # Equals is 10
        tokenized[:, 4*i - 1] = 10

        # Solution
        tokenized[:, 4*i] = torch.div(rhs, 100000)
        tokenized[:, 4*i+1] = torch.div(rhs % 100000, 10000)
        tokenized[:, 4*i+2] = torch.div(rhs % 10000, 1000)
        tokenized[:, 4*i+3] = torch.div(rhs % 1000, 100)
        tokenized[:, 4*i+4] = torch.div(rhs % 100, 10)
        tokenized[:, 4*i+5] = rhs % 10
        # Stop
        tokenized[:, 4*i+6] = 15
        # Padding
        tokenized[:, 4*i+7:] = 16
        # Padding amount
        all_padding[(i-2) * num_samples: (i-1) * num_samples] = (tokenized[:, 4*i+7:]).shape[1]
    dataset = TensorDataset(all_tokens.to(DEVICE, dtype=torch.int), all_padding.to(DEVICE))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# train(batch_size=256, d_model=64, n_head=4, n_layers=2, num_samples=1000000)


def eigen_analysis(d_model, n_head, n_layers, batch_size, num_samples: int=200):
    N_TOKEN = 17
    model = DecoderOnly(d_model, N_TOKEN, n_head, n_layers).to(DEVICE)
    model.load_state_dict(torch.load('./tmp.torch'))
    data_loader = make_arithmetic_loader(batch_size, num_samples)

    i, (seq, _) = next(enumerate(data_loader))

    import pdb
    pdb.set_trace()
    # model = torch.load('./tmp.torch')


eigen_analysis(batch_size=256, d_model=64, n_head=4, n_layers=2, num_samples=200)