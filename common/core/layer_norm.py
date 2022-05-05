import torch
from torch import nn


class MyNorm(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        var = x.var(dim=1)
        return (x - mean.view(-1, 1)) / (torch.sqrt(var.view(-1, 1) + 1e-5))
        # return x / torch.norm(x)


def main():
    mine = MyNorm()
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.rand(batch, embedding_dim) * 10 + 5
    t_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    print(mine(embedding))
    print(t_norm(embedding))


if __name__ == '__main__':
    main()