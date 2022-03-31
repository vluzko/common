import torch


class Module:

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError



class Linear(Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.zeros(self.input_dim, self.output_dim)
        # TODO: initialize

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight)
        
        raise NotImplementedError


class ReLU(Module):

    pass


