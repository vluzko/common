import torch
from torch import profiler, nn, optim
from torch.nn import functional


class Net(nn.Module):
    """"""

    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(5, 10)
        self.layer_2 = nn.Linear(10, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_2(functional.relu(self.layer_1(inputs)))


def main():
    model = Net()
    opt = optim.Adam(model.parameters())

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        for i in range(1000):
            in_t = torch.randn(5)
            out_t = torch.randn(5)

            pred = model(in_t)
            loss = functional.mse_loss(pred, out_t)
            loss.backward
            opt.step()
    print(prof.key_averages())


if __name__ == "__main__":
    main()
