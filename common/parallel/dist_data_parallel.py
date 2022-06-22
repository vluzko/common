import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from typing import List

import os
import sys
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    
class MLP(nn.Module):
    """An MLP"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=128, num_layers: int=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_widths: List[int] = [input_dim, *[hidden_dim]*(num_layers-1)]
        self.layers = nn.Sequential(*(x for i in range(num_layers-1) for x in 
            (nn.Linear(self.layer_widths[i], self.layer_widths[i+1]), nn.ReLU()
        )))
        self.final = nn.Linear(self.layer_widths[-1], output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.final(self.layers(inputs))


def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(rank)
    model = MLP(5, 5).to(rank)

    ddp_model = DistributedDataParallel(model, device_ids=[rank])

    opt = optim.Adam(ddp_model.parameters())

    opt.zero_grad()
    outputs = ddp_model(torch.randn(20, 5).to(rank))
    labels = torch.randn(20, 5).to(rank)
    functional.mse_loss(outputs, labels).backward()
    opt.step()
    cleanup()
    return ddp_model


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])


    CHECKPOINT_PATH = "./.data/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), open(CHECKPOINT_PATH, 'wb+'))
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # if rank == 0:
    #     ddp_model.load_state_dict(
    #         torch.load(CHECKPOINT_PATH, map_location=map_location))
    if rank == 0:
        print(list(ddp_model.parameters())[0])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=500)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()
    optimizer.step()
    if rank == 0:
        print(list(ddp_model.parameters())[0])

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    res = mp.spawn(demo_checkpoint, args=(world_size, ), nprocs=world_size, join=True)
    import pdb
    pdb.set_trace()
