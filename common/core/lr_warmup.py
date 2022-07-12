import torch
from torch import optim


class LRWarmup:
    

    def __init__(self, switch_time):
        self.switch_time = switch_time

    def step(self):
        if t < self.switch_time:
            self.first.step()
        else:
            self.second.step()
