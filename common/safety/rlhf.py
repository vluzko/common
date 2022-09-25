"""Reinforcement learning from human preferences"""
import gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType

# Load text data
# Generate preferences
# Binarize data
# Learn reward model from binarized data


def preference_model_pretrain(model, dataset):
    opt = optim.Adam(model)
    for i, (good, bad) in enumerate(dataset):
        good_r = model(good)
        bad_r = model(bad)

        loss = torch.log(1 + torch.exp(bad_r - good_r))
        loss.backward()
        opt.step()
        opt.zero_grad()


class HH(gym.Env):

    def __init__(self, pref_model, data_loader, seq_len):
        self.pref_model = pref_model
        self.data_loader = data_loader
        self.seq_len = seq_len

    def reset(self):
        self.cur_seq = self.data_loader.__next__()
        self.idx = 0
        toks = self.cur_seq[self.idx: self.idx + self.seq_len]
        self.idx +=1
        return toks

    def step(self, action):
        # Record the action (the predicted next token)
        # Output the next token
        # Output done at the end of the value
        next_token = self.cur_seq[self.idx: self.idx + self.seq_len]


def rl_loop(model, pref_model, dataset, num_trajectories):

    # Generate data
    trajectories = []
    for t in range(num_trajectories):
        trajectory = []
        for inputs, _ in dataset:
            output = model(inputs)
            trajectory.append((inputs, output))
        trajectories.append(trajectory)

    for i, o in trajectories:
        raise NotImplementedError

