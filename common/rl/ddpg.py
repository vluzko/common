# -*- coding: utf-8 -*-
"""ddpg_detailed.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GminGSf01GjdRkEDSJKtWmypPLEC2G3H
"""

import torch
import gym

from copy import deepcopy
from torch import nn, optim
from torch.nn import functional

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ActorCritic(nn.Module):
  actor: nn.Module
  policy: nn.Module

  def __init__(self):
    super().__init__()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def q(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def p(self, obs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

class Replay:

  def __init__(self) -> None:
      pass

  def store(self, obs, act, reward, next_obs, done):
    raise NotImplementedError

def actor_loss(q: torch.Tensor, target: ActorCritic, reward: torch.Tensor, next_state: torch.Tensor, next_action: torch.Tensor, done: bool, gamma: float):
  """The loss used to update the actor's parameters
  In DDPG this is MSBE, computed against the target network:
  loss = mean(
    (
      Q(s, a) - (reward + gamma * Q_target(s_next, a_next))
    )**2
  )
  """
  with torch.no_grad():
    next_action = target.p(next_state)
    next_q = target.q(next_state, next_action)
    backup = reward + gamma * (1 - done) * next_q
  q_loss = ((q - backup)**2).mean()
  return q_loss

def explore_action():
  """During training we use noisy actions since DDPG is deterministic"""
      # def get_action(o, noise_scale):
      #   a = ac.act(torch.as_tensor(o, dtype=torch.float32))
      #   a += noise_scale * np.random.randn(act_dim)
      #   return np.clip(a, -act_limit, act_limit)
  raise NotImplementedError

# TODO: Replay buffer
# TODO: Target networks
def update(env: gym.Env, model: ActorCritic, polyak: float=0.9, trajs: int=100):
  opt = optim.Adam(model.actor.parameters())
  p_opt = optim.Adam(model.policy.parameters())
  target = deepcopy(model)

  for t in range(trajs):
    # Run one trajectory
    state = torch.from_numpy(env.reset(), dtype=torch.float32).to(DEVICE)
    done = False
    while not done:
      action, value = model(state)
      next_state, reward, done, _ = env.step(action)
      next_state = torch.from_numpy(next_state, dtype=torch.float32).to(DEVICE)
  
      next_action = model.p(next_state)
      next_q = model.q(next_state, next_action)
  
      # Compute policy loss (- of Q value of (state, current_policy))
      policy_loss = -model.q(state, model.p(state)).mean()
      policy_loss.backward()
      p_opt.step()
  
      # Update target with polyak averaging
      with torch.no_grad():
        for p, targ_p in zip(model.parameters(), target.parameters()):
          targ_p.data.mul_(polyak)
          targ_p.data.add_((1 - polyak) * p.data)


  raise NotImplementedError

def train(batch_size: int, total_steps: int, update_after: int, update_every: int):
  """Core training loop.
  Run for k steps acquiring experience.
  Every k steps, go through replay buffer and update policies.
  """
  replay_buffer = Replay()
  for t in range(total_steps):
    if t >= update_after and t % update_every == 0:
      for _ in range(update_every):
        batch = replay_buffer.sample_batch(batch_size)
        update(data=batch)