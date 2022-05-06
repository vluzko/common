import cProfile
import pstats
import gym
import torch
from torch import nn, optim, distributions
from torch.nn import functional
from torch.profiler import profile, record_function, ProfilerActivity

from typing import List


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    """Some model"""

    def __init__(self, state_dim: int, action_dim: int, width: int=128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, action_dim)
        self.layers = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layers(inputs)
        return functional.softmax(x)


def train(env: gym.Env, policy: nn.Module, trajectories: int, lr: float=1e-4, gamma: float=0.95):

    opt = optim.Adam(policy.parameters(), lr=lr)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     with record_function('training'):
    for i in range(trajectories):
        state = torch.from_numpy(env.reset()).float().to(DEVICE)
        done = False
        rewards = []
        log_probs = []
        while not done:
            action_probs = policy(state)
            action = distributions.Categorical(action_probs).sample()
            log_prob = torch.log(action_probs[action.item()])

            state, reward, done, _ = env.step(action.item())
            state = torch.from_numpy(state).float().to(DEVICE)
            rewards.append(reward)
            log_probs.append(log_prob)

        # returns = calculate_returns(rewards, gamma)

        last_ret = 0
        for j in range(len(rewards) - 1, -1, -1):
            ret = rewards[j] + gamma * last_ret
            log_prob = log_probs[j]
            loss = -log_prob * ret
            loss.backward()
            last_ret = ret

        opt.step()
        opt.zero_grad()

                # if i % 50 == 0:
                #     print(f'{len(rewards)}')
    # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=50))


def calculate_returns(rewards: List[int], gamma: float) -> torch.Tensor:
    returns = torch.empty(len(rewards)).to(DEVICE)

    for i, r in reversed(list(enumerate(rewards))):
        returns[i] = r + gamma * returns[i + 1] if i < len(rewards) - 1 else r

    return returns


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    policy = Model(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    T = 1000
    profiler = cProfile.Profile()
    profiler.enable()
    train(env, policy, T)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(50)
    # profiler.print_stats(sort='time')