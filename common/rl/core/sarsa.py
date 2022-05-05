import gym
import numpy as np
import torch
from torch import nn, optim, distributions
from torch.nn import functional
from typing import Tuple

from matplotlib import pyplot as plt


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_returns(returns: np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(returns)), returns)
    plt.show()


class Model(nn.Module):
    """Some model"""

    def __init__(self, state_dim: int, action_dim: int, width: int=128, eps: float=0.2) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = eps
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
        return self.layers(inputs)

    def policy(self, state: torch.Tensor) -> Tuple[torch.Tensor, int]:
        values = self.forward(state)
        action: int
        if np.random.uniform() < self.eps:
            action = np.random.randint(self.action_dim)
        else:
            action = torch.argmax(values).item()  # type: ignore
        return values[action], action

    def max_policy(self, state: torch.Tensor) -> int:
        return torch.argmax(self.forward(state)).item()


def train(env: gym.Env, model: nn.Module, max_steps: int, lr: float=1e-5, gamma: float=0.95):

    opt = optim.Adam(model.parameters(), lr=lr)
    for i in range(max_steps):
        done = False
        state = torch.from_numpy(env.reset()).float().to(DEVICE)
        while not done:
            value, action = model.policy(state)  # type: ignore

            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state).float().to(DEVICE)
            with torch.no_grad():
                next_val_est = reward + gamma * model.policy(state)[0] - value  # type: ignore
            loss = functional.mse_loss(value, next_val_est)
            loss.backward()
            opt.step()
            opt.zero_grad()

    return model


def evaluate(env: gym.Env, model: nn.Module, gamma: float=0.95) -> np.ndarray:
    returns = []
    for i in range(100):
        ret = 0
        cur_gam = gamma
        state = torch.from_numpy(env.reset()).float().to(DEVICE)
        done = False
        while not done:
            action = model.max_policy(state)  # type: ignore
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state).float().to(DEVICE)
            ret += cur_gam * reward
            cur_gam *= gamma
        returns.append(ret)
    return np.array(returns)



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model = Model(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    T = 1000
    model = train(env, model, T)
    returns = evaluate(env, model)
    plot_returns(returns)
