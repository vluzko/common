import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, distributions
from torch.nn import functional


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
        return x


def plot_returns(returns: np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(returns)), returns)
    plt.show()


def train(env: gym.Env, model: nn.Module, max_steps: int, lr: float=1e-4, gamma: float=0.95):

    state = torch.from_numpy(env.reset()).float().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)

    for i in range(max_steps):
        action = model(state)

        state, reward, done, _ = env.step(action)
        state = torch.from_numpy(state).float().to(DEVICE)

        loss = 0
        loss.backward()
        opt.step()
        opt.zero_grad()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model = Model(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    T = 1000
    train(env, model, T)