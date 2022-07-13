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
        return functional.log_softmax(self.layers(inputs), dim=1)


def plot_returns(returns: np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(returns)), returns)
    plt.show()


def train(env: gym.Env, model: nn.Module, max_steps: int, lr: float=1e-4, gamma: float=0.95):

    opt = optim.Adam(model.parameters(), lr=lr)
    reward_sum = []
    dones = []
    for i in range(max_steps):
        state = torch.from_numpy(env.reset()).float().to(DEVICE)
        total_reward = 0.0
        logits = model(state)
        actions = torch.distributions.Categorical(logits=logits).sample()
        state, reward, done, _ = env.step(actions.cpu().numpy())
        state = torch.from_numpy(state).float().to(DEVICE)

        loss = -torch.from_numpy(reward).float().to(DEVICE) * torch.take_along_dim(logits, actions)
        loss.mean().backward()
        opt.step()
        opt.zero_grad()
        total_reward += reward
        print(reward)
        reward_sum.append(total_reward)
        dones.append(done)


if __name__ == "__main__":
    env = gym.vector.make('CartPole-v1', num_envs=10)
    model = Model(env.observation_space.shape[1], env.action_space[0].n).to(DEVICE)  # type: ignore
    T = 1000
    train(env, model, T)