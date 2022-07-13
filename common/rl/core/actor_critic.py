"""Just the raw code"""

# Standard imports
import numpy as np
import gym
import torch
from torch import nn, optim, distributions
from torch.nn import functional

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Note: as an alternative, we could combine these into one network with two outputs
class Actor(nn.Module):
    """Policy network"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.fc1(state))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        # Policy outputs probabilities over actions (no batch dimension)
        return functional.softmax(x)


class Critic(nn.Module):
    """Our value predictor"""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.fc1(state))
        x = functional.relu(self.fc2(x))
        # The value network outputs a single value
        return self.fc3(x)


def train(gamma: float=0.95):
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim).to(DEVICE)
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.001)


    state = torch.from_numpy(env.reset()).float().to(DEVICE)
    for i in range(1000):
        action_probs = actor(state)
        pred_value = critic(state)

        action = distributions.Categorical(probs=action_probs).sample()
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.from_numpy(next_state).float().to(DEVICE)
        next_pred_value = critic(next_state) if not done else 0
        # We update with log(pi(a, s)) * (reward + V(s') - V(s))
        next_reward_est = reward + gamma * next_pred_value - pred_value
        actor_loss = -torch.log(action_probs[action]) * next_reward_est

        # Just MSE loss is fine
        critic_loss = functional.mse_loss(pred_value, reward + gamma * next_pred_value)

        loss = actor_loss + critic_loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        print(loss.item())
        if done:
            state = torch.from_numpy(env.reset()).float().to(DEVICE)

    return actor, critic


def evaluate(actor, critic):
    pass


if __name__ == '__main__':
    actor, critic = train()
    evaluate(actor, critic)
