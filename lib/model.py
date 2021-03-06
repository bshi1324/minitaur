import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


HID_SIZE = 128


class D4PGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(D4PGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)


class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LeakyReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.LeakyReLU(),
            nn.Linear(300, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer('supports', torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


class AgentD4PG(ptan.agent.BaseAgent):
    def __init__(self, net, device='cpu', epsilon=0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states, agent_states):
        states = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        mu = self.net(states)
        actions = mu.data.cpu().numpy()
        actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states