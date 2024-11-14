from torch import nn
import torch

class CatDQN(nn.Module):
    def __init__(self, n, num_channels, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        # self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        # self.n = env.single_action_space.n
        self.n = n
        self.network = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]
