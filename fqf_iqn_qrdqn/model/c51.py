from torch import nn
import torch
from fqf_iqn_qrdqn.network import DQNBase, NoisyLinear

class CatDQN(nn.Module):
    def __init__(self, n, num_channels, n_atoms=101, v_min=-100, v_max=100,
                 dueling_net=False, noisy_net=False):
        super().__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        # self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        # self.n = env.single_action_space.n
        self.n = n
        if not dueling_net:
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
        else:
            self.advantage_net =nn.Sequential(
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
            self.baseline_net =nn.Sequential(
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
        # if not self.dueling_net:
        logits = self.network(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        # else:
        #     advantage_logits = self.advantage_net(x / 255.0)
        #     baseline_logits = self.baseline_net(x / 255.0)
        #     advantage_q_values = torch.softmax(advantage_logits.view(len(x), self.n, self.n_atoms), dim=2)
        #     baseline_q_values = torch.softmax(baseline_logits.view(len(x), self.n, self.n_atoms), dim=2)

        return action, pmfs[torch.arange(len(x)), action]
