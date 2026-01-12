import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class SimpleBrain(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleBrain, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.log_std = nn.Parameter(torch.zeros(output_size))
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.tanh(self.fc1(x))
        mean = torch.tanh(self.fc2(x))
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob