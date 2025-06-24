import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, in_channels, nb_actions, screen_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4), nn.Tanh(),
            nn.Conv2d(16, 32, 4, stride=2), nn.Tanh(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, screen_size, screen_size)
            conv_out = self.conv(dummy).shape[-1]

        self.fc = nn.Sequential(nn.Linear(conv_out, 256), nn.Tanh())
        self.actor = nn.Linear(256, nb_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        return self.actor(h), self.critic(h).squeeze(-1)