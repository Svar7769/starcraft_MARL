import torch
import torch.nn as nn
from torch.distributions import Normal # For continuous action spaces (Controller)
from config.spartan import SCREEN_SIZE # Import SCREEN_SIZE from config.py

class BaseCNN(nn.Module):
    """
    Base CNN feature extractor for PySC2 screen/minimap observations.
    Used by King policies.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4), nn.Tanh(),
            nn.Conv2d(16, 32, 4, stride=2), nn.Tanh(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, SCREEN_SIZE, SCREEN_SIZE)
            self.conv_out_dim = self.conv(dummy).shape[-1]
    
    def forward(self, x):
        return self.conv(x)

class KingActorCritic(nn.Module):
    """
    Actor-Critic for Combat/Resource Kings.
    Takes preprocessed spatial observation features AND a controller goal vector.
    Outputs logits for discrete actions.
    """
    def __init__(self, in_channels, nb_actions, controller_goal_dim):
        super().__init__()
        self.feature_extractor = BaseCNN(in_channels) # Takes 3 channels (pr, ut, ud)
        
        # FC layer now takes conv output + controller_goal_dim
        self.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.conv_out_dim + controller_goal_dim, 256),
            nn.Tanh()
        )
        self.actor = nn.Linear(256, nb_actions) # For discrete actions
        self.critic = nn.Linear(256, 1)

    def forward(self, obs_features, controller_goal):
        """
        Forward pass for King policies.
        Args:
            obs_features (torch.Tensor): Preprocessed spatial observation (e.g., [N, C, H, W]).
            controller_goal (torch.Tensor): Latent goal vector from controller (e.g., [N, CONTROLLER_GOAL_DIM]).
        Returns:
            tuple: (action_logits, value)
        """
        h_conv = self.feature_extractor(obs_features)
        
        # Ensure controller_goal has a batch dimension consistent with h_conv
        if h_conv.shape[0] != controller_goal.shape[0]:
             # If controller_goal is for a single agent and h_conv has a batch, expand it
             controller_goal = controller_goal.expand(h_conv.shape[0], -1)

        h_combined = torch.cat((h_conv, controller_goal), dim=-1)
        
        h_fc = self.fc(h_combined)
        return self.actor(h_fc), self.critic(h_fc).squeeze(-1)

class ControllerActorCritic(nn.Module):
    """
    Actor-Critic for the Controller.
    Actor outputs mean and log_std for a continuous latent goal vector.
    """
    def __init__(self, obs_dim, goal_dim):
        super().__init__()
        # Controller observes high-level features, assumed to be flattened
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        # Actor outputs mean for the continuous goal vector
        self.actor_mean = nn.Linear(128, goal_dim)
        # Actor outputs log_std for the continuous goal vector (log to ensure positive std)
        # Initialize log_std to a small negative value for more stable exploration at start
        # Use a Parameter so it's learned
        self.actor_log_std = nn.Parameter(torch.zeros(1, goal_dim) - 0.5) # Initialize slightly negative for small std
        
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        """
        Forward pass for Controller policy.
        Args:
            obs (torch.Tensor): Preprocessed high-level observation (e.g., [N, obs_dim]).
        Returns:
            tuple: (goal_mean, goal_log_std, value)
        """
        h = self.fc(obs)
        
        goal_mean = self.actor_mean(h)
        goal_log_std = self.actor_log_std.expand_as(goal_mean) # Expand to match batch size
        
        value = self.critic(h).squeeze(-1)
        
        return goal_mean, goal_log_std, value

