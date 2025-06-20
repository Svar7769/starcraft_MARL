import gym
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features, actions

from absl import flags
flags.FLAGS([''])    # pretend we parsed an empty arglist


class SC2GymEnv(gym.Env):
    def __init__(self,
                 map_name="Simple64",
                 step_mul=8,
                 screen_size=32,
                 minimap_size=32):
        super().__init__()

        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=screen_size,
                    minimap=minimap_size),
                use_feature_units=True,
            ),
            step_mul=step_mul,
            disable_fog=True,
        )


        # We’ll just stack the “player_relative” screen layer plus the unit density layer
        self.in_channels = 2
        self.screen_size = screen_size

        # Observation: a tensor of shape (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.in_channels, screen_size, screen_size),
            dtype=np.float32
        )

        # Action: choose from the set of raw functions that take no args,
        # plus a small palette of point‐targeted actions.
        # (You may need to prune unavailable functions at each step in your agent.)
        self.available_functions = [
            actions.FUNCTIONS.no_op.id,
            actions.FUNCTIONS.Move_screen.id,
            actions.FUNCTIONS.Attack_screen.id,
            actions.FUNCTIONS.Train_Marine_quick.id,
        ]
        self.action_space = gym.spaces.Discrete(len(self.available_functions))

    def _get_obs(self, timestep):
        screen = timestep.observation.feature_screen.player_relative / 4.0
        # second channel: how many units on each pixel (normalized)
        density = timestep.observation.feature_screen.unit_density / 255.0
        stacked = np.stack([screen, density], axis=0).astype(np.float32)
        return stacked

    def reset(self):
        timesteps = self.env.reset()
        return self._get_obs(timesteps[0])

    def step(self, action_idx):
        fn_id = self.available_functions[action_idx]
        fn = actions.FUNCTIONS[fn_id]

        # simple point‐at‐center for Move/Attack
        x = self.screen_size // 2
        y = self.screen_size // 2
        if fn_id in (actions.FUNCTIONS.Move_screen.id,
                     actions.FUNCTIONS.Attack_screen.id):
            args = [[0], [x, y]]
        else:
            args = [[0]]

        timestep = self.env.step([fn("now", *args)])[0]
        obs = self._get_obs(timestep)
        reward = timestep.reward
        done = timestep.last()
        return obs, reward, done, {}

    def close(self):
        self.env.close()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SC2DQN(nn.Module):
    def __init__(self, in_channels, n_actions, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        conv_out_size = 32 * self._feat_map_dim(32) * self._feat_map_dim(32)
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def _feat_map_dim(self, size, kernel=3, stride=1, padding=1):
        # here both conv layers use padding=kernel//2 & stride=1, so dim stays same
        return size

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

import random
import numpy as np
import torch.optim as optim
from collections import deque

env = SC2GymEnv()
agent = SC2DQN(env.in_channels, env.action_space.n)
target_agent = SC2DQN(env.in_channels, env.action_space.n)
target_agent.load_state_dict(agent.state_dict())

optimizer = optim.Adam(agent.parameters(), lr=1e-4)
criterion = nn.MSELoss()
replay = deque(maxlen=20000)

gamma      = 0.99
batch_size = 32
epsilon    = 1.0
eps_decay  = 0.995
eps_min    = 0.05
update_target_every = 500  # steps

step_count = 0
for ep in range(500):
    obs = env.reset()
    done = False
    while not done:
        # ε-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                qvals = agent(torch.tensor(obs[None], dtype=torch.float32))
                action = qvals.argmax().item()

        nxt_obs, reward, done, _ = env.step(action)
        replay.append((obs, action, reward, nxt_obs, done))
        obs = nxt_obs

        # train
        if len(replay) >= batch_size:
            batch = random.sample(replay, batch_size)
            states, acts, rews, nexts, dones = zip(*batch)
            states = torch.tensor(np.stack(states), dtype=torch.float32)
            acts   = torch.tensor(acts, dtype=torch.int64)[:,None]
            rews   = torch.tensor(rews, dtype=torch.float32)
            nexts  = torch.tensor(np.stack(nexts), dtype=torch.float32)
            dones  = torch.tensor(dones, dtype=torch.float32)

            q_preds = agent(states).gather(1, acts).squeeze()
            with torch.no_grad():
                q_next = target_agent(nexts).max(1)[0]
            q_targets = rews + gamma * q_next * (1 - dones)

            loss = criterion(q_preds, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        step_count += 1
        if step_count % update_target_every == 0:
            target_agent.load_state_dict(agent.state_dict())

    epsilon = max(epsilon * eps_decay, eps_min)
    print(f"Episode {ep:03d}  ε={epsilon:.3f}")

env.close()

