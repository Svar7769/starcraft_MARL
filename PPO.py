import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from absl import app
from pysc2.env import sc2_env, run_loop
from pysc2.agents import base_agent
from pysc2.lib import actions, features

# ==== PPO Hyperparameters ====
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENT_COEF        = 0.01
VF_COEF         = 0.5
LR              = 2.5e-4
UPDATE_EPOCHS   = 4
MINI_BATCH_SIZE = 64
_MAX_EPISODES   = 100

# ==== SC2 Dimensions ====
_SCREEN_DIM = 84

# ==== Feature Keys ====
SCREEN_FEATURES = [
    features.SCREEN_FEATURES.unit_type.index,
    features.SCREEN_FEATURES.player_relative.index
]

Transition = namedtuple('Transition', [
    'state', 'action', 'logprob', 'value', 'reward', 'mask'
])

class PPOAgent(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, in_channels, _SCREEN_DIM, _SCREEN_DIM)
        embed_size = self.conv(dummy).shape[-1]
        self.actor = nn.Sequential(
            nn.Linear(embed_size, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_size, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = x / 255.0
        emb = self.conv(x)
        return self.actor(emb), self.critic(emb)
    def act(self, state):
        logits, value = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(1)
    def evaluate(self, states, actions):
        logits, values = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(1)

class RandomAgent(base_agent.BaseAgent):
    def step(self, obs):
        super().step(obs)
        avail = obs.observation.available_actions
        func = np.random.choice(avail)
        args = []
        for arg in actions.FUNCTIONS[func].args:
            if arg.name in ('screen', 'minimap'):
                args.append([np.random.randint(0, _SCREEN_DIM),
                             np.random.randint(0, _SCREEN_DIM)])
            else:
                args.append([0])
        return actions.FunctionCall(func, args)

class Memory:
    def __init__(self): self.data = []
    def push(self, *tr): self.data.append(Transition(*tr))
    def clear(self): self.data = []
    def sample(self): return Transition(*zip(*self.data))

class PPOTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=LR)
        self.memory = Memory()
    def compute_gae(self, next_value, rewards, masks, values):
        gae = 0; returns = []
        vals = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * vals[t+1] * masks[t] - vals[t]
            gae = delta + GAMMA * GAE_LAMBDA * masks[t] * gae
            returns.insert(0, gae + vals[t])
        return returns
    def update(self):
        trans = self.memory.sample()
        states = torch.stack(trans.state)
        actions = torch.tensor(trans.action)
        old_lp = torch.stack(trans.logprob)
        rewards, masks, vals = list(trans.reward), list(trans.mask), list(trans.value)
        returns = torch.tensor(self.compute_gae(0, rewards, masks, vals))
        values = torch.tensor(vals)
        advs = returns - values
        advs = (advs - advs.mean())/(advs.std()+1e-8)
        for _ in range(UPDATE_EPOCHS):
            for i in range(0, len(states), MINI_BATCH_SIZE):
                idx = slice(i, i+MINI_BATCH_SIZE)
                logp, ent, vs = self.agent.evaluate(states[idx], actions[idx])
                ratio = torch.exp(logp - old_lp[idx])
                s1 = ratio * advs[idx]
                s2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advs[idx]
                a_loss = -torch.min(s1, s2).mean()
                c_loss = (returns[idx] - vs).pow(2).mean()
                loss = a_loss + VF_COEF*c_loss - ENT_COEF*ent.mean()
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.memory.clear()
    def train(self, episodes, max_steps=1000):
        for ep in range(1, episodes+1):
            obs = self.env.reset()
            state = self._prep(obs)
            tot_r = 0
            for _ in range(max_steps):
                a, lp, v = self.agent.act(state)
                obs, r, done, _ = self.env.step(a)
                ns = self._prep(obs)
                mask = 0 if done else 1
                self.memory.push(state, a, lp, v, r, mask)
                state = ns; tot_r += r
                if done: break
            self.update()
            print(f"Episode {ep} Reward: {tot_r}")
    def _prep(self, obs):
        layers = [obs.observation.feature_screen[i] for i in SCREEN_FEATURES]
        arr = np.stack(layers, axis=0)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# ---- Entry Point ----
def main(argv):
    env = sc2_env.SC2Env(
        map_name='Simple64',
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Agent(sc2_env.Race.terran)
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=_SCREEN_DIM, minimap=64),
            use_feature_units=False,
            action_space=actions.ActionSpace.FEATURES
        ),
        step_mul=8,
        visualize=False
    )
    ppo = PPOAgent(in_channels=len(SCREEN_FEATURES), num_actions=len(Transition._fields))
    rnd = RandomAgent()
    trainer = PPOTrainer(env, ppo)
    trainer.train(_MAX_EPISODES)
    # run evaluation loop with random opponent
    run_loop.run_loop([ppo, rnd], env, max_episodes=10)

if __name__ == '__main__':
    app.run(main)
