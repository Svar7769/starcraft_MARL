from __future__ import annotations

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from smac.env import StarCraft2Env
from torchrl.modules import MultiAgentMLP
from het_control.snd import compute_behavioral_distance

def render_snd_heatmap(env: StarCraft2Env, model: ActorCritic) -> plt.Figure:
    obs = env.get_obs()
    obs_tensor = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0)
    dev_out = model.agent_mlps(obs_tensor).squeeze(0)
    n_agents = env.get_env_info()['n_agents']
    dist_matrix = torch.zeros((n_agents, n_agents), dtype=torch.float32)
    for i in range(n_agents):
        for j in range(n_agents):
            dist_matrix[i, j] = torch.dist(dev_out[i], dev_out[j]).item()
    fig, ax = plt.subplots()
    im = ax.imshow(dist_matrix.numpy(), interpolation='nearest')
    fig.colorbar(im, ax=ax)
    return fig


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.n_agents = n_agents
        self.shared = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=n_actions,
            n_agents=n_agents,
            centralised=False,
            share_params=True,
            activation_class=nn.ReLU,
            num_cells=hidden,
        )
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=n_actions,
            n_agents=n_agents,
            centralised=False,
            share_params=False,
            activation_class=nn.ReLU,
            num_cells=hidden,
        )
        feature_size = n_agents * n_actions
        self.critic = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor):
        shared_out = self.shared(obs)
        dev = self.agent_mlps(obs)
        logits = shared_out + dev
        critic_input = logits
        value = self.critic(critic_input)
        return logits, value.squeeze(-1), dev

def save_model(model: ActorCritic, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: ActorCritic, path: str):
    model.load_state_dict(torch.load(path))
    model.eval()

def train(env: StarCraft2Env, model: ActorCritic, config: dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    gamma = config['gamma']
    lam = config['gae_lambda']
    rollout = config['rollout_len']
    desired_snd = config['desired_snd']
    diversity_coef = config['diversity_coef']

    total_steps = 0
    obs = env.reset()
    obs = env.get_obs()

    while total_steps < config['target_steps']:
        obs_buf, logp_buf, val_buf, rew_buf, mask_buf = [], [], [], [], []
        for _ in range(rollout):
            obs_t = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0)
            logits, value, dev_out = model(obs_t)
            logits = logits.squeeze(0)
            actions_list, logps = [], []
            for i in range(model.n_agents):
                avail = env.get_avail_agent_actions(i)
                mask = torch.tensor(avail, dtype=torch.bool, device=logits.device)
                logit = logits[i].masked_fill(~mask, -1e9)
                dist = torch.distributions.Categorical(logits=logit)
                a = dist.sample()
                actions_list.append(a.item())
                logps.append(dist.log_prob(a))
            reward, done, _ = env.step(actions_list)
            next_obs = env.get_obs()

            obs_buf.append(obs_t)
            logp_buf.append(torch.stack(logps).sum())
            val_buf.append(value)
            rew_buf.append(torch.tensor([reward], dtype=torch.float32))
            mask_buf.append(torch.tensor([1.0 - done], dtype=torch.float32))

            obs = next_obs
            total_steps += 1
            if done:
                obs = env.reset()
                obs = env.get_obs()

        obs_t = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0)
        _, next_val, _ = model(obs_t)
        val_buf.append(next_val)
        mask_buf.append(torch.tensor([1.0], dtype=torch.float32))

        logp_arr = torch.stack(logp_buf)
        val_arr = torch.cat(val_buf)
        rew_arr = torch.cat(rew_buf)
        mask_arr = torch.cat(mask_buf)

        adv = torch.zeros_like(rew_arr)
        last = 0
        for t in reversed(range(len(rew_arr))):
            nonterm = mask_arr[t]
            delta = rew_arr[t] + gamma * val_arr[t+1] * nonterm - val_arr[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + val_arr[:-1]

        policy_loss = -(logp_arr * adv.detach()).mean()
        value_loss = 0.5 * (ret.detach() - val_arr[:-1]).pow(2).mean()

        last_obs_tensor = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0)
        _, _, dev_out = model(last_obs_tensor)
        behavioral_dist_scalar = compute_behavioral_distance(dev_out.squeeze(0), just_mean=True)
        diversity_loss = (behavioral_dist_scalar - desired_snd).pow(2).mean()

        total_loss = policy_loss + config['vf_coef'] * value_loss + diversity_coef * diversity_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        wandb.log({
            'step': total_steps,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'diversity_loss': diversity_loss.item(),
            # 'behavioral_snd': behavioral_dist_scalar.item(),
            'reward': rew_arr.mean().item()
        })

        fig = render_snd_heatmap(env, model)
        wandb.log({'heatmap': wandb.Image(fig)})

    env.close()
    save_model(model, config.get('save_path', 'model.pt'))

def main():
    config = {
        'lr': 2.5e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'rollout_len': 128,
        'vf_coef': 0.5,
        'desired_snd': 0.0,
        'diversity_coef': 0.4,
        'target_steps': 500_000,
        'save_path': 'model.pt'
    }
    wandb.init(project='smac_dico_ppo', config=config)

    env = StarCraft2Env(map_name='3m')
    info = env.get_env_info()
    model = ActorCritic(
        obs_dim=info['obs_shape'],
        n_agents=info['n_agents'],
        n_actions=info['n_actions'],
        hidden=[64, 64]
    )
    wandb.watch(model)
    train(env, model, config)

    # Load and evaluate on another map
    # env2 = StarCraft2Env(map_name='3m')
    # load_model(model, 'model.pt')
    # evaluate(env2, model)

if __name__ == '__main__':
    main()
