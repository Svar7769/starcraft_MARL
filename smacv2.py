#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations
from dataclasses import dataclass
from typing import Type, Sequence, Optional

import torch
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm
from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt


def get_het_model(policy):
    """
    Extract the underlying HetControlMlpEmpirical instance from a policy-like wrapper.
    """
    return policy.model


class HetControlMlpEmpirical(nn.Module):
    """
    Empirical DiCo policy model adapted to SMAC (no Benchmarl base).
    """
    def __init__(
        self,
        input_dim: int,
        n_agents: int,
        output_dim: int,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        desired_snd: float,
        probabilistic: bool,
        scale_mapping: Optional[str],
        tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.probabilistic = probabilistic
        self.tau = tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd
        self.process_shared = process_shared

        # Buffers for desired and estimated SND
        self.register_buffer("desired_snd", torch.tensor([desired_snd], dtype=torch.float))
        self.register_buffer("estimated_snd", torch.tensor([float('nan')], dtype=torch.float))

        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None else None
        )

        # Shared MLP processes all agents together
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=input_dim,
            n_agent_outputs=output_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=True,
            activation_class=activation_class,
        )
        # Per-agent networks produce deviations
        agent_outputs = (output_dim // 2) if probabilistic else output_dim
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=input_dim,
            n_agent_outputs=agent_outputs,
            n_agents=n_agents,
            centralised=False,
            share_params=False,
            activation_class=activation_class,
        )

    def forward(
        self,
        tensordict: TensorDictBase,
        agent_index: Optional[int] = None,
        compute_estimate: bool = True,
        update_estimate: bool = True,
    ) -> TensorDictBase:
        """
        tensordict must contain key ("observation") shape [*batch, n_agents, input_dim]
        """
        x = tensordict.get("observation")
        shared_out = self.shared_mlp(x)
        if agent_index is None:
            agent_out = self.agent_mlps(x)
        else:
            agent_out = self.agent_mlps.agent_networks[agent_index](x)

        # Estimate or update SND
        if self.desired_snd > 0 and compute_estimate and self.n_agents > 1:
            dist = self._estimate_snd(x)
            if update_estimate:
                self.estimated_snd[:] = dist.detach()
        else:
            dist = self.estimated_snd

        # Scaling ratio
        if self.desired_snd == 0:
            scaling = 0.0
        elif self.desired_snd < 0 or torch.isnan(dist):
            scaling = 1.0
        else:
            scaling = self.desired_snd / dist

        # Combine outputs
        if self.probabilistic:
            loc, scale = shared_out.chunk(2, -1)
            loc = loc + agent_out * scaling
            out = torch.cat([loc, scale], dim=-1)
        else:
            out = shared_out + agent_out * scaling

        # Logging norms
        out_loc_norm = overflowing_logits_norm(out)
        tensordict.set("logits", out)
        tensordict.set("out_loc_norm", out_loc_norm)
        tensordict.set("estimated_snd", dist.unsqueeze(-1))

        return tensordict

    def _estimate_snd(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute SND estimate given obs: [*batch, n_agents, input_dim]
        """
        agent_actions = [net(obs) for net in self.agent_mlps.agent_networks]
        distance = compute_behavioral_distance(agent_actions, just_mean=True)
        if torch.isnan(self.estimated_snd).any():
            dist = (self.desired_snd if self.bootstrap_from_desired_snd else distance)
        else:
            dist = (1 - self.tau) * self.estimated_snd + self.tau * distance
        return dist


@dataclass
class HetControlMlpEmpiricalConfig:
    input_dim: int
    n_agents: int
    output_dim: int
    activation_class: Type[nn.Module]
    num_cells: Sequence[int]
    desired_snd: float
    probabilistic: bool
    scale_mapping: Optional[str]
    tau: float
    bootstrap_from_desired_snd: bool
    process_shared: bool


# --------------------------------------
# SMAC Integration & Checkpoint Loading
# --------------------------------------

def load_het_model_from_checkpoint(path: str, device: torch.device = torch.device('cpu')) -> HetControlMlpEmpirical:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt['cfg']
    model = HetControlMlpEmpirical(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


class SimplePolicy:
    """Wraps the HetControl model to expose `.model` attribute."""
    def __init__(self, model: HetControlMlpEmpirical):
        self.model = model


def render_snd_heatmap(env: StarCraft2Env, policy: SimplePolicy):
    env_info = env.get_env_info()
    n_agents = env_info['n_agents']
    obs = env.get_obs()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    td = TensorDict({'observation': obs_tensor}, [1])
    td = policy.model(td)
    dist_matrix = compute_behavioral_distance(
        [policy.model.agent_mlps.agent_networks[i](obs_tensor) for i in range(n_agents)],
        just_mean=False
    )
    plt.imshow(dist_matrix.cpu().numpy(), interpolation='nearest')
    plt.title('Behavioral Distance Matrix')
    plt.colorbar()
    plt.show()


def main():
    env = StarCraft2Env(map_name='8m')
    n_agents = env.get_env_info()['n_agents']

    # Load your trained het-control model checkpoint
    model = load_het_model_from_checkpoint('path/to/het_control_checkpoint.pt')
    policy = SimplePolicy(model)

    n_episodes = 10
    for ep in range(n_episodes):
        env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            obs = env.get_obs()
            avail = [env.get_avail_agent_actions(i) for i in range(n_agents)]
            actions = [np.random.choice(np.nonzero(av)[0]) for av in avail]
            reward, terminated, _ = env.step(actions)
            total_reward += reward

        print(f"Episode {ep} reward: {total_reward}")
        render_snd_heatmap(env, policy)

    env.close()


if __name__ == '__main__':
    main()
