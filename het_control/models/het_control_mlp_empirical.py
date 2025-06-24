#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Sequence, Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm


def get_het_model(policy):
    """
    Extract the underlying HetControlMlpEmpirical instance from a policy.
    """
    model = policy.module[0]
    while not isinstance(model, HetControlMlpEmpirical):
        model = model[0]
    return model


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
        tensordict: must contain key ("observation") with shape [*batch, n_agents, input_dim]
        """
        x = tensordict.get("observation")  # [..., n_agents, input_dim]
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
            if self.process_shared:
                loc = loc
            out = torch.cat([loc, scale], dim=-1)
        else:
            out = shared_out + agent_out * scaling
            if self.process_shared:
                out = out

        # Logging norms
        out_loc_norm = overflowing_logits_norm(out)
        tensordict.set("logits", out)
        tensordict.set("out_loc_norm", out_loc_norm)
        tensordict.set("estimated_snd", dist.unsqueeze(-1))

        return tensordict

    def _estimate_snd(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute SND estimate given raw observations.
        obs: [*batch, n_agents, input_dim]
        """
        agent_actions = []
        for net in self.agent_mlps.agent_networks:
            agent_actions.append(net(obs))
        distance = compute_behavioral_distance(agent_actions, just_mean=True)
        # Soft-update:
        if torch.isnan(self.estimated_snd).any():
            dist = (self.desired_snd if self.bootstrap_from_desired_snd else distance)
        else:
            dist = (1 - self.tau) * self.estimated_snd + self.tau * distance
        return dist


@dataclass
class HetControlMlpEmpiricalConfig:
    """
    Config dataclass (optional) for instantiation elsewhere.
    """
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
