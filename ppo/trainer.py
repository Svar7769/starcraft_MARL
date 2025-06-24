import os
from omegaconf import DictConfig
import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging
from typing import Optional
from models.actor_critic import ActorCritic
from env.sc2_env import SC2EnvsMulti
from utils.actions import preprocess, legal_actions, make_pysc2_call, ACTION_LIST, ACTION_INDEX, safe_coords
from pysc2.lib import actions  # Needed for no_op fallback
import torch.nn as nn

logger = logging.getLogger(__name__)

class PPOTrainer:
    def __init__(self, model, device, cfg: DictConfig):
        """
        Initialize PPOTrainer with Hydra config
        
        Args:
            model: The actor-critic model
            device: Torch device (cuda/cpu)
            cfg: Hydra config object containing:
                - hyperparameters (lr, gamma, etc.)
                - training settings (checkpoint_dir, etc.)
        """
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.current_map = cfg.hyperparameters.map_name
        self.nb_actors = cfg.hyperparameters.nb_actors
        self.envs = self._init_env()
        
        # Training parameters from config
        self.lr = cfg.hyperparameters.lr
        self.gamma = cfg.hyperparameters.gamma
        self.gae_lambda = cfg.hyperparameters.gae_lambda
        self.ent_coef = cfg.hyperparameters.ent_coef
        self.vf_coef = cfg.hyperparameters.vf_coef
        self.max_iters = cfg.hyperparameters.max_iters
        self.T = cfg.hyperparameters.T
        self.K = cfg.hyperparameters.K
        self.batch_size = cfg.hyperparameters.batch_size
        self.clip_param = cfg.hyperparameters.clip_param
        self.save_interval = cfg.training.save_interval
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=1.0, 
            end_factor=0.0, 
            total_iters=self.max_iters
        )
        
        # Create checkpoint directory
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    def _init_env(self):
        """Initialize SC2 environment with current parameters"""
        return SC2EnvsMulti(cfg=self.cfg)

    def train(self, map_name: Optional[str] = None):
        """Run PPO training on current or specified map"""
        if map_name and map_name != self.current_map:
            self.switch_map(map_name)
            
        ep_rewards = []
        logger.info(f"Starting PPO on {self.current_map} for {self.max_iters} iterations")
        
        for it in range(self.max_iters):
            if it % self.save_interval == 0:
                logger.info(f"Iteration {it}/{self.max_iters}")
                self.save_checkpoint(it)
            
            # Storage buffers
            obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = self._init_buffers()
            
            # Rollout phase
            with torch.no_grad():
                for t in range(self.T):
                    for i in range(self.nb_actors):
                        ts = self.envs.obs[i]
                        state = preprocess(ts).to(self.device)
                        logits, value = self.model(state)
                        action, logp = self._select_action(logits, ts)
                        fc, pending = make_pysc2_call(action.item(), ts)
                        # If multi-step, handle pending (not implemented here)
                        ts2 = self.envs.step(i, fc)
                        obs_buf[i, t] = state
                        act_buf[i, t] = action
                        logp_buf[i, t] = logp
                        val_buf[i, t] = value.squeeze()
                        rew_buf[i, t] = torch.tensor(ts2.reward, device=self.device)
                        done_buf[i, t] = float(ts2.last())
                        if ts2.last():
                            ep_rewards.append(sum(rew_buf[i, :t+1].cpu().tolist()))
                            self.envs.reset(i)
                # Bootstrap final value
                self._bootstrap_values(val_buf)
            
            # Compute advantages and update
            adv_buf = self._compute_advantages(rew_buf, val_buf, done_buf)
            self._update_policy(obs_buf, act_buf, logp_buf, val_buf, adv_buf)
            self.scheduler.step()
        
        # Final save and cleanup
        self.save_checkpoint(self.max_iters, final=True)
        self._plot_results(ep_rewards)
        self.envs.close()
        logger.info(f"Training complete on {self.current_map}")

    def _init_buffers(self):
        """Initialize empty rollout buffers"""
        obs_buf = torch.zeros(self.nb_actors, self.T, 2, 84, 84, device=self.device)
        act_buf = torch.zeros(self.nb_actors, self.T, dtype=torch.long, device=self.device)
        logp_buf = torch.zeros(self.nb_actors, self.T, device=self.device)
        val_buf = torch.zeros(self.nb_actors, self.T+1, device=self.device)
        rew_buf = torch.zeros(self.nb_actors, self.T, device=self.device)
        done_buf = torch.zeros(self.nb_actors, self.T, device=self.device)
        return obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf

    def _select_action(self, logits, ts):
        """Select action with legal action masking"""
        la = legal_actions(ts)
        mask = torch.full_like(logits, float('-inf'))
        mask[0, la] = 0.0
        dist = Categorical(logits=logits + mask)
        action = dist.sample()
        return action, dist.log_prob(action)

    def _bootstrap_values(self, val_buf):
        """Bootstrap final value estimates"""
        for i in range(self.nb_actors):
            val_buf[i, self.T] = self.model(
                preprocess(self.envs.obs[i]).to(self.device))[1].squeeze()

    def _compute_advantages(self, rew_buf, val_buf, done_buf):
        """Compute GAE advantages"""
        adv_buf = torch.zeros_like(rew_buf)
        for i in range(self.nb_actors):
            gae = 0
            for t in reversed(range(self.T)):
                mask = 1.0 - done_buf[i, t]
                delta = rew_buf[i, t] + self.gamma * val_buf[i, t+1] * mask - val_buf[i, t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                adv_buf[i, t] = gae
        return adv_buf

    def _update_policy(self, obs_buf, act_buf, logp_buf, val_buf, adv_buf):
        """Perform PPO policy updates"""
        # Flatten buffers
        b_s = obs_buf.reshape(-1, 2, 84, 84)
        b_a = act_buf.reshape(-1)
        b_lp = logp_buf.reshape(-1)
        b_v = val_buf[:, :self.T].reshape(-1)
        b_ad = adv_buf.reshape(-1)
        
        # Normalize advantages
        b_ad = (b_ad - b_ad.mean()) / (b_ad.std() + 1e-8)
        
        # Create dataset
        ds = TensorDataset(b_s, b_a, b_lp, b_v, b_ad)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        
        # PPO updates
        for _ in range(self.K):
            for batch in loader:
                s, a, old_lp, old_v, adv = batch
                logits, val = self.model(s)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a)
                ratio = torch.exp(lp - old_lp)
                obj1 = adv * ratio
                obj2 = adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                policy_loss = -torch.min(obj1, obj2).mean()
                v_loss = 0.5 * (val.squeeze() - (adv + old_v)).pow(2).mean()
                loss = (policy_loss 
                        + self.vf_coef * v_loss 
                        - self.ent_coef * dist.entropy().mean())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def switch_map(self, new_map: str):
        """Switch to training on a different map"""
        self.envs.close()
        self.current_map = new_map
        self.envs = self._init_env()
        logger.info(f"Switched to map: {new_map}")

    def save_checkpoint(self, iteration: int, final: bool = False):
        """Save training state"""
        suffix = "final" if final else f"iter{iteration}"
        path = os.path.join(self.cfg.training.checkpoint_dir, f"ppo_{self.current_map}_{suffix}.pt")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'iteration': iteration,
            'map_name': self.current_map,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'clip_param': self.clip_param,
                'nb_actors': self.nb_actors
            }
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_from_checkpoint(cls, path: str, device, cfg: DictConfig):
        """Load trainer from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        model = ActorCritic(
            in_channels=2,
            nb_actions=len(ACTION_LIST),
            screen_size=84
        )
        model.load_state_dict(checkpoint['model_state'])
        
        trainer = cls(
            model=model,
            device=device,
            cfg=cfg
        )
        
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        hp = checkpoint['hyperparameters']
        trainer.lr = hp['lr']
        trainer.gamma = hp['gamma']
        trainer.gae_lambda = hp['gae_lambda']
        trainer.ent_coef = hp['ent_coef']
        trainer.vf_coef = hp['vf_coef']
        trainer.clip_param = hp['clip_param']
        
        logger.info(f"Loaded checkpoint from {path} (iter {checkpoint['iteration']})")
        return trainer

    def _plot_results(self, ep_rewards):
        """Plot training results"""
        if not ep_rewards:
            logger.warning("No episodes completed during training")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(ep_rewards, label="Episode Reward")
        plt.title(f"Training Progress on {self.current_map}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        
        plot_path = os.path.join(self.cfg.training.checkpoint_dir, f"training_{self.current_map}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved training plot to {plot_path}")