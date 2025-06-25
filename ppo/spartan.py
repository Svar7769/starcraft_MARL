import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Normal # For continuous PPO
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt # Still useful for individual plots if needed

# Import common PPO hyperparameters from config
from config.spartan import (
    GAMMA, GAE_LAMBDA, K, BATCH_SIZE, ENT_COEF, VF_COEF, KL_COEF, logger
)

def calculate_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    Calculates Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (torch.Tensor): Tensor of rewards for the rollout.
        values (torch.Tensor): Tensor of value predictions for states in the rollout.
        dones (torch.Tensor): Tensor of done flags for states in the rollout.
        next_value (torch.Tensor): Value prediction for the state after the rollout.
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda parameter.
        
    Returns:
        torch.Tensor: Tensor of advantages.
        torch.Tensor: Tensor of returns (rewards + discounted future rewards).
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    
    # Append next_value to values for GAE calculation
    full_values = torch.cat((values, next_value.unsqueeze(0)), dim=0)

    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * full_values[t + 1] * (1.0 - dones[t]) - full_values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
        
    returns = advantages + values
    return advantages, returns


def ppo_update_discrete(model, optimizer, scheduler, data_loader, clip_param, it_current, it_max, action_list_len):
    """
    Performs PPO update for a discrete action policy.
    
    Args:
        model (nn.Module): The ActorCritic model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        data_loader (DataLoader): DataLoader providing batches of (obs, action, old_log_prob, old_logits, old_value, advantage, returns).
        clip_param (float): PPO clipping parameter.
        it_current (int): Current global iteration for adaptive clipping.
        it_max (int): Max global iterations for adaptive clipping.
        action_list_len (int): Length of the action list for old_logits_buf shape.
    """
    for _ in range(K): # K PPO epochs
        for obs_batch, act_batch, old_logp_batch, old_logits_batch, old_val_batch, adv_batch, ret_batch in data_loader:
            
            # Forward pass to get new logits and values
            # Assumes model is KingActorCritic, needing obs and goal.
            # Here, `obs_batch` is expected to be a tuple (spatial_obs, controller_goal_batch)
            if isinstance(obs_batch, tuple):
                logits, val = model(obs_batch[0], obs_batch[1]) # Pass obs_features and controller_goal
            else: # Fallback for simpler models or if goal is embedded in obs
                logits, val = model(obs_batch)

            dist_new = Categorical(logits=logits)
            logp_new = dist_new.log_prob(act_batch)
            
            # Policy Loss
            ratio = torch.exp(logp_new - old_logp_batch)
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
            p_loss = -torch.min(ratio * adv_batch, clipped_ratio * adv_batch).mean()
            
            # Value Loss (clipped)
            val_clipped = old_val_batch + torch.clamp(val - old_val_batch, -clip_param, clip_param)
            v_loss1 = (val - ret_batch).pow(2)
            v_loss2 = (val_clipped - ret_batch).pow(2)
            v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean() # 0.5 is VF_COEF here

            # Entropy Bonus
            entropy = dist_new.entropy().mean()

            # KL Divergence Penalty (against old policy that collected data)
            dist_old = Categorical(logits=old_logits_batch)
            kl_divergence = torch.distributions.kl_divergence(dist_old, dist_new).mean()
            kl_divergence = torch.nan_to_num(kl_divergence, nan=0.0, posinf=0.0, neginf=0.0)

            # Total Loss
            # PPO paper recommends P_loss + C1*V_loss - C2*Entropy.
            # Your code uses VF_COEF * V_loss - ENT_COEF * Entropy.
            loss = p_loss + VF_COEF * v_loss - ENT_COEF * entropy + KL_COEF * kl_divergence

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Max grad norm of 0.5
            optimizer.step()
    
    scheduler.step() # Step scheduler once per PPO update (after all K epochs)

def ppo_update_continuous(model, optimizer, scheduler, data_loader, clip_param, it_current, it_max, goal_dim):
    """
    Performs PPO update for a continuous action policy (e.g., Controller).
    
    Args:
        model (nn.Module): The ActorCritic model (ControllerActorCritic).
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        data_loader (DataLoader): DataLoader providing batches of (obs, action_mean, old_log_prob, old_log_std, old_value, advantage, returns).
        clip_param (float): PPO clipping parameter.
        it_current (int): Current global iteration for adaptive clipping.
        it_max (int): Max global iterations for adaptive clipping.
        goal_dim (int): Dimension of the continuous goal vector.
    """
    for _ in range(K): # K PPO epochs
        for obs_batch, act_mean_batch, old_logp_batch, old_log_std_batch, old_val_batch, adv_batch, ret_batch in data_loader:
            
            # Forward pass to get new mean, log_std, and value
            new_mean, new_log_std, new_val = model(obs_batch)
            
            # Policy Loss
            # Create new distribution from current policy output
            new_dist = Normal(new_mean, new_log_std.exp())
            logp_new = new_dist.log_prob(act_mean_batch).sum(axis=-1) # Sum log_probs across dimensions for multi-dim actions
            
            ratio = torch.exp(logp_new - old_logp_batch)
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
            p_loss = -torch.min(ratio * adv_batch, clipped_ratio * adv_batch).mean()
            
            # Value Loss (clipped)
            val_clipped = old_val_batch + torch.clamp(new_val - old_val_batch, -clip_param, clip_param)
            v_loss1 = (new_val - ret_batch).pow(2)
            v_loss2 = (val_clipped - ret_batch).pow(2)
            v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            # Entropy Bonus (for continuous actions)
            entropy = new_dist.entropy().sum(axis=-1).mean() # Sum entropy across dimensions
            
            # KL Divergence Penalty (against old policy distribution)
            # Need old_mean and old_log_std stored in buffer. For simplicity here, use old_logp_batch and old_log_std_batch
            # A more complete implementation would store old_mean too.
            # Here, we use the property that KL(N1 || N2) = sum(log(std2/std1) + (std1^2 + (mean1-mean2)^2)/(2*std2^2) - 0.5)
            # Simpler: just use the old_log_prob for ratio, and entropy for exploration.
            # The KL penalty for continuous PPO can be based on exact KL between Normal distributions.
            
            # For simplicity for this function, if KL_COEF is used with Normal, ensure old_mean is also passed
            # For now, let's omit explicit KL term for continuous PPO (as it's often more about entropy).
            # If `KL_COEF` is intended for categorical divergence, it might not apply directly to Normal dist.
            # You could add KL between `Normal(new_mean, new_std)` and `Normal(old_mean, old_std)`.
            kl_divergence = torch.tensor(0.0, device=DEVICE) # Placeholder if not using explicit KL for continuous

            # Total Loss
            loss = p_loss + VF_COEF * v_loss - ENT_COEF * entropy + KL_COEF * kl_divergence

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    
    scheduler.step() # Step scheduler once per PPO update (after all K epochs)

