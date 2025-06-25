import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from pysc2.lib import actions
from config.config import (
    SCREEN_SIZE, DEVICE, NB_ACTORS, T, K, BATCH_SIZE,
    GAMMA, GAE_LAMBDA, LR, ENT_COEF, VF_COEF, MAX_ITERS, KL_COEF, logger # Import KL_COEF
)
from models.actor_critic import ActorCritic
from utils.utils import preprocess, legal_actions, make_pysc2_call, ACTION_LIST

def train_ppo(envs, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERS
    )

    ep_rewards = []
    
    pending_action = [None] * envs.nb

    logger.info("▶️  Starting PPO for %d iterations", MAX_ITERS)
    for it in range(MAX_ITERS):
        if it % 1000 == 0:
            logger.info("🔄 Iter %d / %d", it, MAX_ITERS)

        # storage buffers
        obs_buf        = torch.zeros(envs.nb, T, 2, SCREEN_SIZE, SCREEN_SIZE, device=DEVICE)
        act_buf        = torch.zeros(envs.nb, T,      dtype=torch.long, device=DEVICE)
        logp_buf       = torch.zeros(envs.nb, T,                     device=DEVICE)
        # --- MODIFIED: Add buffer for old_logits ---
        old_logits_buf = torch.zeros(envs.nb, T, len(ACTION_LIST), device=DEVICE) # Store logits from policy that collected data
        val_buf        = torch.zeros(envs.nb, T+1,                   device=DEVICE)
        rew_buf        = torch.zeros(envs.nb, T,                     device=DEVICE)
        done_buf       = torch.zeros(envs.nb, T,                     device=DEVICE)
        adv_buf        = torch.zeros(envs.nb, T,                     device=DEVICE)
        
        last_cumulative_score = [ts.observation["score_cumulative"][0] for ts in envs.obs]


        # ─── Rollout ─────────────────────────────────────────────────────────
        with torch.no_grad():
            for t in range(T):
                for i in range(envs.nb):
                    ts    = envs.obs[i]
                    state = preprocess(ts).to(DEVICE)
                    
                    logits, value = model(state)

                    LA   = legal_actions(ts)
                    mask = torch.full_like(logits, float('-inf'))
                    mask[0, LA] = 0.0
                    dist = Categorical(logits=logits + mask)

                    action = dist.sample()
                    logp   = dist.log_prob(action)
                    
                    # --- MODIFIED: Store old_logits ---
                    old_logits_buf[i,t] = logits.squeeze(0) # Store the current logits as 'old' for next update cycle

                    pysc2_action_call, pending_action[i] = make_pysc2_call(action.item(), ts, pending_action[i])

                    try:
                        ts2 = envs.step(i, pysc2_action_call)
                    except ValueError as e:
                        logger.warning(f"Error stepping env {i} with action {pysc2_action_call}: {e}. Falling back to no-op.")
                        ts2 = envs.step(i, actions.FunctionCall(actions.FUNCTIONS.no_op.id, []))
                        pending_action[i] = None

                    current_cumulative_score = ts2.observation["score_cumulative"][0]
                    r = current_cumulative_score - last_cumulative_score[i]
                    last_cumulative_score[i] = current_cumulative_score
                    
                    d = float(ts2.last())

                    obs_buf[i,t]  = state.squeeze(0)
                    act_buf[i,t]  = action
                    logp_buf[i,t] = logp
                    val_buf[i,t]  = value
                    rew_buf[i,t]  = r
                    done_buf[i,t] = d

                    if d:
                        ep_rewards.append(sum(rew_buf[i, :t+1].tolist()))
                        envs.reset(i)
                        last_cumulative_score[i] = envs.obs[i].observation["score_cumulative"][0]
                        pending_action[i] = None

            for i in range(envs.nb):
                final_state = preprocess(envs.obs[i]).to(DEVICE)
                val_buf[i,T] = model(final_state)[1]

        # ─── GAE & flatten ────────────────────────────────────────────────────
        for i in range(envs.nb):
            gae = 0
            for t in reversed(range(T)):
                mask  = 1.0 - done_buf[i,t]
                delta = rew_buf[i,t] + GAMMA*val_buf[i,t+1]*mask - val_buf[i,t]
                gae   = delta + GAMMA*GAE_LAMBDA*mask*gae
                adv_buf[i,t] = gae

        b_s  = obs_buf.reshape(-1,2,SCREEN_SIZE,SCREEN_SIZE)
        b_a  = act_buf.reshape(-1)
        b_lp = logp_buf.reshape(-1)
        # --- MODIFIED: Flatten old_logits_buf ---
        b_ol = old_logits_buf.reshape(-1, len(ACTION_LIST)) # Flatten old_logits
        b_v  = val_buf[:,:T].reshape(-1)
        b_ad = adv_buf.reshape(-1)

        # ─── PPO updates ─────────────────────────────────────────────────────
        for _ in range(K):
            # --- MODIFIED: Include old_logits in TensorDataset ---
            ds     = TensorDataset(b_s,b_a,b_lp,b_ol,b_v,b_ad)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            # --- MODIFIED: Extract old_logits from batch ---
            for st, ac, old_lp, old_ol, old_v, adv in loader:
                logits, val = model(st)
                dist_new    = Categorical(logits=logits)
                lp          = dist_new.log_prob(ac)
                ratio       = torch.exp(lp - old_lp)

                clip   = 0.1 * (1 - it/MAX_ITERS)
                obj1   = adv * ratio
                obj2   = adv * torch.clamp(ratio, 1-clip, 1+clip)
                p_loss = -torch.min(obj1,obj2).mean()

                ret     = adv + old_v
                v1      = (val - ret).pow(2)
                v2      = (torch.clamp(val,old_v-clip,old_v+clip)-ret).pow(2)
                v_loss  = 0.5 * torch.max(v1,v2).mean()

                entropy = dist_new.entropy().mean()
                
                # --- NEW: KL Divergence Penalty ---
                dist_old = Categorical(logits=old_ol)
                kl_divergence = torch.distributions.kl_divergence(dist_old, dist_new).mean()
                # Ensure KL_divergence is not NaN or inf, often clip or add epsilon
                kl_divergence = torch.nan_to_num(kl_divergence, nan=0.0, posinf=0.0, neginf=0.0)

                loss    = p_loss + VF_COEF*v_loss - ENT_COEF*entropy + KL_COEF*kl_divergence # Add KL penalty

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),0.5)
                optimizer.step()

        scheduler.step()

    # --- Plot learning curve ───────────────────────────────────────────────────
    plt.figure(figsize=(10,5))
    plt.plot(ep_rewards, label="episode reward")
    plt.title("Environment Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig("learning_curve.png")
    plt.show()

    envs.close()
    logger.info("✅ Training complete")
    logger.info(f"Saved learning_curve.png over {len(ep_rewards)} episodes")