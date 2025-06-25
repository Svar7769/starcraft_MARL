import sys
import os
from absl import flags # For PySC2 flags fix
import matplotlib.pyplot as plt
from collections import deque # For rich table (if used for demo/logging)
import random # For random choices in action mapping
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
from torch.distributions import Normal # For Controller's continuous action space

# Fix for absl.flags in Jupyter or script context (must be at the very top level)
flags.FLAGS(sys.argv, known_only=True)

# Import components from your modularized files
from config.spartan import (
    NB_ACTORS, DEVICE, MAX_ITERS, logger, SCREEN_SIZE, MINIMAP_SIZE, STEP_MUL,
    CONTROLLER_GOAL_DIM, CONTROLLER_ACTION_FREQ, CONTROLLER_LR, CONTROLLER_ROLLOUT_LENGTH,
    KING_LR, KING_ROLLOUT_LENGTH,
    K, BATCH_SIZE, GAMMA, GAE_LAMBDA, ENT_COEF, VF_COEF, KL_COEF
)
from env.sc2_env import SC2Envs
from models.kings import KingActorCritic, ControllerActorCritic
from ppo.spartan import calculate_gae, ppo_update_discrete, ppo_update_continuous # Import PPO helpers
from utils.preprocessing import preprocess_king_obs, preprocess_controller_obs, get_controller_obs_dim
from utils.action_mapping import make_pysc2_action_call, get_legal_actions_for_policy, COMBAT_ACTION_LIST, RESOURCE_ACTION_LIST

def main_train_hierarchical(_):
    # --- 1. Setup Environments ---
    envs = SC2Envs(NB_ACTORS)

    # --- 2. Setup Policies ---
    # Combat King
    combat_model = KingActorCritic(in_channels=3, nb_actions=len(COMBAT_ACTION_LIST), 
                                   controller_goal_dim=CONTROLLER_GOAL_DIM).to(DEVICE)
    combat_optimizer = optim.Adam(combat_model.parameters(), lr=KING_LR)
    combat_scheduler = optim.lr_scheduler.LinearLR(combat_optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERS // KING_ROLLOUT_LENGTH)

    # Resource King
    resource_model = KingActorCritic(in_channels=3, nb_actions=len(RESOURCE_ACTION_LIST), 
                                     controller_goal_dim=CONTROLLER_GOAL_DIM).to(DEVICE)
    resource_optimizer = optim.Adam(resource_model.parameters(), lr=KING_LR)
    resource_scheduler = optim.lr_scheduler.LinearLR(resource_optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERS // KING_ROLLOUT_LENGTH)

    # Controller Policy (Master)
    controller_obs_dim = get_controller_obs_dim()
    controller_model = ControllerActorCritic(obs_dim=controller_obs_dim, goal_dim=CONTROLLER_GOAL_DIM).to(DEVICE)
    controller_optimizer = optim.Adam(controller_model.parameters(), lr=CONTROLLER_LR)
    controller_scheduler = optim.lr_scheduler.LinearLR(controller_optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERS // CONTROLLER_ROLLOUT_LENGTH)
    
    # --- 3. Data Buffers for PPO Rollouts (per environment, then concatenated) ---
    # Combat King Buffers
    combat_obs_buf = [] # Stores tuple (spatial_obs, controller_goal)
    combat_action_buf = []
    combat_logp_buf = []
    combat_old_logits_buf = [] # For KL divergence for discrete actions
    combat_val_buf = []
    combat_rew_buf = []
    combat_done_buf = []

    # Resource King Buffers
    resource_obs_buf = [] # Stores tuple (spatial_obs, controller_goal)
    resource_action_buf = []
    resource_logp_buf = []
    resource_old_logits_buf = []
    resource_val_buf = []
    resource_rew_buf = []
    resource_done_buf = []

    # Controller Buffers (for continuous action PPO)
    controller_obs_buf = []
    controller_action_mean_buf = [] # Stored mean of the goal sampled
    controller_logp_buf = []
    controller_old_log_std_buf = [] # For continuous KL (or just old_logp for ratio)
    controller_val_buf = []
    controller_rew_buf = []
    controller_done_buf = []

    # --- 4. Training State & Logging ---
    combat_ep_rewards = []
    resource_ep_rewards = []
    controller_ep_rewards = [] # Overall game reward for controller

    # Track last cumulative score for reward calculation per environment
    last_cumulative_score_per_env = [ts.observation["score_cumulative"][0] for ts in envs.obs]

    # Keep track of pending two-step actions for each King, per environment
    # Dictionary mapping env_idx to another dict: {'combat': None, 'resource': None}
    pending_actions_per_env = {i: {"combat": None, "resource": None} for i in range(envs.nb)}

    # Controller's current active goal for Kings (updated periodically)
    # Initialize with a zero goal
    current_controller_goal = torch.zeros(1, CONTROLLER_GOAL_DIM).to(DEVICE)
    # Track when the last controller action was taken
    last_controller_action_step = -1

    logger.info("▶️  Starting Hierarchical PPO Training...")
    
    # --- 5. Main Training Loop ---
    for global_step in range(MAX_ITERS):
        # Update Controller's goal periodically
        if (global_step - last_controller_action_step) >= CONTROLLER_ACTION_FREQ:
            # Get current controller observation
            ts = envs.obs[0] # Assuming NB_ACTORS=1 for simplicity for now
            controller_current_obs = preprocess_controller_obs(ts).to(DEVICE)

            # Controller takes an action (generates a goal vector)
            with torch.no_grad():
                goal_mean, goal_log_std, controller_value = controller_model(controller_current_obs)
                
                # Sample from Normal distribution for actual goal
                controller_dist = Normal(goal_mean, goal_log_std.exp())
                sampled_controller_goal = controller_dist.sample()
                controller_logp = controller_dist.log_prob(sampled_controller_goal).sum(axis=-1)

            # Store data for Controller's PPO update
            controller_obs_buf.append(controller_current_obs.squeeze(0).cpu())
            controller_action_mean_buf.append(sampled_controller_goal.squeeze(0).cpu())
            controller_logp_buf.append(controller_logp.item())
            controller_old_log_std_buf.append(goal_log_std.squeeze(0).cpu()) # Store old log_std for KL
            controller_val_buf.append(controller_value.item())
            
            # The 'reward' for the controller will be accumulated global reward until its next action
            # This is handled later when the environment steps.
            
            current_controller_goal = sampled_controller_goal.detach() # Detach for King's inputs
            last_controller_action_step = global_step
            logger.debug(f"Controller new goal: {current_controller_goal.cpu().numpy()}")


        # --- Kings' Action Selection and Environment Step ---
        # For simplicity, we'll iterate through envs (here, just one)
        # and then decide which King's action to execute.
        
        for i in range(envs.nb):
            ts = envs.obs[i] # Current timestep from environment
            
            # --- Combat King Action ---
            combat_current_obs = preprocess_king_obs(ts).to(DEVICE)
            with torch.no_grad():
                combat_logits, combat_value = combat_model(combat_current_obs, current_controller_goal)
                combat_legal_actions = get_legal_actions_for_policy(ts, "combat")
                combat_mask = torch.full_like(combat_logits, float('-inf'))
                combat_mask[0, combat_legal_actions] = 0.0
                combat_dist = Categorical(logits=combat_logits + combat_mask)
                combat_action = combat_dist.sample()
                combat_logp = combat_dist.log_prob(combat_action)
            
            # Store data for Combat King's PPO update
            combat_obs_buf.append((combat_current_obs.squeeze(0).cpu(), current_controller_goal.squeeze(0).cpu()))
            combat_action_buf.append(combat_action.item())
            combat_logp_buf.append(combat_logp.item())
            combat_old_logits_buf.append(combat_logits.squeeze(0).cpu())
            combat_val_buf.append(combat_value.item())

            combat_pysc2_action_call, pending_actions_per_env[i]["combat"] = \
                make_pysc2_action_call(combat_action.item(), ts, "combat", pending_actions_per_env[i]["combat"])

            # --- Resource King Action ---
            resource_current_obs = preprocess_king_obs(ts).to(DEVICE)
            with torch.no_grad():
                resource_logits, resource_value = resource_model(resource_current_obs, current_controller_goal)
                resource_legal_actions = get_legal_actions_for_policy(ts, "resource")
                resource_mask = torch.full_like(resource_logits, float('-inf'))
                resource_mask[0, resource_legal_actions] = 0.0
                resource_dist = Categorical(logits=resource_logits + resource_mask)
                resource_action = resource_dist.sample()
                resource_logp = resource_dist.log_prob(resource_action)
            
            # Store data for Resource King's PPO update
            resource_obs_buf.append((resource_current_obs.squeeze(0).cpu(), current_controller_goal.squeeze(0).cpu()))
            resource_action_buf.append(resource_action.item())
            resource_logp_buf.append(resource_logp.item())
            resource_old_logits_buf.append(resource_logits.squeeze(0).cpu())
            resource_val_buf.append(resource_value.item())

            resource_pysc2_action_call, pending_actions_per_env[i]["resource"] = \
                make_pysc2_action_call(resource_action.item(), ts, "resource", pending_actions_per_env[i]["resource"])

            # --- Decide which action to execute in the environment ---
            # For this sanity test, let's alternate execution between combat and resource actions
            # Or prioritize based on game state / controller output (more advanced)
            # Simple alternating:
            if global_step % 2 == 0: # Execute Combat King's action on even steps
                action_to_execute = combat_pysc2_action_call
            else: # Execute Resource King's action on odd steps
                action_to_execute = resource_pysc2_action_call

            # Execute action in the environment
            current_episode_global_score = ts.observation["score_cumulative"][0]
            try:
                ts2 = envs.step(i, action_to_execute)
            except ValueError as e:
                logger.warning(f"Env {i} step error: {e}. Action: {action_to_execute}. Falling back to no_op.")
                ts2 = envs.step(i, actions.FunctionCall(actions.FUNCTIONS.no_op.id, []))
                pending_actions_per_env[i]["combat"] = None # Clear pending for safety
                pending_actions_per_env[i]["resource"] = None
            
            step_reward = ts2.observation["score_cumulative"][0] - current_episode_global_score

            # --- Update Buffers with Rewards and Dones ---
            # Controller reward (global reward)
            controller_rew_buf.append(step_reward)
            controller_done_buf.append(float(ts2.last()))

            # Kings' rewards (can be global or shaped)
            combat_rew_buf.append(step_reward)
            combat_done_buf.append(float(ts2.last()))
            resource_rew_buf.append(step_reward)
            resource_done_buf.append(float(ts2.last()))


            if ts2.last(): # Episode ended for this environment
                logger.info(f"Episode ended (Step {global_step}). Global Reward: {sum(controller_rew_buf[-global_step:]) if controller_rew_buf else 0:.2f}") # Log reward
                
                # Append episode rewards for plotting
                # These sum over the last full episode's rewards
                # Need to be careful with indexing if episodes end mid-rollout.
                # A robust way is to sum rewards from `rew_buf` up to the point of 'done'.
                
                # For simplicity, append the total cumulative score difference for the episode.
                combat_ep_rewards.append(sum(combat_rew_buf[-(global_step - last_controller_action_step if global_step > 0 else 1):])) # Sum from last controller action if possible
                resource_ep_rewards.append(sum(resource_rew_buf[-(global_step - last_controller_action_step if global_step > 0 else 1):]))
                controller_ep_rewards.append(sum(controller_rew_buf[-(global_step - last_controller_action_step if global_step > 0 else 1):])) # Global reward

                envs.reset(i) # Reset the environment
                last_cumulative_score_per_env[i] = envs.obs[i].observation["score_cumulative"][0]
                pending_actions_per_env[i] = {"combat": None, "resource": None} # Reset pending actions

        # --- 6. PPO Updates (triggered when rollouts are full) ---

        # Combat King Update
        # Only update if enough data collected for a full rollout
        if len(combat_obs_buf) >= NB_ACTORS * KING_ROLLOUT_LENGTH:
            logger.info("Updating Combat King...")
            # Prepare data for DataLoader
            # combat_obs_buf stores (spatial_obs, controller_goal)
            s_obs_batch_combat = torch.stack([item[0] for item in combat_obs_buf]).to(DEVICE)
            g_obs_batch_combat = torch.stack([item[1] for item in combat_obs_buf]).to(DEVICE)
            
            b_a_combat = torch.tensor(combat_action_buf, dtype=torch.long, device=DEVICE)
            b_lp_combat = torch.tensor(combat_logp_buf, dtype=torch.float, device=DEVICE)
            b_ol_combat = torch.stack(combat_old_logits_buf).to(DEVICE)
            b_v_combat = torch.tensor(combat_val_buf, dtype=torch.float, device=DEVICE)
            b_r_combat = torch.tensor(combat_rew_buf, dtype=torch.float, device=DEVICE)
            b_d_combat = torch.tensor(combat_done_buf, dtype=torch.float, device=DEVICE)

            # Calculate GAE for Combat King
            with torch.no_grad():
                final_king_obs = preprocess_king_obs(envs.obs[0]).to(DEVICE) # Current obs after rollout
                final_controller_goal = controller_model(preprocess_controller_obs(envs.obs[0]).to(DEVICE))[0].detach() # Current goal
                final_val_combat = combat_model(final_king_obs, final_controller_goal)[1] # Value of final state
            
            combat_adv, combat_ret = calculate_gae(
                rewards=b_r_combat, values=b_v_combat, dones=b_d_combat,
                next_value=final_val_combat, gamma=GAMMA, gae_lambda=GAE_LAMBDA
            )
            
            # Ensure correct size for PPO update data
            data_to_load = (s_obs_batch_combat, g_obs_batch_combat, b_a_combat, b_lp_combat, b_ol_combat, b_v_combat, combat_adv, combat_ret)
            # Create a custom dataset that returns a tuple of (spatial_obs, controller_goal) for obs
            # And then action, logp, old_logits, val, adv, ret
            
            # Need to create a custom TensorDataset or adapt ppo_update_discrete
            # Current `ppo_update_discrete` expects `obs_batch` to be a tuple `(obs_features, controller_goal_batch)`
            # So, create a custom TensorDataset:
            ds_combat = TensorDataset(
                (s_obs_batch_combat, g_obs_batch_combat), # Group these as the 'obs_batch' tuple
                b_a_combat, b_lp_combat, b_ol_combat, b_v_combat, combat_adv, combat_ret
            )
            loader_combat = DataLoader(ds_combat, batch_size=BATCH_SIZE, shuffle=True)
            
            clip_param_current = 0.1 * (1 - global_step / MAX_ITERS) # Adaptive clipping
            ppo_update_discrete(combat_model, combat_optimizer, combat_scheduler, loader_combat, 
                                clip_param_current, global_step, MAX_ITERS, len(COMBAT_ACTION_LIST))
            
            # Clear buffers after update
            combat_obs_buf.clear()
            combat_action_buf.clear()
            combat_logp_buf.clear()
            combat_old_logits_buf.clear()
            combat_val_buf.clear()
            combat_rew_buf.clear()
            combat_done_buf.clear()

        # Resource King Update (similar logic)
        if len(resource_obs_buf) >= NB_ACTORS * KING_ROLLOUT_LENGTH:
            logger.info("Updating Resource King...")
            s_obs_batch_resource = torch.stack([item[0] for item in resource_obs_buf]).to(DEVICE)
            g_obs_batch_resource = torch.stack([item[1] for item in resource_obs_buf]).to(DEVICE)

            b_a_resource = torch.tensor(resource_action_buf, dtype=torch.long, device=DEVICE)
            b_lp_resource = torch.tensor(resource_logp_buf, dtype=torch.float, device=DEVICE)
            b_ol_resource = torch.stack(resource_old_logits_buf).to(DEVICE)
            b_v_resource = torch.tensor(resource_val_buf, dtype=torch.float, device=DEVICE)
            b_r_resource = torch.tensor(resource_rew_buf, dtype=torch.float, device=DEVICE)
            b_d_resource = torch.tensor(resource_done_buf, dtype=torch.float, device=DEVICE)

            with torch.no_grad():
                final_king_obs = preprocess_king_obs(envs.obs[0]).to(DEVICE)
                final_controller_goal = controller_model(preprocess_controller_obs(envs.obs[0]).to(DEVICE))[0].detach()
                final_val_resource = resource_model(final_king_obs, final_controller_goal)[1]
            
            resource_adv, resource_ret = calculate_gae(
                rewards=b_r_resource, values=b_v_resource, dones=b_d_resource,
                next_value=final_val_resource, gamma=GAMMA, gae_lambda=GAE_LAMBDA
            )
            
            ds_resource = TensorDataset(
                (s_obs_batch_resource, g_obs_batch_resource), # Group these as the 'obs_batch' tuple
                b_a_resource, b_lp_resource, b_ol_resource, b_v_resource, resource_adv, resource_ret
            )
            loader_resource = DataLoader(ds_resource, batch_size=BATCH_SIZE, shuffle=True)
            
            clip_param_current = 0.1 * (1 - global_step / MAX_ITERS)
            ppo_update_discrete(resource_model, resource_optimizer, resource_scheduler, loader_resource, 
                                clip_param_current, global_step, MAX_ITERS, len(RESOURCE_ACTION_LIST))
            
            resource_obs_buf.clear()
            resource_action_buf.clear()
            resource_logp_buf.clear()
            resource_old_logits_buf.clear()
            resource_val_buf.clear()
            resource_rew_buf.clear()
            resource_done_buf.clear()


        # Controller Agent Update
        if len(controller_obs_buf) >= NB_ACTORS * CONTROLLER_ROLLOUT_LENGTH:
            logger.info("Updating Controller...")
            b_s_controller = torch.stack(controller_obs_buf).to(DEVICE)
            b_a_mean_controller = torch.stack(controller_action_mean_buf).to(DEVICE)
            b_lp_controller = torch.tensor(controller_logp_buf, dtype=torch.float, device=DEVICE)
            b_ols_controller = torch.stack(controller_old_log_std_buf).to(DEVICE) # Old log_std
            b_v_controller = torch.tensor(controller_val_buf, dtype=torch.float, device=DEVICE)
            b_r_controller = torch.tensor(controller_rew_buf, dtype=torch.float, device=DEVICE)
            b_d_controller = torch.tensor(controller_done_buf, dtype=torch.float, device=DEVICE)

            with torch.no_grad():
                final_obs_controller = preprocess_controller_obs(envs.obs[0]).to(DEVICE)
                final_val_controller = controller_model(final_obs_controller)[2] # Index 2 for value
            
            controller_adv, controller_ret = calculate_gae(
                rewards=b_r_controller, values=b_v_controller, dones=b_d_controller,
                next_value=final_val_controller, gamma=GAMMA, gae_lambda=GAE_LAMBDA
            )
            
            ds_controller = TensorDataset(
                b_s_controller, b_a_mean_controller, b_lp_controller, b_ols_controller,
                b_v_controller, controller_adv, controller_ret
            )
            loader_controller = DataLoader(ds_controller, batch_size=BATCH_SIZE, shuffle=True)
            
            clip_param_current = 0.1 * (1 - global_step / MAX_ITERS)
            ppo_update_continuous(controller_model, controller_optimizer, controller_scheduler, loader_controller, 
                                  clip_param_current, global_step, MAX_ITERS, CONTROLLER_GOAL_DIM)
            
            controller_obs_buf.clear()
            controller_action_mean_buf.clear()
            controller_logp_buf.clear()
            controller_old_log_std_buf.clear()
            controller_val_buf.clear()
            controller_rew_buf.clear()
            controller_done_buf.clear()


        if global_step % 100 == 0 and global_step > 0:
            avg_combat_reward = (sum(combat_ep_rewards[-50:]) / len(combat_ep_rewards[-50:])) if combat_ep_rewards else 0
            avg_resource_reward = (sum(resource_ep_rewards[-50:]) / len(resource_ep_rewards[-50:])) if resource_ep_rewards else 0
            avg_controller_reward = (sum(controller_ep_rewards[-50:]) / len(controller_ep_rewards[-50:])) if controller_ep_rewards else 0
            logger.info(f"Step {global_step}/{MAX_ITERS} | Avg Combat R: {avg_combat_reward:.2f} | Avg Resource R: {avg_resource_reward:.2f} | Avg Controller R: {avg_controller_reward:.2f}")

    # --- 7. Finalization ---
    envs.close()
    logger.info("✅ Hierarchical PPO Training complete")
    
    # Plotting at the end
    plt.figure(figsize=(12, 6))
    plt.plot(combat_ep_rewards, label="Combat Agent Episode Reward", alpha=0.7)
    plt.plot(resource_ep_rewards, label="Resource Agent Episode Reward", alpha=0.7)
    plt.plot(controller_ep_rewards, label="Controller Episode Reward (Global)", alpha=0.7, linestyle='--')
    plt.title("Hierarchical Agent Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("hierarchical_learning_curve.png")
    plt.show()

    # Optional: Save trained models
    os.makedirs("./models", exist_ok=True)
    torch.save(combat_model.state_dict(), "./models/combat_king_model.pth")
    torch.save(resource_model.state_dict(), "./models/resource_king_model.pth")
    torch.save(controller_model.state_dict(), "./models/controller_master_model.pth")
    logger.info("Trained models saved.")


if __name__ == "__main__":
    from absl import app
    app.run(main_train_hierarchical)
