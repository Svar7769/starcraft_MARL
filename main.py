import sys
from absl import flags
import matplotlib.pyplot as plt
from collections import deque
import random # Still needed for fallback or random choices if model is not used/loaded
import torch # Needed for model inference
from pysc2.lib import actions, features

# Fix for absl.flags in Jupyter or script context
flags.FLAGS(sys.argv, known_only=True)

# Import components from your modularized files
from config.config import NB_ACTORS, DEVICE, MAX_ITERS, logger, SCREEN_SIZE # Corrected import for config
from env.sc2_env import SC2Envs # Corrected import for env
from models.actor_critic import ActorCritic # Corrected import for model
from ppo.ppo import train_ppo # Corrected import for ppo
from utils.utils import preprocess, legal_actions, make_pysc2_call, ACTION_INDEX, ACTION_LIST # Corrected import for utils

# Optional: Function to run a quick evaluation/demo
# Now, this function can take a model as input for inference
def run_interactive_demo(num_steps=100, model_to_use=None, model_path=None):
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    
    console = Console()
    envs = SC2Envs(nb_actor=1) # Using a single env for interactive demo
    
    model = model_to_use # Use the model passed in, or load one
    if model_path and model is None:
        logger.info(f"Loading model from {model_path} for demo...")
        model = ActorCritic(2, len(ACTION_LIST)).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        logger.info("Model loaded.")
    
    pending_action = [None] * envs.nb
    
    MAX_ROWS = 20
    recent_rows = deque(maxlen=MAX_ROWS)

    episode_score = [0] * envs.nb
    scores = [] # To store scores of completed episodes

    def generate_table():
        table = Table(title=f"SC2 Agent Actions (Last {MAX_ROWS} Steps)", expand=True)
        table.add_column("Step", justify="right")
        table.add_column("Function ID", justify="right")
        table.add_column("Args", justify="left")
        table.add_column("Reward", justify="right")
        table.add_column("Total Episode Reward", justify="right")
        for row in recent_rows:
            table.add_row(*row)
        return table

    logger.info(f"Starting interactive demo for {num_steps} steps...")
    with Live(generate_table(), refresh_per_second=10, console=console, transient=True) as live:
        for step in range(num_steps):
            for i in range(envs.nb):
                ts = envs.obs[i]
                
                current_score_cumulative = ts.observation["score_cumulative"][0]
                
                if pending_action[i]:
                    action_call, pending_action[i] = make_pysc2_call(None, ts, pending_action[i])
                else:
                    legal = legal_actions(ts)
                    if not legal:
                        action_call = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                        action_idx = ACTION_INDEX['do_nothing']
                    elif model is not None:
                        # Use the model to predict action
                        state = preprocess(ts).to(DEVICE)
                        with torch.no_grad():
                            logits, _ = model(state)
                        
                        # Apply legal actions mask
                        mask = torch.full_like(logits, float('-inf'))
                        mask[0, legal] = 0.0
                        dist = torch.distributions.Categorical(logits=logits + mask)
                        action_idx = dist.sample().item() # Get the action index
                        
                        action_call, pending_action[i] = make_pysc2_call(action_idx, ts)
                    else:
                        # Fallback to random if no model is provided
                        action_idx = random.choice(legal)
                        action_call, pending_action[i] = make_pysc2_call(action_idx, ts)
                        
                try:
                    ts2 = envs.step(i, action_call)
                except ValueError as e:
                    logger.warning(f"Error stepping env {i} with action {action_call}: {e}. Falling back to no-op.")
                    ts2 = envs.step(i, actions.FunctionCall(actions.FUNCTIONS.no_op.id, []))
                    pending_action[i] = None # Clear pending action if no-op was forced

                reward_this_step = ts2.observation["score_cumulative"][0] - current_score_cumulative
                episode_score[i] += reward_this_step

                recent_rows.append((str(step), str(action_call.function.name), str(action_call.arguments), f"{reward_this_step:.2f}", f"{episode_score[i]:.2f}"))
                live.update(generate_table())

                if ts2.last():
                    scores.append(episode_score[i])
                    logger.info(f"Episode {len(scores)} finished with total reward: {episode_score[i]:.2f}")
                    episode_score[i] = 0  # reset for next episode
                    envs.reset(i)
                    pending_action[i] = None # Clear pending action on episode end
                    
    envs.close()
    logger.info("Demo finished.")

    # Plot episode scores from demo
    if scores:
        plt.figure(figsize=(10, 4))
        plt.plot(scores, label="Episode Score", marker='o', linewidth=1.5)
        plt.xlabel("Episode")
        plt.ylabel("Total Score")
        plt.title("Agent Score per Episode (Demo)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def main_train(_):
    # Setup environments
    envs = SC2Envs(NB_ACTORS)

    # Setup model
    model = ActorCritic(2, len(ACTION_LIST)).to(DEVICE)
    
    # Start training
    train_ppo(envs, model)

    # Optional: Save the trained model
    torch.save(model.state_dict(), "./models/trained_ppo_model.pth")
    logger.info("Trained model saved to ./models/trained_ppo_model.pth")

if __name__ == "__main__":
    from absl import app
    
    # Option A: Run Training
    app.run(main_train)
    
    # Option B: Run Demo (Choose ONE of these)
    # 1. Run demo with a random agent (as before, but commented out if training is run)
    # run_interactive_demo(num_steps=500) 
    
    # 2. Run demo with a *trained* model (if you uncommented and saved the model in main_train)
    # You would need to ensure the model path is correct
    run_interactive_demo(num_steps=500, model_path="./models/trained_ppo_model.pth")