import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions
from config.constants import *
from agents.smart_agent import SmartAgent

import logging
import traceback
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

# Suppress absl internal logs
logging.getLogger("absl").setLevel(logging.WARNING)

# Paths
QTABLE_PATH = os.path.join(MODEL_DIR, "q_table_screen.pkl")

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

def get_difficulty(win_rate):
    if win_rate > 0.8:
        return sc2_env.Difficulty.medium
    elif win_rate > 0.6:
        return sc2_env.Difficulty.easy
    else:
        return sc2_env.Difficulty.very_easy

def main(_):
    agent = SmartAgent()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if RESUME_TRAINING and os.path.exists(QTABLE_PATH) and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            start_episode = int(f.read().strip()) + 1
        rewards = pd.read_csv(REWARD_LOG_PATH)['Reward'].tolist() if os.path.exists(REWARD_LOG_PATH) else []
        recent_results = [1 if r > 0 else 0 for r in rewards[-10:]]
        logging.info(f"{Fore.CYAN}📘 Resuming from episode {start_episode} with {len(rewards)} rewards logged.")
    else:
        start_episode = 1
        rewards = []
        recent_results = []
        logging.info(f"{Fore.CYAN}📘 Starting fresh training.")

    # Initialize live plot
    plt.ion()
    fig, ax = plt.subplots()
    reward_line, = ax.plot([], [], label='Total Reward')
    ax.set_title("Live Reward Tracking")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend()

    for episode in range(start_episode, 100):
        win_rate = sum(recent_results[-10:]) / min(len(recent_results), 10) if recent_results else 0.0
        difficulty = get_difficulty(win_rate)

        logging.info(Fore.YELLOW + "=" * 60)
        logging.info(f"{Fore.MAGENTA}🎮 EPISODE {episode:>3} | Difficulty: {difficulty.name.upper()} | Win Rate: {win_rate:.2f}")

        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, difficulty)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_raw_units=True,  # <-- Make sure this is True!
                use_unit_counts=True,
                use_camera_position=True,
                action_space=actions.ActionSpace.RAW  # <-- Use RAW action space
            ),
            step_mul=16,
            visualize=False,
        ) as env:
            agent.reset()
            try:
                run_loop.run_loop([agent], env, max_episodes=1)
                total_reward = agent.total_reward
                rewards.append(total_reward)
                agent.qlearn.save()

                with open(CHECKPOINT_PATH, "w") as f:
                    f.write(str(episode))

                pd.DataFrame({
                    'Episode': list(range(1, len(rewards)+1)),
                    'Reward': rewards
                }).to_csv(REWARD_LOG_PATH, index=False)

                result = getattr(env._obs[0].observation, 'player_result', None)
                won = result[0].result == 1 if result else None
                recent_results.append(int(won) if won is not None else 0)

                outcome_color = Fore.GREEN if won else Fore.RED
                outcome_text = "WIN" if won else "LOSS"
                logging.info(outcome_color + f"🏁 EPISODE {episode:>3} RESULT: {outcome_text}")
                logging.info(f"{Fore.CYAN}📊 Reward: {total_reward:.2f} | Avg Last 10: {np.mean(rewards[-10:]):.2f} | Epsilon: {agent.qlearn.epsilon:.3f}")

                # Update live plot
                reward_line.set_data(range(1, len(rewards)+1), rewards)
                ax.set_xlim(0, len(rewards)+5)
                ax.set_ylim(min(rewards)-10, max(rewards)+10)
                fig.canvas.draw()
                fig.canvas.flush_events()

            except Exception as e:
                logging.error(Fore.RED + f"[Episode {episode}] ❌ ERROR: {str(e)}")
                logging.error(traceback.format_exc())
                rewards.append(0)
                recent_results.append(0)

    plt.ioff()
    logging.info(Fore.CYAN + "\n📈 Training complete. Finalizing reward plot...")
    plt.show()

if __name__ == "__main__":
    os.environ["SC2PATH"] = "D:/Games/StarCraft II"
    app.run(main)
