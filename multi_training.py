import os
import random
import numpy as np
import pickle
import pandas as pd
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from multiprocessing import get_context
from q_agent import SmartAgent  # assumes your main agent is in q_agent.py

flags.FLAGS(['run'])  # Required to avoid Abseil flags error

QTABLE_PATH = "q_table_screen.pkl"
os.environ["NO_QTABLE_LOAD"] = "1"  # Prevent Q-table loading in subprocesses

def get_difficulty(win_rate):
    if win_rate > 0.8:
        return sc2_env.Difficulty.medium
    elif win_rate > 0.6:
        return sc2_env.Difficulty.easy
    else:
        return sc2_env.Difficulty.very_easy

def run_episode(episode_id, win_rate):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    difficulty = get_difficulty(win_rate)
    total_reward = 0
    result_value = 0

    try:
        agent = SmartAgent()

        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, difficulty)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=False,
                use_raw_units=False,
                use_unit_counts=True,
                use_camera_position=True,
                action_space=actions.ActionSpace.FEATURES
            ),
            step_mul=16,
            disable_fog=False,
            visualize=False,
            realtime=False,
        ) as env:
            agent.reset()
            agent.episode = episode_id

            timesteps = env.reset()
            agent.setup(env.observation_spec(), env.action_spec())

            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    obs = timesteps[0].observation
                    if isinstance(obs, dict) and 'player_result' in obs:
                        result = obs['player_result'][0]['result']
                        result_value = 1 if result == 1 else 0
                    break
                timesteps = env.step(step_actions)

            total_reward = agent.total_reward

    except Exception as e:
        print(f"[Episode {episode_id}] Error: {e}")

    return (episode_id, total_reward, result_value)

def main():
    num_episodes = 20
    num_workers = 4
    rewards = []
    win_stats = []

    with get_context("spawn").Pool(processes=num_workers) as pool:
        win_rate = 0.0
        for batch_start in range(0, num_episodes, num_workers):
            batch = list(range(batch_start, min(batch_start + num_workers, num_episodes)))
            results = pool.starmap(run_episode, [(eid, win_rate) for eid in batch])

            for eid, reward, win in results:
                rewards.append(reward)
                win_stats.append(win)
                print(f"Episode {eid}: Reward = {reward:.2f}, Win = {bool(win)}")

            recent = win_stats[-10:]
            win_rate = sum(recent) / len(recent) if recent else 0.0

    # Save Q-table from a new agent to avoid sharing conflict
    agent = SmartAgent()
    agent.qlearn.save()

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
    plt.title("Parallel Adaptive Difficulty Training")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()