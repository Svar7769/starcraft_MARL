# Updated StarCraft II SmartAgent with debug, bootstrap, and fixed select_rect
import os
import random
import numpy as np
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
from absl import app
from collections import defaultdict

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features
import absl.logging

# Suppress logs
os.environ["SDL_VIDEODRIVER"] = "dummy"
absl.logging.set_verbosity('error')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# === Constants ===
_NO_OP                = actions.FUNCTIONS.no_op.id
_SELECT_POINT         = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT   = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS       = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE         = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY          = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP       = actions.FUNCTIONS.Attack_minimap.id
_TRAIN_SCV            = actions.FUNCTIONS.Train_SCV_quick.id
_MOVE_SCREEN          = actions.FUNCTIONS.Move_screen.id

_PLAYER_RELATIVE      = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE            = features.SCREEN_FEATURES.unit_type.index
_PLAYER_SELF          = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV           = 45
_TERRAN_SUPPLY_DEPOT  = 19
_TERRAN_BARRACKS      = 21

_NOT_QUEUED = [0]
_QUEUED     = [1]

MODEL_DIR         = "model"
QTABLE_PATH       = os.path.join(MODEL_DIR, "q_table_screen.pkl")
CHECKPOINT_PATH   = os.path.join(MODEL_DIR, "episode_checkpoint.txt")
REWARD_LOG_PATH   = os.path.join(MODEL_DIR, "reward_log.csv")
RESUME_TRAINING   = False

smart_actions = [
    'donothing', 'selectscv', 'buildsupplydepot', 'buildbarracks',
    'selectbarracks', 'buildmarine', 'buildscv', 'selectarmy',
    'attack', 'selectgroup', 'move'
]

class QLearningTable:
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.30):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.q = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.load()

    def choose_action(self, state, avail_actions):
        self._check(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(avail_actions)
        q_vals = self.q.loc[state, avail_actions]
        return q_vals.sample(frac=1).idxmax()

    def learn(self, s, a, r, s_):
        self._check(s); self._check(s_)
        predict = self.q.loc[s, a]
        target = r + self.gamma * self.q.loc[s_, :].max()
        self.q.loc[s, a] += self.lr * (target - predict)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _check(self, state):
        if state not in self.q.index:
            self.q.loc[state] = {a: 0.0 for a in self.actions}

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(QTABLE_PATH, "wb") as f:
            pickle.dump(self.q, f)

    def load(self):
        if os.path.exists(QTABLE_PATH) and os.path.getsize(QTABLE_PATH) > 0:
            try:
                with open(QTABLE_PATH, "rb") as f:
                    self.q = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load Q-table: {e}")
                self.q = pd.DataFrame(columns=self.actions, dtype=np.float64)

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.qlearn = QLearningTable(actions=smart_actions)
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_action = None
        self.prev_killed_units = 0
        self.prev_killed_buildings = 0
        self.total_reward = 0
        self.prev_depot_count = 0
        self.prev_barracks_count = 0
        self.prev_marine_count = 0
        self.state_visits = defaultdict(int)
        self.last_action_taken = None
        self.action_repeat_count = 0

    def step(self, obs):
        super().step(obs)

        # DEBUG available actions
        raw_avail = obs.observation.available_actions
        print("RAW available_actions IDs:", raw_avail)

        # base position
        try:
            player_y, _ = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31
        except:
            self.base_top_left = True

        # counts and buckets
        ut = obs.observation.feature_screen[_UNIT_TYPE]
        depot_count = int((ut == _TERRAN_SUPPLY_DEPOT).sum())
        barracks_count = int((ut == _TERRAN_BARRACKS).sum())
        supply = min(int(obs.observation['player'][4] / 10), 5)
        army = min(int(obs.observation['player'][5] / 5), 5)
        marine_count = army
        killed_units = int(obs.observation['score_cumulative'][5])
        killed_buildings = int(obs.observation['score_cumulative'][6])

        # state & reward
        state = str([int(depot_count>0), int(barracks_count>0), supply, army])
        self.state_visits[state] += 1
        reward = 0.01 if self.state_visits[state] == 1 else 0.0
        if depot_count > self.prev_depot_count:    reward += 0.3
        if barracks_count > self.prev_barracks_count: reward += 0.5
        if marine_count > self.prev_marine_count:   reward += 0.1
        reward += max(0, killed_units - self.prev_killed_units) * 0.2
        reward += max(0, killed_buildings - self.prev_killed_buildings) * 0.5

        # mask actions
        available = self._available_actions(obs)
        filtered = [a for a in smart_actions if a in available]
        print(" filtered_actions:", filtered)
        self.qlearn._check(state)

        # bootstrap first depot
        if depot_count == 0:
            if 'selectscv' in filtered and self.prev_action != 'selectscv':
                action = 'selectscv'
            elif 'buildsupplydepot' in filtered:
                action = 'buildsupplydepot'
            else:
                action = 'donothing'
        else:
            action = 'donothing' if not filtered else self.qlearn.choose_action(state, filtered)

        # repeat penalty
        if action == self.last_action_taken:
            self.action_repeat_count += 1
        else:
            self.action_repeat_count = 1
            self.last_action_taken = action
        if action != 'donothing' and self.action_repeat_count > 3:
            reward -= 0.5
            print(f"[Penalty] Repeating '{action}' x{self.action_repeat_count}")

        # negative shaping
        if marine_count < self.prev_marine_count: reward -= 0.2

        # learn step
        self.total_reward += reward
        if self.prev_state is not None:
            self.qlearn.learn(self.prev_state, self.prev_action, reward, state)

        # save for next
        self.prev_state = state
        self.prev_action = action
        self.prev_killed_units = killed_units
        self.prev_killed_buildings = killed_buildings
        self.prev_depot_count = depot_count
        self.prev_barracks_count = barracks_count
        self.prev_marine_count = marine_count

        return self._dispatch(action, obs)

    def _safe_coord(self, x, y, max_val=83):
        return [int(np.clip(x, 0, max_val)), int(np.clip(y, 0, max_val))]

    def _dispatch(self, action_name, obs):
        available = obs.observation.available_actions
        units     = getattr(obs.observation, "feature_units", [])

        if action_name == 'donothing':
            return actions.FunctionCall(_NO_OP, [])

        if action_name == 'selectscv' and _SELECT_POINT in available:
            scvs = [u for u in units if u.unit_type == _TERRAN_SCV]
            if scvs:
                u = random.choice(scvs)
                # queue flag must be a list ([0]), coords is a list
                return actions.FunctionCall(
                    _SELECT_POINT,
                    [_NOT_QUEUED, self._safe_coord(u.x, u.y)]
                )

        if action_name == 'selectbarracks' and _SELECT_POINT in available:
            brs = [u for u in units if u.unit_type == _TERRAN_BARRACKS]
            if brs:
                u = random.choice(brs)
                return actions.FunctionCall(
                    _SELECT_POINT,
                    [_NOT_QUEUED, self._safe_coord(u.x, u.y)]
                )

        if action_name == 'buildsupplydepot' and _BUILD_SUPPLY_DEPOT in available:
            cc = [u for u in units if u.unit_type == _TERRAN_COMMANDCENTER]
            if cc:
                x, y = cc[0].x + 10, cc[0].y + 10
                return actions.FunctionCall(
                    _BUILD_SUPPLY_DEPOT,
                    [_NOT_QUEUED, self._safe_coord(x, y)]
                )

        if action_name == 'buildbarracks' and _BUILD_BARRACKS in available:
            cc = [u for u in units if u.unit_type == _TERRAN_COMMANDCENTER]
            if cc:
                x, y = cc[0].x - 10, cc[0].y - 10
                return actions.FunctionCall(
                    _BUILD_BARRACKS,
                    [_NOT_QUEUED, self._safe_coord(x, y)]
                )

        if action_name == 'buildmarine' and _TRAIN_MARINE in available:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        if action_name == 'buildscv' and _TRAIN_SCV in available:
            return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

        if action_name == 'selectarmy' and _SELECT_ARMY in available:
            return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        if action_name == 'attack' and _ATTACK_MINIMAP in available:
            target = [39, 45] if self.base_top_left else [21, 24]
            return actions.FunctionCall(
                _ATTACK_MINIMAP,
                [_NOT_QUEUED, target]
            )

        if action_name == 'move' and _MOVE_SCREEN in available:
            tx, ty = random.randint(10, 70), random.randint(10, 70)
            return actions.FunctionCall(
                _MOVE_SCREEN,
                [_NOT_QUEUED, self._safe_coord(tx, ty)]
            )

        if action_name == 'selectgroup' and actions.FUNCTIONS.select_rect.id in available:
            marines = [u for u in units if u.unit_type == 48]
            if marines:
                xs = [u.x for u in marines]; ys = [u.y for u in marines]
                x0, x1 = max(0, min(xs) - 2), min(83, max(xs) + 2)
                y0, y1 = max(0, min(ys) - 2), min(83, max(ys) + 2)
                # select_rect wants exactly three arguments:
                #   [queue], [x1,y1], [x2,y2]
                return actions.FunctionCall(
                    actions.FUNCTIONS.select_rect.id,
                    [_NOT_QUEUED, [x0, y0], [x1, y1]]
                )

        # fallback
        return actions.FunctionCall(_NO_OP, [])

    def _available_actions(self, obs):
        avail = set()
        ut = obs.observation.feature_screen[_UNIT_TYPE]
        available = obs.observation.available_actions

        if _SELECT_POINT in available and (ut == _TERRAN_SCV).any():
            avail.add('selectscv')
        if _SELECT_POINT in available and (ut == _TERRAN_BARRACKS).any():
            avail.add('selectbarracks')
        if _BUILD_SUPPLY_DEPOT in available:
            avail.add('buildsupplydepot')
        if _BUILD_BARRACKS in available:
            avail.add('buildbarracks')
        if _TRAIN_MARINE in available:
            avail.add('buildmarine')
        if _SELECT_ARMY in available:
            avail.add('selectarmy')
        if _ATTACK_MINIMAP in available:
            avail.add('attack')
        if _TRAIN_SCV in available:
            avail.add('buildscv')
        if _MOVE_SCREEN in available:
            avail.add('move')
        if actions.FUNCTIONS.select_rect.id in available:
            avail.add('selectgroup')

        avail.add('donothing')
        return list(avail)

# get_difficulty and main() remain unchanged

def get_difficulty(win_rate):
    if   win_rate > 0.8: return sc2_env.Difficulty.medium
    elif win_rate > 0.6: return sc2_env.Difficulty.easy
    else:                return sc2_env.Difficulty.very_easy

def main(_):
    agent = SmartAgent()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if RESUME_TRAINING and os.path.exists(QTABLE_PATH) and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            start_episode = int(f.read().strip()) + 1
        rewards = pd.read_csv(REWARD_LOG_PATH)['Reward'].tolist() if os.path.exists(REWARD_LOG_PATH) else []
        recent_results = [1 if r > 0 else 0 for r in rewards[-10:]]
    else:
        start_episode, rewards, recent_results = 1, [], []

    for episode in range(start_episode, 1001):
        win_rate = (sum(recent_results[-10:]) / min(len(recent_results), 10)) if recent_results else 0.0
        difficulty = get_difficulty(win_rate)

        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, difficulty)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True,
                use_raw_units=False,
                use_unit_counts=True,
                use_camera_position=True,
                action_space=actions.ActionSpace.FEATURES
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

                # This uses a private _obs but kept for now:
                result = env._obs[0].observation.player_result[0].result \
                         if hasattr(env._obs[0].observation, 'player_result') else 0
                won = (result == 1)
                recent_results.append(int(won))

                print(
                    f"Episode {episode} | {'Win' if won else 'Loss'} | "
                    f"Reward: {total_reward:.2f} | Avg(10): {np.mean(rewards[-10:]):.2f} | "
                    f"Epsilon: {agent.qlearn.epsilon:.3f}"
                )
            except Exception as e:
                print(f"[Episode {episode}] Error: {e}")
                rewards.append(0)
                recent_results.append(0)

    # Plot training curve
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png")
    # plt.show()

if __name__ == "__main__":
    app.run(main)
