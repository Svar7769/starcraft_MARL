import os
import random
import pickle
import numpy as np
import cv2
import logging

from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# ——— Logging Configuration ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ——— Environment Configuration ———
MODEL_DIR    = "model"
QTABLE_PATH  = os.path.join(MODEL_DIR, "q_table.pkl")
MAP_NAME     = "CollectMineralsAndGas"
SCREEN_SIZE  = 84
MINIMAP_SIZE = 64
STEP_MUL     = 16
MAX_EPISODES = 100

# ——— SC2 Function & Unit IDs ———
_NO_OP          = actions.FUNCTIONS.no_op.id
_SELECT_IDLE    = actions.FUNCTIONS.select_idle_worker.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_HARVEST        = actions.FUNCTIONS.Harvest_Gather_screen.id

# ——— Unit-type IDs ———
MINERAL_PATCH   = 341
VESPENE_GEYSER  = 342
SCV_UNIT_TYPE   = 45

# ——— Feature-layer indices ———
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE       = features.SCREEN_FEATURES.unit_type.index

# ——— Discrete Q-learning actions ———
ACTIONS = [
    'do_nothing',
    'select_idle',
    'build_refinery',
    'harvest',
]

class QLearningTable:
    def __init__(self, actions, lr=0.1, gamma=0.9, epsilon=0.2):
        self.actions = actions
        self.lr       = lr
        self.gamma    = gamma
        self.epsilon  = epsilon
        self.q        = {}  # state_str -> {action: q_value}
        self._load()

    def _load(self):
        if os.path.exists(QTABLE_PATH):
            with open(QTABLE_PATH, 'rb') as f:
                self.q = pickle.load(f)

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(QTABLE_PATH, 'wb') as f:
            pickle.dump(self.q, f)

    def _ensure(self, s):
        # If it's a brand-new state, create full action dict
        if s not in self.q:
            self.q[s] = {a: 0.0 for a in self.actions}
        else:
            # Back-fill any new actions into existing states
            for a in self.actions:
                if a not in self.q[s]:
                    self.q[s][a] = 0.0

    def choose(self, state, avail):
        self._ensure(state)
        if np.random.rand() < self.epsilon:
            return random.choice(avail)
        return max(avail, key=lambda a: self.q[state][a])

    def learn(self, s, a, r, s2):
        self._ensure(s)
        self._ensure(s2)
        q_pred   = self.q[s][a]
        q_target = r + self.gamma * max(self.q[s2].values())
        self.q[s][a] += self.lr * (q_target - q_pred)
        # decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)


class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.qtable       = QLearningTable(ACTIONS)
        self.prev_state   = None
        self.prev_action  = None
        self.prev_score   = None
        self.total_reward = 0
        self.episode      = 0
        self.step_count   = 0

    def reset(self):
        super().reset()
        self.episode      += 1
        self.step_count    = 0
        self.prev_state    = None
        self.prev_action   = None
        self.prev_score    = None
        self.total_reward  = 0
        logger.info("=== Starting Episode %d ===", self.episode)

    def step(self, obs):
        super().step(obs)
        self.step_count += 1

        # 1) Reward = delta of cumulative score
        cur_score = int(obs.observation['score_cumulative'][0])
        reward = 0 if self.prev_score is None else (cur_score - self.prev_score)
        self.prev_score = cur_score
        self.total_reward += reward

        # 2) What’s available?
        acts = obs.observation.available_actions
        fus  = getattr(obs.observation, "feature_units", [])

        can_idle    = _SELECT_IDLE    in acts
        can_refine  = _BUILD_REFINERY in acts
        can_harvest = _HARVEST        in acts

        # 3) Compact state
        state = (int(can_harvest), int(can_refine))

        # 4) Learn from last transition
        if self.prev_state is not None:
            self.qtable.learn(self.prev_state, self.prev_action, reward, state)

        # 5) Choose next action
        avail = ['do_nothing']
        if can_idle:    avail.append('select_idle')
        if can_refine:  avail.append('build_refinery')
        if can_harvest: avail.append('harvest')

        action = self.qtable.choose(state, avail)
        self.prev_state, self.prev_action = state, action

        # 6) Show feature-layers for debugging
        fs = obs.observation.feature_screen
        pr = (fs[_PLAYER_RELATIVE].astype(np.uint8) * 60)
        ut = fs[_UNIT_TYPE].astype(np.uint8)
        cv2.imshow("Player-relative", pr)
        cv2.imshow("Unit-type", ut)
        cv2.waitKey(1)

        # 7) Execute it
        return self._execute(action, obs)

    def _execute(self, action, obs):
        fus = getattr(obs.observation, "feature_units", [])

        if action == 'select_idle':
            # select *all* idle workers at once
            return actions.FunctionCall(_SELECT_IDLE, [[2]])  # 2 = SelectAdd.All

        if action == 'build_refinery':
            geysers = [u for u in fus if u.unit_type == VESPENE_GEYSER]
            if geysers:
                g = random.choice(geysers)
                return actions.FunctionCall(_BUILD_REFINERY, [[0], [g.x, g.y]])

        if action == 'harvest':
            minerals = [u for u in fus if u.unit_type == MINERAL_PATCH]
            if minerals:
                m = random.choice(minerals)
                return actions.FunctionCall(_HARVEST, [[0], [m.x, m.y]])

        # fallback
        return actions.FunctionCall(_NO_OP, [])


def main(argv):
    agent = SmartAgent()
    os.makedirs(MODEL_DIR, exist_ok=True)

    with sc2_env.SC2Env(
        map_name=MAP_NAME,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions      =features.Dimensions(screen=SCREEN_SIZE,
                                                        minimap=MINIMAP_SIZE),
            use_feature_units       =True,
            use_raw_units           =False,
            use_camera_position     =True,
            action_space            =actions.ActionSpace.FEATURES
        ),
        step_mul=STEP_MUL,
        visualize=True,
        game_steps_per_episode=60000,  # 0 means no limit
    ) as env:
        for _ in range(MAX_EPISODES):
            timesteps = env.reset()
            agent.reset()

            while True:
                action = agent.step(timesteps[0])
                timesteps = env.step([action])
                if timesteps[0].last():
                    break

            # Episode done → log summary
            logger.info(
                "=== Episode %d ended: total_reward=%.2f, ε=%.3f ===",
                agent.episode,
                agent.total_reward,
                agent.qtable.epsilon
            )

    # save once at the very end
    agent.qtable.save()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
