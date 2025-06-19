import os
import pickle
import numpy as np
import pandas as pd

MODEL_DIR = "model"
QTABLE_PATH = os.path.join(MODEL_DIR, "q_table_screen.pkl")

class QLearningTable:
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.40):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.load()

    def choose_action(self, state, e_greedy=0.9):
        self.check_state_exist(state)
        if np.random.uniform() < e_greedy:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[state, :]
            action = state_action.idxmax()
        return str(action)  # <-- Ensure Python str

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        a = str(a)  # <-- Ensure Python str
        predict = self.q_table.loc[s, a]
        target = r + self.gamma * self.q.loc[s_, :].max()
        self.q.loc[s, a] += self.lr * (target - predict)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _check(self, state, actions_subset=None):
        if state not in self.q.index:
            self.q.loc[state] = {a: 0.0 for a in self.actions}
        if actions_subset:
            for a in actions_subset:
                if a not in self.q.columns:
                    self.q[a] = 0.0

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(QTABLE_PATH, "wb") as f:
            pickle.dump(self.q, f)

    def load(self):
        if os.path.exists(QTABLE_PATH) and os.path.getsize(QTABLE_PATH) > 0:
            try:
                with open(QTABLE_PATH, "rb") as f:
                    self.q = pickle.load(f)
            except Exception:
                self.q = pd.DataFrame(columns=self.actions, dtype=np.float64)
