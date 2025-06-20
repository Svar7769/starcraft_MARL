import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, TensorDataset

import logging
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
MAP_NAME     = "CollectMineralsAndGas"
SCREEN_SIZE  = 84
MINIMAP_SIZE = 64
STEP_MUL     = 16
NB_ACTORS    = 2
T            = 128
K            = 10
BATCH_SIZE   = 256
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
LR           = 2.5e-4
ENT_COEF     = 0.01
VF_COEF      = 1.0
MAX_ITERS    = 40_000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── SC2 feature indices & function IDs ─────────────────────────────────────
_PLAYER_RELATIVE   = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE         = features.SCREEN_FEATURES.unit_type.index

ACTION_LIST = [
    'do_nothing',
    'select_idle',
    'build_refinery',
    'harvest',
]
N_ACTIONS = len(ACTION_LIST)

FUNC_ID = {
    'do_nothing':     actions.FUNCTIONS.no_op.id,
    'select_idle':    actions.FUNCTIONS.select_idle_worker.id,
    'build_refinery': actions.FUNCTIONS.Build_Refinery_screen.id,
    'harvest':        actions.FUNCTIONS.Harvest_Gather_screen.id,
}

# ─── Actor‐Critic Model ──────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, in_channels, nb_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4), nn.Tanh(),
            nn.Conv2d(16, 32, 4, stride=2),          nn.Tanh(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, SCREEN_SIZE, SCREEN_SIZE)
            conv_out = self.conv(dummy).shape[-1]
        self.fc     = nn.Sequential(nn.Linear(conv_out, 256), nn.Tanh())
        self.actor  = nn.Linear(256, nb_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        return self.actor(h), self.critic(h).squeeze(-1)

# ─── Batched SC2 Env Wrapper ─────────────────────────────────────────────────
class SC2Envs:
    def __init__(self, nb_actor):
        logger.info("Initializing %d SC2 env(s)...", nb_actor)
        self.nb   = nb_actor
        self.envs = [self._make_env() for _ in range(nb_actor)]
        self.obs  = [None]*nb_actor
        self.done = [False]*nb_actor
        self._init_all()
        logger.info("All SC2 env(s) ready.")

    def _make_env(self):
        return sc2_env.SC2Env(
            map_name=MAP_NAME,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=SCREEN_SIZE, minimap=MINIMAP_SIZE),
                use_feature_units=True,
                use_raw_units=False,
                use_camera_position=True,
                action_space=actions.ActionSpace.FEATURES
            ),
            step_mul=STEP_MUL,
            game_steps_per_episode=0,
            visualize=False,
        )

    def _init_all(self):
        for i, e in enumerate(self.envs):
            ts = e.reset()[0]
            self.obs[i]  = ts
            self.done[i] = False

    def reset(self, i):
        ts = self.envs[i].reset()[0]
        self.obs[i]  = ts
        self.done[i] = False
        return ts

    def step(self, i, fc):
        ts = self.envs[i].step([fc])[0]
        self.obs[i] = ts
        self.done[i] = ts.last()
        return ts

    def close(self):
        for e in self.envs:
            e.close()

# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(ts):
    fs = ts.observation.feature_screen
    pr = fs[_PLAYER_RELATIVE].astype(np.float32) / 4.0
    ut = fs[_UNIT_TYPE].astype(np.float32)       / fs[_UNIT_TYPE].max()
    stacked = np.stack([pr, ut], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).float().to(DEVICE)

# ─── Which of our 4 actions are legal right now ─────────────────────────────
def legal_actions(ts):
    avail = set(ts.observation.available_actions)
    fus   = ts.observation.feature_units
    legal = [0]
    if FUNC_ID['select_idle'] in avail:
        legal.append(1)
    if FUNC_ID['build_refinery'] in avail and any(u.unit_type==342 for u in fus):
        legal.append(2)
    if FUNC_ID['harvest'] in avail and any(u.unit_type==341 for u in fus):
        legal.append(3)
    return legal

# ─── pysc2 FunctionCall from discrete idx ───────────────────────────────────
def make_pysc2_call(action_idx, ts):
    name = ACTION_LIST[action_idx]
    fid  = FUNC_ID[name]
    if name == 'select_idle':
        return actions.FunctionCall(fid, [[2]])
    if name in ('build_refinery','harvest'):
        fus = ts.observation.feature_units
        if name=='build_refinery':
            cand = [u for u in fus if u.unit_type==342]
        else:
            cand = [u for u in fus if u.unit_type==341]
        if not cand:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        u = random.choice(cand)
        return actions.FunctionCall(fid, [[0],[u.x,u.y]])
    return actions.FunctionCall(fid, [])

# ─── PPO Training ────────────────────────────────────────────────────────────
def PPO(envs, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERS
    )

    logger.info("▶️  Starting PPO for %d iterations", MAX_ITERS)
    for it in range(MAX_ITERS):
        if it % 1000 == 0:
            logger.info("🔄 Iter %d / %d", it, MAX_ITERS)

        # storage buffers
        obs_buf  = torch.zeros(envs.nb, T, 2, SCREEN_SIZE, SCREEN_SIZE, device=DEVICE)
        act_buf  = torch.zeros(envs.nb, T,      dtype=torch.long, device=DEVICE)
        logp_buf = torch.zeros(envs.nb, T,                     device=DEVICE)
        val_buf  = torch.zeros(envs.nb, T+1,                   device=DEVICE)
        rew_buf  = torch.zeros(envs.nb, T,                     device=DEVICE)
        done_buf = torch.zeros(envs.nb, T,                     device=DEVICE)
        adv_buf  = torch.zeros(envs.nb, T,                     device=DEVICE)

        # track last cumulative score per actor
        last_score = [0]*envs.nb

        # ─── Rollout ─────────────────────────────────────────────────────────
        with torch.no_grad():
            for t in range(T):
                for i in range(envs.nb):
                    ts    = envs.obs[i]
                    state = preprocess(ts)
                    logits, value = model(state)

                    # mask illegal
                    LA   = legal_actions(ts)
                    mask = torch.full_like(logits, float('-inf'))
                    mask[0, LA] = 0.0
                    dist = Categorical(logits=logits + mask)

                    action = dist.sample()
                    logp   = dist.log_prob(action)
                    fc     = make_pysc2_call(action.item(), ts)

                    # step (fallback to no-op if SC2 rejects it)
                    try:
                        ts2 = envs.step(i, fc)
                    except ValueError:
                        ts2 = envs.step(i, actions.FunctionCall(actions.FUNCTIONS.no_op.id, []))

                    # ← curriculum reward = Δscore_cumulative
                    cur_score = int(ts2.observation['score_cumulative'][0])
                    r = cur_score - last_score[i]
                    last_score[i] = cur_score

                    d = float(ts2.last())

                    obs_buf[i,t]  = state
                    act_buf[i,t]  = action
                    logp_buf[i,t] = logp
                    val_buf[i,t]  = value
                    rew_buf[i,t]  = r
                    done_buf[i,t] = d

                    if d:
                        envs.reset(i)

            # bootstrap final value
            for i in range(envs.nb):
                ts = envs.obs[i]
                val_buf[i,T] = model(preprocess(ts))[1]

        # ─── GAE & flatten ────────────────────────────────────────────────────
        for i in range(envs.nb):
            gae = 0
            for t in reversed(range(T)):
                mask = 1.0 - done_buf[i,t]
                delta= rew_buf[i,t] + GAMMA*val_buf[i,t+1]*mask - val_buf[i,t]
                gae  = delta + GAMMA*GAE_LAMBDA*mask*gae
                adv_buf[i,t] = gae

        b_s = obs_buf.reshape(-1,2,SCREEN_SIZE,SCREEN_SIZE)
        b_a = act_buf.reshape(-1)
        b_lp= logp_buf.reshape(-1)
        b_v = val_buf[:,:T].reshape(-1)
        b_ad= adv_buf.reshape(-1)

        # ─── PPO updates ─────────────────────────────────────────────────────
        for _ in range(K):
            ds     = TensorDataset(b_s,b_a,b_lp,b_v,b_ad)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            for st, ac, old_lp, old_v, adv in loader:
                logits, val = model(st)
                dist        = Categorical(logits=logits)
                lp          = dist.log_prob(ac)
                ratio       = torch.exp(lp - old_lp)

                clip   = 0.1 * (1 - it/MAX_ITERS)
                obj1   = adv * ratio
                obj2   = adv * torch.clamp(ratio, 1-clip, 1+clip)
                p_loss = -torch.min(obj1,obj2).mean()

                ret     = adv + old_v
                v1      = (val - ret).pow(2)
                v2      = (torch.clamp(val,old_v-clip,old_v+clip)-ret).pow(2)
                v_loss  = 0.5 * torch.max(v1,v2).mean()

                entropy = dist.entropy().mean()
                loss    = p_loss + VF_COEF*v_loss - ENT_COEF*entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),0.5)
                optimizer.step()

        scheduler.step()

    envs.close()
    logger.info("✅ Training complete")

def main(_):
    envs  = SC2Envs(NB_ACTORS)
    model = ActorCritic(2, N_ACTIONS).to(DEVICE)
    PPO(envs, model)

if __name__ == "__main__":
    app.run(main)
