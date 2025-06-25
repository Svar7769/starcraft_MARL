import numpy as np
import random
import torch
from pysc2.lib import actions, features, units

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

ACTION_LIST = [
    "no_op",
    "select_scv",
    "build_supply_depot",
    "build_barracks",
    "train_marine",
    "attack",
    # ... add more as needed
]
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_LIST)}

SCREEN_SIZE = 84

# Terran building unit_type IDs (partial list — expand as needed)
TERRAN_STRUCTURE_TYPES = [
    18,   # CommandCenter
    20,   # SupplyDepot
    21,   # Barracks
    22,   # EngineeringBay
    23,   # MissileTurret
    24,   # Bunker
    25,   # Refinery
    27,   # Factory
    28,   # GhostAcademy
    29,   # Starport
    30,   # Armory
    130, 131, 132, 133  # Tech lab, Reactor, etc.
]

def safe_coords(x, y, screen_size=SCREEN_SIZE):
    """Ensure coordinates are within the screen bounds."""
    x = int(max(0, min(screen_size - 1, x)))
    y = int(max(0, min(screen_size - 1, y)))
    return [x, y]

def preprocess(ts):
    """Preprocess the timestep observation for neural network input."""
    fs = ts.observation.feature_screen
    pr = fs[_PLAYER_RELATIVE].astype(np.float32) / 4.0
    ut = fs[_UNIT_TYPE].astype(np.float32)
    ut = ut / (ut.max() if ut.max() > 0 else 1)
    stacked = np.stack([pr, ut], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).float()

def legal_actions(ts):
    """Return a list of legal action indices for the current timestep."""
    avail = set(ts.observation.available_actions)
    fus = ts.observation.feature_units
    legal = [ACTION_INDEX['no_op']]  # <-- FIXED: was 'do_nothing'

    if actions.FUNCTIONS.Move_screen.id in avail:
        legal.append(ACTION_INDEX['move'])

    if actions.FUNCTIONS.Attack_screen.id in avail:
        legal.append(ACTION_INDEX['attack'])

    if any('Build' in actions.FUNCTIONS[a].name for a in avail):
        legal.append(ACTION_INDEX['build'])

    if actions.FUNCTIONS.Harvest_Gather_screen.id in avail and any(u.unit_type == 341 for u in fus):
        legal.append(ACTION_INDEX['gather'])

    if any('Research' in actions.FUNCTIONS[a].name for a in avail):
        legal.append(ACTION_INDEX['upgrade'])

    if any('Train' in actions.FUNCTIONS[a].name for a in avail):
        legal.append(ACTION_INDEX['train'])

    return legal

def make_pysc2_call(action_idx, ts, pending=None):
    """
    Convert action index to proper FunctionCall, handling multi-step actions.
    Returns (FunctionCall, pending_action_dict or None)
    """
    action_name = ACTION_LIST[action_idx]
    obs = ts.observation
    units = getattr(obs, "raw_units", None)
    available = obs["available_actions"] if "available_actions" in obs else []

    if action_name == "no_op":
        return actions.RAW_FUNCTIONS.no_op(), None

    if action_name == "select_scv":
        if units is not None:
            scvs = [u for u in units if u.unit_type == units.Terran.SCV and u.alliance == 1]
            if scvs:
                return actions.RAW_FUNCTIONS.select_unit("select", [scvs[0].tag]), None

    if action_name == "build_supply_depot":
        if units is not None:
            scvs = [u for u in units if u.unit_type == units.Terran.SCV and u.alliance == 1]
            if scvs:
                x, y = 40, 40  # or random/safe location
                return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scvs[0].tag, [x, y]), None

    if action_name == "build_barracks":
        if units is not None:
            scvs = [u for u in units if u.unit_type == units.Terran.SCV and u.alliance == 1]
            if scvs:
                x, y = 45, 45
                return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scvs[0].tag, [x, y]), None

    if action_name == "train_marine":
        barracks = [u for u in units if u.unit_type == units.Terran.Barracks and u.alliance == 1]
        if barracks:
            return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks[0].tag), None

    if action_name == "attack":
        marines = [u for u in units if u.unit_type == units.Terran.Marine and u.alliance == 1]
        enemies = [u for u in units if u.alliance == 4]
        if marines and enemies:
            return actions.RAW_FUNCTIONS.Attack_unit("now", marines[0].tag, [enemies[0].tag]), None

    # fallback
    return actions.RAW_FUNCTIONS.no_op(), None

def make_pysc2_call_core(action_idx, ts):
    """
    Simpler version: directly issues the action if available, otherwise no_op.
    """
    obs = ts.observation
    fus = obs.feature_units
    avail = set(obs.available_actions)

    if action_idx == ACTION_INDEX['do_nothing']:
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    if action_idx == ACTION_INDEX['move'] and actions.FUNCTIONS.Move_screen.id in avail:
        x, y = np.random.randint(0, SCREEN_SIZE), np.random.randint(0, SCREEN_SIZE)
        return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], safe_coords(x, y)])

    if action_idx == ACTION_INDEX['attack'] and actions.FUNCTIONS.Attack_screen.id in avail:
        enemies = [u for u in fus if u.alliance == features.PlayerRelative.ENEMY]
        if enemies:
            target = random.choice(enemies)
            return actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [[0], safe_coords(target.x, target.y)])

    if action_idx == ACTION_INDEX['build']:
        build_actions = [a for a in avail if 'Build' in actions.FUNCTIONS[a].name]
        if build_actions:
            build_action = random.choice(build_actions)
            buildable = np.argwhere(obs.feature_screen.buildable == 1)
            if buildable.size > 0:
                y, x = random.choice(buildable)
                return actions.FunctionCall(build_action, [[0], safe_coords(x, y)])

    if action_idx == ACTION_INDEX['gather'] and actions.FUNCTIONS.Harvest_Gather_screen.id in avail:
        minerals = [u for u in fus if u.unit_type == 341]
        if minerals:
            target = random.choice(minerals)
            return actions.FunctionCall(actions.FUNCTIONS.Harvest_Gather_screen.id, [[0], safe_coords(target.x, target.y)])

    if action_idx == ACTION_INDEX['upgrade']:
        upgrade_actions = [a for a in avail if 'Research' in actions.FUNCTIONS[a].name]
        if upgrade_actions:
            upgrade_action = random.choice(upgrade_actions)
            return actions.FunctionCall(upgrade_action, [[0]])

    if action_idx == ACTION_INDEX['train']:
        train_actions = [a for a in avail if 'Train' in actions.FUNCTIONS[a].name]
        if train_actions:
            train_action = random.choice(train_actions)
            return actions.FunctionCall(train_action, [[0]])

    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
