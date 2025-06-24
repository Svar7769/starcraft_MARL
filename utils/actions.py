import numpy as np
import random
import torch
from pysc2.lib import actions, features, units

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

ACTION_LIST = ['do_nothing', 'move', 'attack', 'build', 'gather', 'upgrade', 'train']
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_LIST)}

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
    legal = [ACTION_INDEX['do_nothing']]

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
    obs = ts.observation
    fus = obs.feature_units
    avail = set(obs.available_actions)

    # Handle pending actions first
    if pending:
        if pending['action_fn'] in avail:
            args = pending['args']
            # If the action requires coordinates, ensure they're safe
            if len(args) > 1 and isinstance(args[1], list) and len(args[1]) == 2:
                x, y = args[1]
                return actions.FunctionCall(pending['action_fn'], [args[0], safe_coords(x, y)]), None
            return actions.FunctionCall(pending['action_fn'], args), None
        # If not available, do nothing
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None

    # Train action sequence (select building, then train)
    if action_idx == ACTION_INDEX['train']:
        buildings = [u for u in fus 
                   if u.alliance == features.PlayerRelative.SELF 
                   and u.unit_type in TERRAN_STRUCTURE_TYPES]
        if not buildings or actions.FUNCTIONS.select_point.id not in avail:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None
        
        building = random.choice(buildings)
        select_action = actions.FunctionCall(
            actions.FUNCTIONS.select_point.id,
            [[0], safe_coords(building.x, building.y)]
        )
        
        train_actions = [a for a in avail if 'Train' in actions.FUNCTIONS[a].name]
        if train_actions:
            return select_action, {
                'action_fn': random.choice(train_actions),
                'args': [[0]]
            }
        return select_action, None

    # Select a unit first for most actions
    units_self = [u for u in fus if u.alliance == features.PlayerRelative.SELF]
    if not units_self or actions.FUNCTIONS.select_point.id not in avail:
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None
    
    unit = random.choice(units_self)
    select_action = actions.FunctionCall(
        actions.FUNCTIONS.select_point.id,
        [[0], safe_coords(unit.x, unit.y)]
    )

    # Now handle the actual action types
    if action_idx == ACTION_INDEX['move'] and actions.FUNCTIONS.Move_screen.id in avail:
        x, y = np.random.randint(0, SCREEN_SIZE), np.random.randint(0, SCREEN_SIZE)
        return select_action, {
            'action_fn': actions.FUNCTIONS.Move_screen.id,
            'args': [[0], [x, y]]
        }

    elif action_idx == ACTION_INDEX['attack'] and actions.FUNCTIONS.Attack_screen.id in avail:
        enemies = [u for u in fus if u.alliance == features.PlayerRelative.ENEMY]
        if enemies:
            target = random.choice(enemies)
            return select_action, {
                'action_fn': actions.FUNCTIONS.Attack_screen.id,
                'args': [[0], [target.x, target.y]]
            }
    
    elif action_idx == ACTION_INDEX['build'] and any('Build' in actions.FUNCTIONS[a].name for a in avail):
        build_actions = [a for a in avail if 'Build' in actions.FUNCTIONS[a].name]
        if build_actions:
            build_action = random.choice(build_actions)
            buildable = np.argwhere(obs.feature_screen.buildable == 1)
            if buildable.size > 0:
                y, x = random.choice(buildable)
                return select_action, {
                    'action_fn': build_action,
                    'args': [[0], [x, y]]
                }
    
    elif action_idx == ACTION_INDEX['gather'] and actions.FUNCTIONS.Harvest_Gather_screen.id in avail:
        minerals = [u for u in fus if u.unit_type == 341]  # Mineral field ID
        if minerals:
            target = random.choice(minerals)
            return select_action, {
                'action_fn': actions.FUNCTIONS.Harvest_Gather_screen.id,
                'args': [[0], [target.x, target.y]]
            }
    
    elif action_idx == ACTION_INDEX['upgrade'] and any('Research' in actions.FUNCTIONS[a].name for a in avail):
        upgrade_actions = [a for a in avail if 'Research' in actions.FUNCTIONS[a].name]
        if upgrade_actions:
            return select_action, {
                'action_fn': random.choice(upgrade_actions),
                'args': [[0]]
            }

    return select_action, None

def make_pysc2_call_core(action_idx, ts):
    """
    Simpler version: directly issues the action if available, otherwise no_op.
    """
    obs = ts.observation
    fus = obs.feature_units
    avail = set(obs.available_actions)

    if action_idx == ACTION_INDEX['do_nothing']:
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None

    if action_idx == ACTION_INDEX['move'] and actions.FUNCTIONS.Move_screen.id in avail:
        x, y = np.random.randint(0, SCREEN_SIZE), np.random.randint(0, SCREEN_SIZE)
        return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], safe_coords(x, y)]), None

    if action_idx == ACTION_INDEX['attack'] and actions.FUNCTIONS.Attack_screen.id in avail:
        enemies = [u for u in fus if u.alliance == features.PlayerRelative.ENEMY]
        if enemies:
            target = random.choice(enemies)
            return actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [[0], safe_coords(target.x, target.y)]), None

    if action_idx == ACTION_INDEX['build']:
        build_actions = [a for a in avail if 'Build' in actions.FUNCTIONS[a].name]
        if build_actions:
            build_action = random.choice(build_actions)
            buildable = np.argwhere(obs.feature_screen.buildable == 1)
            if buildable.size > 0:
                y, x = random.choice(buildable)
                return actions.FunctionCall(build_action, [[0], safe_coords(x, y)]), None

    if action_idx == ACTION_INDEX['gather'] and actions.FUNCTIONS.Harvest_Gather_screen.id in avail:
        minerals = [u for u in fus if u.unit_type == 341]
        if minerals:
            target = random.choice(minerals)
            return actions.FunctionCall(actions.FUNCTIONS.Harvest_Gather_screen.id, [[0], safe_coords(target.x, target.y)]), None

    if action_idx == ACTION_INDEX['upgrade']:
        upgrade_actions = [a for a in avail if 'Research' in actions.FUNCTIONS[a].name]
        if upgrade_actions:
            upgrade_action = random.choice(upgrade_actions)
            return actions.FunctionCall(upgrade_action, [[0]]), None

    if action_idx == ACTION_INDEX['train']:
        train_actions = [a for a in avail if 'Train' in actions.FUNCTIONS[a].name]
        if train_actions:
            train_action = random.choice(train_actions)
            return actions.FunctionCall(train_action, [[0]]), None

    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None
