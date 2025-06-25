import random
import numpy as np
from pysc2.lib import actions, features
from config.spartan import SCREEN_SIZE # Ensure SCREEN_SIZE is imported from config
from utils.preprocessing import safe_coords # Import safe_coords from preprocessing


# Define specialized action lists for each King
# These lists define the discrete action space for each King's policy
COMBAT_ACTION_LIST = [
    'do_nothing',
    'select_army',       # Select all military units (e.g., control group 0)
    'attack_group_target', # Attack a specific screen coordinate/enemy (requires army selected)
    'move_group_target',   # Move to a specific screen coordinate (requires army selected)
    # Add more specific combat actions if needed, e.g., 'attack_closest_enemy', 'retreat'
]

RESOURCE_ACTION_LIST = [
    'do_nothing',
    'train_scv',          # Train an SCV (requires Command Center selected)
    'build_supply_depot', # Build Supply Depot (requires SCV selected)
    'build_barracks',     # Build Barracks (requires SCV selected)
    'harvest_gather',     # Gather minerals/gas (requires SCV selected)
    # Add more economic actions, e.g., 'build_refinery', 'expand_base'
]

COMBAT_ACTION_INDEX = {name: idx for idx, name in enumerate(COMBAT_ACTION_LIST)}
RESOURCE_ACTION_INDEX = {name: idx for idx, name in enumerate(RESOURCE_ACTION_LIST)}

# Terran unit_type IDs (partial list for common checks)
UNIT_TYPE_COMMAND_CENTER = 18
UNIT_TYPE_SCV = 45
UNIT_TYPE_MINERAL_FIELD = 341
UNIT_TYPE_BARRACKS = 21


def make_pysc2_action_call(action_idx: int, ts, policy_type: str, pending: dict = None):
    """
    Converts a King's discrete action_idx into a PySC2 FunctionCall.
    Handles two-step actions (e.g., select_unit then perform action) via 'pending'.
    
    Args:
        action_idx (int): The discrete action ID chosen by the King's policy.
        ts (TimeStep): The current PySC2 TimeStep observation.
        policy_type (str): "combat" or "resource" to identify the King.
        pending (dict, optional): Dictionary indicating a pending two-step action. Defaults to None.
    
    Returns:
        tuple: (pysc2_function_call, next_pending_action)
               - pysc2_function_call: The FunctionCall object to execute in the environment.
               - next_pending_action: A dictionary for the next step's pending action, or None.
    """
    obs = ts.observation
    fus = obs.feature_units # Feature units (list of Unit objects)
    avail = set(obs.available_actions) # Set of available PySC2 function IDs

    if policy_type == "combat":
        action_list = COMBAT_ACTION_LIST
        # action_index = COMBAT_ACTION_INDEX # Not directly used here, but good for reference
    elif policy_type == "resource":
        action_list = RESOURCE_ACTION_LIST
        # action_index = RESOURCE_ACTION_INDEX
    else:
        # Should not happen if called correctly
        print(f"Error: Unknown policy_type '{policy_type}' in make_pysc2_action_call.")
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None

    # --- 1. Handle Pending Actions ---
    # If there's a pending action from a previous step (e.g., unit was selected, now execute command)
    if pending:
        if pending['action_fn'] in avail:
            # If the pending function is available, execute it
            args = pending['args']
            # Safely get coordinates if they are part of the arguments
            if len(args) > 1 and isinstance(args[1], list) and len(args[1]) == 2:
                x, y = args[1]
                return actions.FunctionCall(pending['action_fn'], [args[0], safe_coords(x, y)]), None
            else:
                return actions.FunctionCall(pending['action_fn'], args), None
        else:
            # If pending action is no longer available, fall back to no_op and clear pending
            print(f"Warning: Pending function {actions.FUNCTIONS[pending['action_fn']].name} not available. Falling back to no_op.")
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None

    # --- 2. Determine Current Action based on action_idx and policy_type ---
    action_name = action_list[action_idx]

    # Handle 'do_nothing' for any policy
    if action_name == 'do_nothing':
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None

    # --- Combat King Specific Actions ---
    if policy_type == "combat":
        if action_name == 'select_army':
            if actions.FUNCTIONS.select_army.id in avail:
                return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]]), None # [[0]] for select all (replacing current selection)
            
        elif action_name == 'attack_group_target':
            if actions.FUNCTIONS.Attack_screen.id in avail:
                enemies = [u for u in fus if u.alliance == features.PlayerRelative.ENEMY]
                if enemies:
                    # Pick a random enemy to target (simple strategy for now)
                    target = random.choice(enemies)
                    # [[0]] for default control group (usually current selection)
                    return actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [[0], safe_coords(target.x, target.y)]), None
            
        elif action_name == 'move_group_target':
            if actions.FUNCTIONS.Move_screen.id in avail:
                # Pick a random point on screen to move to
                x, y = np.random.randint(0, SCREEN_SIZE), np.random.randint(0, SCREEN_SIZE)
                return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], safe_coords(x, y)]), None

    # --- Resource King Specific Actions ---
    elif policy_type == "resource":
        if action_name == 'train_scv':
            # Check if Train_SCV_quick is available immediately
            if actions.FUNCTIONS.Train_SCV_quick.id in avail:
                return actions.FunctionCall(actions.FUNCTIONS.Train_SCV_quick.id, [[0]]), None
            else:
                # If not available, try to select a Command Center first
                ccs = [u for u in fus if u.alliance == features.PlayerRelative.SELF and u.unit_type == UNIT_TYPE_COMMAND_CENTER]
                if ccs and actions.FUNCTIONS.select_point.id in avail:
                    cc = random.choice(ccs)
                    select_action = actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], safe_coords(cc.x, cc.y)])
                    # Set up pending action to train SCV in the next step
                    next_pending = {'action_fn': actions.FUNCTIONS.Train_SCV_quick.id, 'args': [[0]]}
                    return select_action, next_pending

        elif action_name == 'build_supply_depot':
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id in avail:
                # Find a buildable location
                buildable_spots = np.argwhere(obs.feature_screen.buildable == 1)
                if buildable_spots.size > 0:
                    y, x = random.choice(buildable_spots)
                    return actions.FunctionCall(actions.FUNCTIONS.Build_SupplyDepot_screen.id, [[0], safe_coords(x, y)]), None
            else:
                # Try to select an SCV first
                scvs = [u for u in fus if u.alliance == features.PlayerRelative.SELF and u.unit_type == UNIT_TYPE_SCV]
                if scvs and actions.FUNCTIONS.select_point.id in avail:
                    scv = random.choice(scvs)
                    select_action = actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], safe_coords(scv.x, scv.y)])
                    buildable_spots = np.argwhere(obs.feature_screen.buildable == 1)
                    if buildable_spots.size > 0:
                        y, x = random.choice(buildable_spots)
                        next_pending = {'action_fn': actions.FUNCTIONS.Build_SupplyDepot_screen.id, 'args': [[0], [x, y]]}
                        return select_action, next_pending
        
        elif action_name == 'build_barracks':
            if actions.FUNCTIONS.Build_Barracks_screen.id in avail:
                buildable_spots = np.argwhere(obs.feature_screen.buildable == 1)
                if buildable_spots.size > 0:
                    y, x = random.choice(buildable_spots)
                    return actions.FunctionCall(actions.FUNCTIONS.Build_Barracks_screen.id, [[0], safe_coords(x, y)]), None
            else:
                scvs = [u for u in fus if u.alliance == features.PlayerRelative.SELF and u.unit_type == UNIT_TYPE_SCV]
                if scvs and actions.FUNCTIONS.select_point.id in avail:
                    scv = random.choice(scvs)
                    select_action = actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], safe_coords(scv.x, scv.y)])
                    buildable_spots = np.argwhere(obs.feature_screen.buildable == 1)
                    if buildable_spots.size > 0:
                        y, x = random.choice(buildable_spots)
                        next_pending = {'action_fn': actions.FUNCTIONS.Build_Barracks_screen.id, 'args': [[0], [x, y]]}
                        return select_action, next_pending
        
        elif action_name == 'harvest_gather':
            if actions.FUNCTIONS.Harvest_Gather_screen.id in avail:
                minerals = [u for u in fus if u.unit_type == UNIT_TYPE_MINERAL_FIELD]
                if minerals:
                    target = random.choice(minerals)
                    return actions.FunctionCall(actions.FUNCTIONS.Harvest_Gather_screen.id, [[0], safe_coords(target.x, target.y)]), None
            else:
                scvs = [u for u in fus if u.alliance == features.PlayerRelative.SELF and u.unit_type == UNIT_TYPE_SCV]
                if scvs and actions.FUNCTIONS.select_point.id in avail:
                    scv = random.choice(scvs)
                    select_action = actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], safe_coords(scv.x, scv.y)])
                    minerals = [u for u in fus if u.unit_type == UNIT_TYPE_MINERAL_FIELD]
                    if minerals:
                        target = random.choice(minerals)
                        next_pending = {'action_fn': actions.FUNCTIONS.Harvest_Gather_screen.id, 'args': [[0], [target.x, target.y]]}
                        return select_action, next_pending

    # --- Default Fallback: If action not handled or not available, return no_op ---
    return actions.FunctionCall(actions.FUNCTIONS.no_op.id, []), None

def get_legal_actions_for_policy(ts, policy_type: str):
    """
    Determines legal actions for a given policy type based on available PySC2 actions.
    
    Args:
        ts (TimeStep): The current PySC2 TimeStep observation.
        policy_type (str): "combat" or "resource".
        
    Returns:
        list: A list of legal action indices for the specified policy.
    """
    avail = set(ts.observation.available_actions)
    fus = ts.observation.feature_units
    
    if policy_type == "combat":
        action_list = COMBAT_ACTION_LIST
        action_index = COMBAT_ACTION_INDEX
    elif policy_type == "resource":
        action_list = RESOURCE_ACTION_LIST
        action_index = RESOURCE_ACTION_INDEX
    else:
        return [0] # Default to do_nothing if policy type is unknown

    legal = [action_index['do_nothing']] # No-op is always legal

    if policy_type == "combat":
        # Combat actions
        if actions.FUNCTIONS.select_army.id in avail:
            legal.append(action_index['select_army'])
        # If army is selected or combat units exist, these might be available
        if actions.FUNCTIONS.Attack_screen.id in avail:
            legal.append(action_index['attack_group_target'])
            # Assuming 'attack_screen' and 'move_screen' from base action_list are implicitly handled by specific units
            # For this simplified model, we only care about 'group' targets for combat
        if actions.FUNCTIONS.Move_screen.id in avail:
            legal.append(action_index['move_group_target'])

    elif policy_type == "resource":
        # Resource actions
        # Check specific functions and if prerequisites (units, buildings) exist
        
        # Train SCV
        if actions.FUNCTIONS.Train_SCV_quick.id in avail: # Direct availability
            legal.append(action_index['train_scv'])
        else: # Check if Command Center is present to select it for training
            if any(u.unit_type == UNIT_TYPE_COMMAND_CENTER and u.alliance == features.PlayerRelative.SELF for u in fus):
                 if actions.FUNCTIONS.select_point.id in avail: # If select point is available for CC
                     legal.append(action_index['train_scv']) # Allow action to attempt selection sequence

        # Build Supply Depot
        if actions.FUNCTIONS.Build_SupplyDepot_screen.id in avail:
            legal.append(action_index['build_supply_depot'])
        else: # Check for SCV to build
            if any(u.unit_type == UNIT_TYPE_SCV and u.alliance == features.PlayerRelative.SELF for u in fus):
                if actions.FUNCTIONS.select_point.id in avail: # If select point is available for SCV
                    legal.append(action_index['build_supply_depot']) # Allow action to attempt selection sequence

        # Build Barracks
        if actions.FUNCTIONS.Build_Barracks_screen.id in avail:
            legal.append(action_index['build_barracks'])
        else: # Check for SCV to build
            if any(u.unit_type == UNIT_TYPE_SCV and u.alliance == features.PlayerRelative.SELF for u in fus):
                if actions.FUNCTIONS.select_point.id in avail: # If select point is available for SCV
                    legal.append(action_index['build_barracks']) # Allow action to attempt selection sequence
        
        # Harvest/Gather
        if actions.FUNCTIONS.Harvest_Gather_screen.id in avail:
            legal.append(action_index['harvest_gather'])
        else: # Check for SCV and Mineral Field
            if any(u.unit_type == UNIT_TYPE_SCV and u.alliance == features.PlayerRelative.SELF for u in fus) \
            and any(u.unit_type == UNIT_TYPE_MINERAL_FIELD for u in fus):
                if actions.FUNCTIONS.select_point.id in avail: # If select point is available for SCV
                    legal.append(action_index['harvest_gather']) # Allow action to attempt selection sequence

    return list(set(legal)) # Return unique legal actions

