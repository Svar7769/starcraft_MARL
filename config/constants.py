from pysc2.lib import actions, features

# Action Function IDs
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

# Feature Indices
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_SELF = 1

# Unit Type IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

# Queue Constants
_NOT_QUEUED = [0]
_QUEUED = [1]

# High-level Smart Action Names (used in some agents)
smart_actions = [
    'donothing', 'selectscv', 'buildsupplydepot', 'buildbarracks',
    'selectbarracks', 'buildmarine', 'buildscv', 'selectarmy',
    'attack', 'selectgroup', 'move'
]

# Model Paths
MODEL_DIR = "model"
CHECKPOINT_PATH = f"{MODEL_DIR}/episode_checkpoint.txt"
REWARD_LOG_PATH = f"{MODEL_DIR}/reward_log.csv"
RESUME_TRAINING = False
