from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import numpy as np
import random
from collections import defaultdict
from qlearn.q_table import QLearningTable

# Macro actions
ACTIONS = ("do_nothing", "harvest_minerals", "build_supply_depot", "build_barracks", "train_marine", "attack")

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.actions = ACTIONS
        self.qtable = QLearningTable(actions=self.actions)
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_action = None

    # Helper functions (copied from run_agent.py)
    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.build_progress == 100 and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1) if units_xy else np.array([])

    # State representation similar to run_agent.py
    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
        barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barracks = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        enemy_units = self.get_enemy_units_by_type(obs, units.Terran.Marine) + \
                      self.get_enemy_units_by_type(obs, units.Terran.SCV) + \
                      self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
        enemy_scv = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_cc = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
        queued_marines = 0
        for bar in barracks:
            queued_marines += bar.order_length

        supply_limit = obs.observation.player[4]
        supply_used = obs.observation.player[3]
        supply_left = supply_limit - supply_used
        minerals = obs.observation.player[1]
        vespene = obs.observation.player[2]
        can_afford_depot = minerals >= 100
        can_afford_barracks = minerals >= 150
        can_afford_marine = minerals >= 50

        state = (
            len(scvs), len(marines), len(depots), len(completed_depots),
            len(barracks), len(completed_barracks), len(enemy_units),
            len(enemy_marines), len(enemy_scv), len(enemy_cc),
            queued_marines, supply_limit, supply_used, supply_left,
            minerals, vespene, int(can_afford_depot), int(can_afford_barracks), int(can_afford_marine),
            int(any([unit.order_length > 0 for unit in barracks])),
            int(any([unit.order_length > 0 for unit in depots]))
        )
        return str(state)

    def step(self, obs):
        super().step(obs)
        if obs.last():
            # Learn from the last transition
            if self.prev_state is not None and self.prev_action is not None:
                self.qtable.learn(self.prev_state, self.prev_action, obs.reward, 'terminal')
            return actions.RAW_FUNCTIONS.no_op()

        state = self.get_state(obs)
        action = self.qtable.choose_action(state)

        # Learn from the previous transition
        if self.prev_state is not None and self.prev_action is not None:
            self.qtable.learn(self.prev_state, self.prev_action, obs.reward, state)

        self.prev_state = state
        self.prev_action = action

        # Dispatch the action
        return getattr(self, action)(obs)

    # Action implementations
    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        mineral_patches = [unit for unit in obs.observation.raw_units if unit.unit_type in [units.Neutral.MineralField, units.Neutral.MineralField750]]
        if scvs and mineral_patches:
            scv = random.choice(scvs)
            mineral = random.choice(mineral_patches)
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, [mineral.tag])
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if scvs:
            scv = random.choice(scvs)
            x = random.randint(0, 83)
            y = random.randint(0, 83)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, [x, y])
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if scvs:
            scv = random.choice(scvs)
            x = random.randint(0, 83)
            y = random.randint(0, 83)
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, [x, y])
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        barracks = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        if barracks:
            barrack = random.choice(barracks)
            return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        enemy_units = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        if marines and enemy_units:
            marine = random.choice(marines)
            enemy = random.choice(enemy_units)
            return actions.RAW_FUNCTIONS.Attack_unit("now", marine.tag, [enemy.tag])
        return actions.RAW_FUNCTIONS.no_op()