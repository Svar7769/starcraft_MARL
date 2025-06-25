from pysc2.env import sc2_env
from pysc2.lib import actions, features
from config.spartan import MAP_NAME, SCREEN_SIZE, MINIMAP_SIZE, STEP_MUL, logger # Import from config.py

class SC2Envs: # Renamed from SC2EnvsMulti to SC2Envs for simplicity based on your structure
    def __init__(self, nb_actor):
        logger.info("Initializing %d SC2 env(s)...", nb_actor)
        self.nb = nb_actor
        self.envs = [self._make_env() for _ in range(nb_actor)]
        # Initial observation and done status for all environments
        self.obs = [env.reset()[0] for env in self.envs]
        self.done = [False] * nb_actor
        logger.info("All SC2 env(s) ready.")

    def _make_env(self):
        return sc2_env.SC2Env(
            map_name=MAP_NAME,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
            ],
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
            visualize=False # Keep False for training, can be True for evaluation
        )

    def step(self, i, action):
        timestep = self.envs[i].step([action])[0]
        self.obs[i] = timestep
        self.done[i] = timestep.last()
        return timestep

    def reset(self, i):
        self.obs[i] = self.envs[i].reset()[0]
        self.done[i] = False

    def close(self):
        for env in self.envs:
            env.close()