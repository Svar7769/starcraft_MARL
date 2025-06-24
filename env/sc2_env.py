import logging
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from omegaconf import DictConfig
from absl import flags

logger = logging.getLogger(__name__)

# Only parse absl flags once
_ABSL_FLAGS_PARSED = False

class SC2EnvsMulti:
    def __init__(self, cfg: DictConfig):
        global _ABSL_FLAGS_PARSED
        self.cfg = cfg

        # Parse ABSL flags only once
        if not _ABSL_FLAGS_PARSED:
            try:
                flags.FLAGS(['main.py'])  # Dummy parsing to avoid the error
            except flags.DuplicateFlagError:
                pass
            _ABSL_FLAGS_PARSED = True

        if cfg.pysc2.sc2_run_config:
            flags.FLAGS.sc2_run_config = cfg.pysc2.sc2_run_config
        flags.FLAGS.sc2_version = cfg.pysc2.sc2_version

        logger.info("Initializing %d SC2 env(s)...", cfg.hyperparameters.nb_actors)
        self.nb = cfg.hyperparameters.nb_actors
        self.envs = [self._make_env() for _ in range(self.nb)]
        self.obs = [env.reset()[0] for env in self.envs]
        self.done = [False] * self.nb
        self.episode_count = 0

    def _make_env(self):
        try:
            return sc2_env.SC2Env(
                map_name=self.cfg.hyperparameters.map_name,
                players=[
                    sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
                ],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(
                        screen=self.cfg.hyperparameters.screen_size,
                        minimap=self.cfg.hyperparameters.minimap_size),
                    use_feature_units=True,
                    use_raw_units=False,
                    use_camera_position=True,
                    action_space=actions.ActionSpace.FEATURES
                ),
                step_mul=self.cfg.hyperparameters.step_mul,
                game_steps_per_episode=0,
                visualize=self.cfg.pysc2.render
            )
        except Exception as e:
            logger.error(f"Failed to create SC2Env: {e}")
            raise

    def step(self, i, action):
        try:
            timestep = self.envs[i].step([action])[0]
            self.obs[i] = timestep
            self.done[i] = timestep.last()

            if timestep.last():
                self.episode_count += 1
                score = timestep.observation["score_cumulative"][0]
                logger.info(f"🎯 Episode {self.episode_count} completed with score: {score}")

            return timestep
        except Exception as e:
            logger.error(f"Error during env step: {e}")
            raise

    def reset(self, i):
        try:
            self.obs[i] = self.envs[i].reset()[0]
            self.done[i] = False
            return self.obs[i]
        except Exception as e:
            logger.error(f"Error during env reset: {e}")
            raise

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing env: {e}")
        self.envs