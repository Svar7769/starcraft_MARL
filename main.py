import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
from absl import flags
from env.sc2_env import SC2EnvsMulti
from models.actor_critic import ActorCritic
from ppo.trainer import PPOTrainer

# Setup logging
logger = logging.getLogger(__name__)

def init_absl_flags():
    # This is needed to prevent ABSL flag errors
    if 'absl.flags' in sys.modules:
        del sys.modules['absl.flags']
    flags.FLAGS(['main.py'])  # Dummy parsing

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Initialize ABSL flags before anything else
    init_absl_flags()
    
    # Print config
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Initialize model
    model = ActorCritic(
        in_channels=2,  # player_relative + unit_type
        nb_actions=len(cfg.action_list),
        screen_size=cfg.hyperparameters.screen_size
    ).to(cfg.hyperparameters.device)
    
    # Initialize trainer (which will handle environment creation)
    trainer = PPOTrainer(
        model=model,
        device=cfg.hyperparameters.device,
        cfg=cfg  # Pass the full config
    )
    
    # Training options
    if cfg.training.load_checkpoint:
        trainer = PPOTrainer.load_from_checkpoint(
            cfg.training.checkpoint_path,
            device=cfg.hyperparameters.device,
            cfg=cfg
        )
    
    if cfg.training.multi_map:
        # Train sequentially on multiple maps
        for map_name in cfg.training.map_sequence:
            trainer.switch_map(map_name)
            trainer.train()
    else:
        # Single map training
        trainer.train()

if __name__ == "__main__":
    # Set environment variable for full error reporting
    os.environ['HYDRA_FULL_ERROR'] = '1'
    main()