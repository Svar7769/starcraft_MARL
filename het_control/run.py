#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from het_control.callback import *
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig
from smacv2 import render_callback_smac, SMACv2Task


def get_experiment(cfg: DictConfig) -> Experiment:
    # register custom model
    benchmarl.models.model_config_registry.update(
        {"hetcontrolmlpempirical": HetControlMlpEmpiricalConfig}
    )

    # load hydra choices
    choices = HydraConfig.get().runtime.choices
    task_name = choices.task
    algorithm_name = choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    # load configs
    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    # adjust probabilistic settings
    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False

    # create Experiment
    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=[
            SndCallback(),
            NormLoggerCallback(),
            ActionSpaceLoss(
                use_action_loss=cfg.use_action_loss,
                action_loss_lr=cfg.action_loss_lr
            ),
        ] + (
            [TagCurriculum(cfg.simple_tag_freeze_policy_after_frames, cfg.simple_tag_freeze_policy)]
            if task_name == "vmas/simple_tag" else []
        ),
    )

    # attach SMACv2 render callback if using smacv2 task
    if task_name.startswith("smacv2"):
        experiment.callbacks.append(
            lambda exp, data: render_callback_smac(exp, exp.task.env, data)
        )
    # attach VMAS render callback
    if task_name == "vmas/navigation":
        VmasTask.render_callback = render_callback

    return experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
