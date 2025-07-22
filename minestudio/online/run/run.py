from numpy import roll
from omegaconf import OmegaConf
import hydra
import logging
from minestudio.online.rollout.rollout_manager import RolloutManager
from minestudio.online.utils.rollout import get_rollout_manager
from minestudio.online.utils.train.training_session import TrainingSession
import ray
import wandb
import uuid
import torch
from minestudio.online.rollout.start_manager import start_rolloutmanager
from minestudio.online.trainer.start_trainer import start_trainer
from minestudio.simulator import MinecraftSim
from minestudio.models import load_vpt_policy, VPTPolicy
from minestudio.utils import get_compute_device
from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback
)

from minestudio.online.run.config.config_shoot import env_generator
env_generator_1=env_generator
    

def policy_generator():
    from minestudio.models import load_vpt_policy, VPTPolicy
    return load_vpt_policy(
      model_path="C:/Users/Neo/Desktop/MineStudio&RL_GPT/vpt model+weights/example/2x.model",
      weights_path="C:/Users/Neo/Desktop/MineStudio&RL_GPT/vpt model+weights/example/foundation-model-2x.weights"
  ).to(get_compute_device())


if __name__=='__main__':
    config_name = "config_shoot"
    print("\033[1;32m Starting training session WITH CONFIG: " + config_name + " \033[0m")
    module_name = "minestudio.online.run.config."+config_name

    import importlib
    module = importlib.import_module(module_name)

    ray.init()

    online_dict = {
    "trainer_name": "PPOTrainer",
    "detach_rollout_manager": True,
    "rollout_config": {
        "num_rollout_workers": 1,
        "num_gpus_per_worker": 0.1,
        "num_cpus_per_worker": 1,
        "fragment_length": 256,
        "to_send_queue_size": 8,
        "worker_config": {
            "num_envs": 4,
            "batch_size": 8,
            "restart_interval": 3600,  # 1h
            "video_fps": 20,
            "video_output_dir": "output/videos",
        },
        "replay_buffer_config": {
            "max_chunks": 1200,
            "max_reuse": 1,
            "max_staleness": 1,
            "fragments_per_report": 40,
            "fragments_per_chunk": 1,
            "database_config": {
                "path": "output/replay_buffer_cache",
                "num_shards": 8,
            },
        },
        "episode_statistics_config": {},
    },
    "train_config": {
        "num_workers": 1,
        "num_gpus_per_worker": 0.5,
        "num_iterations": 4000,
        "vf_warmup": 0,
        "learning_rate": 0.00002,
        "anneal_lr_linearly": False,
        "weight_decay": 0.04,
        "adam_eps": 1e-8,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 200, #200
        "gradient_accumulation": 10,  # TODO: check
        "epochs_per_iteration": 1,  # TODO: check
        "context_length": 64,
        "discount": 0.999,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "clip_vloss": False,  # TODO: check
        "max_grad_norm": 5,  # ????
        "zero_initial_vf": True,
        "ppo_policy_coef": 1.0,
        "ppo_vf_coef": 0.5,  # TODO: check
        "kl_divergence_coef_rho": 0.2,
        "entropy_bonus_coef": 0.0,
        "coef_rho_decay": 0.9995,
        "log_ratio_range": 50,  # for numerical stability
        "normalize_advantage_full_batch": True,  # TODO: check!!!
        "use_normalized_vf": True,
        "num_readers": 4,
        "num_cpus_per_reader": 0.1,
        "prefetch_batches": 2,
        "save_interval": 10,
        "keep_interval": 40,
        "record_video_interval": 2,
        "enable_ref_update": True,
        "resume": None, #"/scratch/hekaichen/tmpdir/ray/session_2024-12-12_21-10-40_218613_2665801/artifacts/2024-12-12_21-10-58/TorchTrainer_2024-12-12_21-10-58/working_dirs/TorchTrainer_8758b_00000_0_2024-12-12_21-10-58/checkpoints/150",
        "resume_optimizer": True,
        "save_path": "/scratch/hekaichen/workspace/MineStudio/minestudio/online/run/output"
    },

    "logger_config": {
        "project": "minestudio_online",
        "name": "bow_cow"
    },
}
    online_cfg = OmegaConf.create(online_dict)

    '''
    with open("/home/neo/MineStudio/minestudio/online/run/config/"+config_name+".py", "r") as f:
        whole_config = f.read()
        for line in f:
            print(line.strip())  # Use strip() to remove leading/trailing whitespace including newline
    '''            
    start_rolloutmanager(policy_generator, env_generator_1, online_cfg)
    start_trainer(policy_generator, env_generator_1, online_cfg, online_dict)

# training_session = None
# try:
#     training_session = ray.get_actor("training_session")
# except ValueError:
#     pass
# if training_session is not None:
#     logger.error("Trainer already running!")
#     exit()

# training_session = TrainingSession.options(name="training_session").remote(hyperparams=cfg, logger_config=cfg.logger_config) # type: ignore
# ray.get(training_session.get_session_id.remote()) # Assure that the session is created before the trainer
# ray.get(rollout_manager.update_training_session.remote())
# print("Making trainer")
# trainer = registry.get_trainer_class(online_cfg.trainer_name)(
#     rollout_manager=rollout_manager,
#     policy_generator=policy_generator,
#     env_generator=env_generator,
#     **online_cfg.train_config
# )
# trainer.fit()


import time
time.sleep(1000000)
