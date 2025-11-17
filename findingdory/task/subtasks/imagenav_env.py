from typing import TYPE_CHECKING, Dict, Optional, TYPE_CHECKING, Any, List, Type

import gym

import habitat
from habitat import Dataset
from habitat.gym.gym_wrapper import HabGymWrapper
from habitat.core.environments import RLTaskEnv

from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat_baselines.common.env_factory import VectorEnvFactory
from habitat.gym import make_gym_from_config
import torch
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size

if TYPE_CHECKING:
    from omegaconf import DictConfig
import random
import os


class ImageNavTaskEnv(RLTaskEnv):
    def get_reward(self, observations):
        return self._env.get_metrics()[self._reward_measure_name]


@habitat.registry.register_env(name="GymImageNavEnv")
class GymImageNavEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = ImageNavTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)
        
def read_scenes_by_group(file_path, group_id):
    scenes = []
    total_ep_count = 0

    # Open the file containing the scene-to-group mapping
    with open(file_path, 'r') as file:
        for line in file:
            # Each line contains a scene ID followed by a group ID
            scene_id, scene_group, ep_count = line.strip().split(',')
            
            # If the scene belongs to the given group ID, add it to the list
            if int(scene_group) == group_id:
                scenes.append(scene_id)
                total_ep_count += int(ep_count)
    
    return scenes, total_ep_count


class ImageNavVectorEnvFactory(VectorEnvFactory):
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
        distribute_envs_across_gpus=None
    ) -> VectorEnv:
        r"""Create VectorEnv object with specified config and env class type.
        To allow better performance, dataset are split into small ones for
        each individual env, grouped by scenes.
        """

        if distribute_envs_across_gpus is None:
            distribute_envs_across_gpus = enforce_scenes_greater_eq_environments  
        num_environments = config.habitat_baselines.num_environments
        configs = []
        dataset = make_dataset(config.habitat.dataset.type)
        scenes = list(config.habitat.dataset.content_scenes)
        if "*" in config.habitat.dataset.content_scenes:
            scenes = dataset.get_scenes_to_load(config.habitat.dataset)
            scenes = sorted(scenes)
            local_rank, world_rank, world_size = get_distrib_size()
            split_size = len(scenes)/world_size
            orig_size = len(scenes)
            scene_id_start = round(world_rank*split_size)
            scene_id_end = round((world_rank + 1)*split_size)
            scenes = scenes[scene_id_start:scene_id_end]
            scenes_ids = list(range(orig_size))[scene_id_start:scene_id_end]
            logger.warn(
                f"Loading {len(scenes)}/{orig_size}. IDs: {scenes_ids}"
            )

        if num_environments < 1:
            raise RuntimeError("num_environments must be strictly positive")

        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        random.shuffle(scenes)

        scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
        for idx in range(max(len(scene_splits), len(scenes))):
            scene_splits[idx % len(scene_splits)].append(scenes[idx % len(scenes)])

        logger.warn(
            f"Scene splits: {scene_splits}."
        )
        assert all(scene_splits)

        for env_index in range(num_environments):
            proc_config = config.copy()
            with read_write(proc_config):
                if distribute_envs_across_gpus:
                    if torch.cuda.is_available():  # For CUDA
                        proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = env_index % torch.cuda.device_count()
                    else:   # For MLX
                        proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0  
                task_config = proc_config.habitat
                task_config.seed = task_config.seed + env_index
                remove_measure_names = []
                if not is_first_rank:
                    # Filter out non rank0_measure from the task config if we are not on rank0.
                    remove_measure_names.extend(
                        task_config.task.rank0_measure_names
                    )
                if (env_index != 0) or not is_first_rank:
                    # Filter out non-rank0_env0 measures from the task config if we
                    # are not on rank0 env0.
                    remove_measure_names.extend(
                        task_config.task.rank0_env0_measure_names
                    )

                task_config.task.measurements = {
                    k: v
                    for k, v in task_config.task.measurements.items()
                    if k not in remove_measure_names
                }

                if len(scenes) > 0:
                    task_config.dataset.content_scenes = scene_splits[
                        env_index
                    ]

            configs.append(proc_config)

        vector_env_cls: Type[Any]
        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = ThreadedVectorEnv
        else:
            vector_env_cls = VectorEnv

        envs = vector_env_cls(
            make_env_fn=make_gym_from_config,
            env_fn_args=tuple((c,) for c in configs),
            workers_ignore_signals=workers_ignore_signals,
        )

        if config.habitat.simulator.renderer.enable_batch_renderer:
            envs.initialize_batch_renderer(config)

        return envs