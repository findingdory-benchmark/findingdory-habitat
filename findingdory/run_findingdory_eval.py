import os
from tqdm import tqdm
from typing import Dict, Optional
import json 

import hydra
import numpy as np
import random
import torch
from omegaconf import DictConfig

from habitat.config.default import patch_config
from habitat.core.env import Env
from habitat.core.logging import logger

import findingdory.config
import findingdory.policies
import findingdory.task
from findingdory.dataset.data_collector import DataCollector
from findingdory.task.utils import save_imagenav_rollouts
from findingdory.dataset.utils import save_mp4

from findingdory.policies.end_to_end.qwen_imagenav_agent import QwenImageNavAgent
from findingdory.policies.heuristic.vlm_mapper import VLMMapperAgent


def save_metrics_json(fname, per_ep_task_metrics):
    with open(fname, 'w') as f:
            json.dump(per_ep_task_metrics, f, indent=2)

def save_metrics(env, per_ep_metrics, fname):                
    metrics = env.get_metrics()
    metrics_to_print = {}
    for m, v in metrics.items():
        if m == "top_down_map" or m == "fog_of_war_mask":
            continue
        metrics_to_print[m] = v

    per_ep_metrics.update({env.current_episode.episode_id : env._task.metrics_per_task})
    save_metrics_json(fname, env._task.metrics_per_task)

def evaluate(
    env,
    agent,
    num_episodes: Optional[int] = None,
    cfg: DictConfig = None,
) -> Dict[str, float]:
    r"""..

    :param agent: agent to be evaluated in environment.
    :param num_episodes: count of number of episodes for which the
        evaluation should be run.
    :return: dict containing metrics tracked by environment.
    """
    data_collector = DataCollector(env, cfg.habitat_baselines.video_dir)

    if num_episodes is None:
        num_episodes = len(env.episodes)
    else:
        assert num_episodes <= len(env.episodes), (
            "num_episodes({}) is larger than number of episodes "
            "in environment ({})".format(num_episodes, len(env.episodes))
        )

    assert num_episodes > 0, "num_episodes should be greater than 0"
    
    logger.info("Evaluating Episode IDs: {}".format(cfg.habitat.dataset.episode_ids))

    count_episodes = 0

    pbar = tqdm(total=num_episodes)
    
    current_task_id = None
    per_ep_metrics = {}
    
    ep_ids = ""
    for ep in env.episodes:
        ep_ids = ep_ids + "_" + ep.episode_id

    while count_episodes < num_episodes:
        observations = env.reset()
            
        agent.episode_index = int(env.current_episode.episode_id)
        agent.task_id = env._task.current_task_id

        agent.reset()

        data_collector.reset()

        data_collector.update_metadata(observations)

        pbar1 = tqdm(total=env._max_episode_steps, miniters=100)
        # Perform data collection in current episode
        while not env.episode_over:
            
            current_task_id = env._task.current_task_id
            
            if isinstance(agent, QwenImageNavAgent):
                action, imagenav_rollouts = agent.act(observations)
            elif isinstance(agent, VLMMapperAgent):
                action, sem_map_result = agent.act(observations)
            else:
                action = agent.act(observations)

            observations = env.step(action)
            
            # Reset internal agent variables to attempt the next instruction in queue
            if env._task._switch_to_new_task:
                logger.info("Switching imagenav policy to new task....")
                
                if isinstance(agent, QwenImageNavAgent):
                    if imagenav_rollouts is not None and len(imagenav_rollouts) > 0:
                        logger.info("Length of imagenav rollouts: {}".format(len(imagenav_rollouts)))
                        file_path = os.path.join(cfg.habitat_baselines.agent.config.output_folder, 'ep_id_' + str(agent.episode_index) + '_' + current_task_id + '_imagenav_rollouts.mp4')
                        save_imagenav_rollouts(imagenav_rollouts, file_path)
                        logger.info("Saved imagenav rollouts to: {}".format(file_path))
                    
                agent.reset_new_task(env._task.current_task_id)
                print("Reset agent for new task !")

            if isinstance(agent, VLMMapperAgent):     
                data_collector.update_metadata(observations, sem_map_result)
            pbar1.update(1)
            
        assert len(env._task._instructions_to_evaluate) == 0, "Some instructions still remain to be evaluated !"        
        
        if isinstance(agent, QwenImageNavAgent):
            if imagenav_rollouts is not None and len(imagenav_rollouts) > 0:
                file_path = os.path.join(cfg.habitat_baselines.agent.config.output_folder, 'ep_id_' + env.current_episode.episode_id + '_' + current_task_id + '_imagenav_rollouts.mp4')
                save_imagenav_rollouts(imagenav_rollouts, file_path)
                print("Saved imagenav rollouts to: ", file_path)

        metrics = env.get_metrics()
        metrics_to_print = {}
        for m, v in metrics.items():
            if m == "top_down_map" or m == "fog_of_war_mask":
                continue
            metrics_to_print[m] = v
        cur_task_id = env._task._sim.ep_info.instructions[env._task._chosen_instr_idx].task_id
        env._task.metrics_per_task.update({cur_task_id : metrics_to_print})        
        print(f"Metrics for task_ID {cur_task_id}: {metrics_to_print}")

        fname = os.path.join(cfg.habitat_baselines.agent.config.output_folder, "metrics_ep_ids_" + env.current_episode.episode_id + '.json.gz')
        save_metrics(env, per_ep_metrics, fname)

        # Save the data collector mapping visualization output
        folder_name = os.path.join(cfg.habitat_baselines.agent.config.output_folder, env.current_episode.episode_id)
        os.makedirs(folder_name, exist_ok=True)
        save_mp4(data_collector._images, folder_name)
   
        count_episodes += 1
        pbar.update(1)
        
    return per_ep_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="baseline/qwen_mapper",
)
def main(cfg: "DictConfig"):
    cfg = patch_config(cfg)

    set_seed(cfg.habitat.seed)
    
    env = Env(config=cfg)
    agent = hydra.utils.instantiate(cfg.habitat_baselines.agent)

    start_episode = 0
    for i in range(start_episode):
        env.reset()

    per_ep_task_metrics = evaluate(env, agent, num_episodes=None, cfg=cfg)
    logger.info("Evaluation Complete")
    logger.info("Per ep metrics: {}".format(per_ep_task_metrics))


if __name__ == "__main__":
    main()
