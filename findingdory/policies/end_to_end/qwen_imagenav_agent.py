#!/usr/bin/env python3
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import magnum as mn

from findingdory.policies.llm.qwen_agent import QwenAgent
import findingdory.task
from collections import OrderedDict

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size
from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
import hydra
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import get_action_space_info
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    is_continuous_action_space,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from findingdory.utils import get_device
from findingdory.dataset.utils import magnum_to_python_quaternion


class QwenImageNavAgent(QwenAgent):
    """
    ImageNav infernece agent for Habitat environments. Uses VLM for high-level image goal selection
    """

    def __init__(self, config) -> None:
        """
        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        super().__init__(config)
        self.episode_index = 0
        self.task_id = None
        
        self._init_imagenav_agent(config.imagenav_ckpt_path, config.eval_dataset_path)
        self._set_imagenav_goals = False
        self._change_vlm_selected_frame = config.change_vlm_selected_frame
        self._imagenav_max_steps = config.imagenav_max_steps
        self._delay_between_tasks = config.delay_between_tasks
        
    def reset(self):
        super().reset()
        self._imagenav_rollouts = []
        self._agent_in_nav_mode = []
        self._cur_subgoal_idx = None
        self._set_imagenav_goals = False
        self.nav_goal_images = []
        self._vlm_frames_processed = False
        
        if not os.path.exists(self.output_folder):
            # Create the new directory
            os.makedirs(self.output_folder)
        
        self._reset_policy_tensors()
        
    def reset_new_task(self, task_id):
        self.action_index = 0
        self.task_id = task_id
        self.output_folder_with_episode_index = os.path.join(
            self.output_folder, f"ep_{self.episode_index:04d}"
        )
        self.output_folder_with_episode_task_index = os.path.join(
            self.output_folder, f"ep_{self.episode_index:04d}_" + self.task_id
        )
        self._imagenav_rollouts = []
        self._cur_subgoal_idx = None
        self._set_imagenav_goals = False
        self.nav_goal_images = []
                
        self._reset_policy_tensors()

        # Sleep for "N" seconds to prevent violating Gemini tokens per minute limit in free tier
        time.sleep(self._delay_between_tasks)
        
        print("Reset low level policy to evaluate next instruction in queue !")
    
    def _reset_policy_tensors(self):
        # Initialise RL policy related tensors        
        action_shape, discrete_actions = get_action_space_info(
            self.imagenav_agent.actor_critic.policy_action_space
        )

        # Hardcode to use single env during evaluation
        num_envs = 1
        self.current_episode_reward = torch.zeros(num_envs, 1, device="cpu")

        self.test_recurrent_hidden_states = torch.zeros(
            (
                self.imagenav_cfg.habitat_baselines.num_environments,
                *self.imagenav_agent.actor_critic.hidden_state_shape,
            ),
            device=self.device,
        )

        self.hidden_state_lens = self.imagenav_agent.actor_critic.hidden_state_shape_lens
        self.action_space_lens = self.imagenav_agent.actor_critic.policy_action_space_shape_lens

        self.prev_actions = torch.zeros(
            self.imagenav_cfg.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        # We permanently set not_dones to True
        self.not_done_masks = torch.ones(
            self.imagenav_cfg.habitat_baselines.num_environments,
            *self.imagenav_agent.masks_shape,
            device=self.device,
            dtype=torch.bool,
        )

    def update_to_nav_mode_frame(self, nav_index):
        """Find the nearest previous index that corresponds to an image taken in navigation mode."""
        for i in range(nav_index - 1, -1, -1):  # Start from `nav_index - 1` and go backwards
            if self._agent_in_nav_mode[i]:
                return i
        return 0  # Fallback to the first index if no navigation mode image is found

    def act(self, obs):
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        
        # If the agent is in manipulation mode, the base is turned by +90 in the manipulation mode action step() so we need to correct for that
        if obs["manipulation_mode"]:
            agent_state = obs["agent_state"]
            quat_list = [
                agent_state.rotation.x,
                agent_state.rotation.y,
                agent_state.rotation.z,
                agent_state.rotation.w
            ]
            goal_quat = mn.Quaternion(mn.Vector3(quat_list[:3]), quat_list[3])
            turn_angle = -np.pi / 2  # Turn left by -90 degrees
            rot_quat = mn.Quaternion(
                mn.Vector3(0, np.sin(turn_angle / 2), 0),
                np.cos(turn_angle / 2),
            )
            final_goal_quat = rot_quat * goal_quat
            agent_state.rotation = magnum_to_python_quaternion(final_goal_quat)
        else:
            agent_state = obs["agent_state"]
            
        # Cache the observations and agent states for subsequent high level goal success verification
        # We only add observations when data collection is being done as we dont need to cache the observations/states when the low level policy operates
        if not obs["can_take_action"]:
            self._observations.append(obs)
            self._agent_states.append(agent_state)
            self._agent_in_nav_mode.append(not obs["manipulation_mode"])

        lang_goal = obs["lang_memory_goal"]

        if obs["can_take_action"]:
            # Run VLM inference for subgoal prediction
            if self.action_index % self.vlm_query_freq == 0:
                
                if not self._vlm_frames_processed:
                    self._frames = self.process_frames_for_vlm(save_images=False)
                    self._vlm_frames_processed = True
                
                self.nav_indices  = self.run_vlm(self._frames, lang_goal)
                self.num_nav_targets = len(self.nav_indices)
                                    
                # The predicted VLM subgoals need to be verified for task success -> so we return the predicted subgoals as an "action_dict" and pass it to the task for PDDL verification
                # Clip nav indices to valid range
                print("--------------------------> Frame index returned by VLM: ", self.nav_indices)
                clipped_indices = [max(0, min(idx, len(self._frame_num_to_original_frame_num) - 1)) for idx in self.nav_indices]
                output_nav_indices = [self._frame_num_to_original_frame_num[idx] for idx in clipped_indices]
                self._clipped_indices = clipped_indices
                
                print("--------------------------> Predicted (clipped) frame indices mapped to original indices: ", output_nav_indices)

                # The predicted VLM subgoals need to be verified for task success -> so we return the predicted subgoals as an "action_dict" and pass it to the task for PDDL verification
                action = {
                    "action": "high_level_policy_action",
                    "action_args": {
                        "nav_indices": output_nav_indices,            # Pass navigation goal information to task
                        "nav_goal_states": [self._agent_states[idx] for idx in clipped_indices] if clipped_indices else [],    # Store corresponding agent states for the selected navigation goals
                        "nav_mode_flag": [self._agent_in_nav_mode[idx] for idx in clipped_indices] if clipped_indices else []
                    },
                }
                self.action_index += 1
                return action, None 

            assert len(self.nav_indices) > 0, "Trying to activate imagenav policy but could not find any high level goals selected by the VLM. This implies high level goal verification should have failed and the task should have been stopped !"

            # If the high level subgoals were predicted correctly, we check if the subgoal indices returned by VLM correspond to images taken while in manipulation mode
            # If yes, then we find the nearest frame index which was taken in navigation_mode as the imagenav policy is trained only for navigation_mode image goals
            if not self._set_imagenav_goals:
                if self._change_vlm_selected_frame:
                    self.nav_goal_images = []
                    updated_nav_goal_indices = []
                    for i, frame_idx in enumerate(self._clipped_indices):
                        if not self._agent_in_nav_mode[frame_idx]:
                            # If image was taken in manipulation, then find the just previous goal frame index which is taken in navigation mode
                            new_idx = self.update_to_nav_mode_frame(frame_idx)
                            updated_nav_goal_indices.append(new_idx)
                        else:
                            updated_nav_goal_indices.append(self._clipped_indices[i])
                            
                    self._clipped_indices = updated_nav_goal_indices            
                    print("--------------------------> Updated goal indices for image goal navigation: ", self._clipped_indices)
                    
                else:
                    print("--------------------------> NOT Updating indices for image goal nav phase !: ", self._clipped_indices)

                # Append the goal image for each valid index
                for index in self._clipped_indices:
                    self.nav_goal_images.append(self._observations[index]["head_rgb"])
                
                self._set_imagenav_goals = True
                                 
            # Start image goal navigation if VLM successfully located the goal frames                
            if self._cur_subgoal_idx is None:
                print("Beginning imagenav inference...")
                self._cur_subgoal_idx = 0
            
            # This implies that the imagenav policy has tried reachign all the subgoals so we just invoke STOP
            if self._cur_subgoal_idx >= len(self.nav_goal_images):
                action = {
                    "action": ("pddl_intermediate_stop"),
                    "action_args": {},
                }
                return action, self._imagenav_rollouts
        
            # Prepare the policy inference batch input
            new_obs = OrderedDict()
            new_obs['head_rgb'] = obs['head_rgb']
            new_obs['imagegoal_rotation'] = self.nav_goal_images[self._cur_subgoal_idx]
            obs_list = [new_obs]
            batch = batch_obs(obs_list, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            
            self._imagenav_rollouts.append(np.hstack((new_obs['head_rgb'],new_obs['imagegoal_rotation'])))
                                            
            with torch.no_grad():
                action_data = self.imagenav_agent.actor_critic.act(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    deterministic=False,
                )

                if action_data.should_inserts is None:
                    self.test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    self.prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    self.imagenav_agent.actor_critic.update_hidden_state(
                        self.test_recurrent_hidden_states, self.prev_actions, action_data
                    )
            
            if is_continuous_action_space(self.env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self.env_spec.action_space.low,
                        self.env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            action = {
                'action': step_data[0],
                'override_turn_angle': self.imagenav_cfg.habitat.simulator.turn_angle
            }
            
            if step_data[0] == HabitatSimActions.stop:
                self._reset_policy_tensors()
                action = {
                    "action": ("low_level_subgoal_switch_action"),
                    "action_args": {},
                }

                self._cur_subgoal_idx += 1
                print("Reset imagenav policy and switching imagenav goal idx to : ", self._cur_subgoal_idx)

            self.action_index += 1

            # Low level policy timeout
            if self.action_index == self._imagenav_max_steps:
                print("Imagenav policy timed out !")
                action = {
                    "action": (HabitatSimActions.stop),
                }
                
        else:
            action = self.get_random_action()       # Return random action as the oracle agent action will override this during data collection phase

        return action, self._imagenav_rollouts
    
    def _init_imagenav_agent(self, imagenav_config_path, eval_data_path):
        '''
        Inspired from standalone PPO agent in home-robot: https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/agent/ovmm_agent/ppo_agent.py
        Mostly reproduces the functionality of habitat_baselines.rl.ppo.single_agent_access_mgr.SingleAgentAccessMgr._create_policy() to init and load the trained imagenav policy
        '''
        imagenav_policy = torch.load(imagenav_config_path, map_location="cpu", weights_only=False)
        imagenav_cfg = imagenav_policy['config']
        self.imagenav_cfg = imagenav_cfg
        self.device = get_device()
        
        # Hardcode to run the policy with only single environment
        with read_write(imagenav_cfg):
            imagenav_cfg.habitat_baselines.num_environments = 1
            imagenav_cfg.habitat_baselines.rl.ppo.num_mini_batch = 1
            imagenav_cfg.habitat_baselines.rl.ddppo.pretrained_encoder = False      # Ensure no pretrained weights are loaded in the background while creating the env_spec object
            imagenav_cfg.habitat_baselines.rl.ddppo.pretrained = False
            imagenav_cfg.habitat.dataset.data_path = eval_data_path.data_path
            imagenav_cfg.habitat.dataset.viewpoints_matrix_path = eval_data_path.viewpoints_matrix_path
            imagenav_cfg.habitat.dataset.transformations_matrix_path = eval_data_path.transformations_matrix_path

        # NOTE: We temporarily create single vectorized env to use the env_spec object for instantiating the imagenav agent. Envs will be closed later
        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            imagenav_cfg.habitat_baselines.vector_env_factory
        )
        envs = env_factory.construct_envs(
            imagenav_cfg,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=True,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )
        self.env_spec = EnvironmentSpec(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            orig_action_space=envs.orig_action_spaces[0],
        )

        # Initialize the imagenav agent using the single agent access manager
        self.obs_transforms = get_active_obs_transforms(imagenav_cfg)
        self.env_spec.observation_space = apply_obs_transforms_obs_space(
            self.env_spec.observation_space, self.obs_transforms
        )
        self.imagenav_agent = baseline_registry.get_agent_access_mgr(
            imagenav_cfg.habitat_baselines.rl.agent.type
        )(
            config=imagenav_cfg,
            env_spec=self.env_spec,
            is_distrib=get_distrib_size()[2] > 1,
            device=self.device,
            resume_state=None,
            num_envs=envs.num_envs,
            percent_done_fn=self.percent_done,
        )
        
        self.imagenav_agent.load_state_dict(imagenav_policy)
        
        # Close the envs that were created for the imagenav policy initialization
        envs.close()
        
        print("ImageNav Policy Created !")

    def percent_done(self):
        return 0.5