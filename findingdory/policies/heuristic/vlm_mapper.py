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
from PIL import Image
import cv2

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from findingdory.utils import get_device
from findingdory.dataset.utils import magnum_to_python_quaternion
from findingdory.dataset.utils import get_agent_yaw, coeff_to_yaw

from findingdory.policies.heuristic.mapping.config_utils import get_config as get_mapper_config
from findingdory.policies.heuristic.mapping.semantic_mapper_agent import SemanticMapperAgent
from findingdory.policies.heuristic.mapping.semantic_map_env import SemanticMappingEnv
from findingdory.policies.heuristic.mapping.interfaces import DiscreteNavigationAction

class VLMMapperAgent(QwenAgent):
    """
    VLM + Explicit Mapping based agent for Habitat environments.
    Uses VLM for high-level image goal selection
    Uses FMM local planner with explicit mapping for low level navigation
    """

    def __init__(self, config) -> None:
        """
        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        super().__init__(config.habitat_baselines.agent.config)
        self.env_config = config
        self.episode_index = 0
        self.task_id = None
        
        self._set_nav_goals = False
        self._pointnav_max_steps = config.habitat_baselines.agent.config.pointnav_max_steps
        self.turn_angle = self.env_config.habitat.simulator.turn_angle
                
    def reset(self):
        super().reset()
        self._agent_in_nav_mode = []
        self._cur_subgoal_idx = None
        self._vlm_frames_processed = False
        self._set_nav_goals = False
        
        if not os.path.exists(self.output_folder):
            # Create the new directory
            os.makedirs(self.output_folder)
            
        self.init_semantic_mapper()
        self.mapper_env.reset()
        self.mapper_agent.reset()
        
        self._global_poses = []
        self.nav_goal_targets = []
        self._rotate_in_place_mode = False
        self._rotate_in_place_executed = False
        self._global_obstacle_map = None

    def init_semantic_mapper(self):
        '''
        Initialise the semantic mapper
        '''
        mapper_config = get_mapper_config(self.env_config)
        self.mapper_env = SemanticMappingEnv(config=mapper_config.habitat.task.semantic_mapper)
        self.mapper_agent = SemanticMapperAgent(config=mapper_config.habitat.task.semantic_mapper)

    def reset_new_task(self, task_id):
        self.action_index = 0
        self.task_id = task_id
        self.output_folder_with_episode_index = os.path.join(
            self.output_folder, f"ep_{self.episode_index:04d}"
        )
        self.output_folder_with_episode_task_index = os.path.join(
            self.output_folder, f"ep_{self.episode_index:04d}_" + self.task_id
        )
        self._cur_subgoal_idx = None
        self._set_nav_goals = False       
        self.nav_goal_targets = []
        self._rotate_in_place_mode = False
        self._rotate_in_place_executed = False
        
        print("Reset low level policy to evaluate next instruction in queue !")
    
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
            
        lang_goal = obs["lang_memory_goal"]
                        
        if obs["can_take_action"]:
            
            # Run VLM inference for subgoal prediction
            if self.action_index % self.vlm_query_freq == 0:
                
                if not self._vlm_frames_processed:
                    self._frames = self.process_frames_for_vlm(save_images=False)
                    
                    # Also subsample the global goal poses returned by the semantic mapper
                    if self.subsample_frames and len(self._global_poses) > self.chunk_size:
                        print(f"Subsampling global goal poses with chunk size {self.chunk_size}")
                        num_frames = np.linspace(0, len(self._global_poses)-1, self.chunk_size, dtype=int)
                        num_frames = num_frames.tolist()  # Convert NumPy array to a standard Python list
                        self._global_poses = [self._global_poses[i] for i in num_frames]

                    self._vlm_frames_processed = True
                
                self.nav_indices, _, _  = self.run_vlm(self._frames, lang_goal)
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

                # Update the semantic map
                sem_map_vis, sem_map_frame, obstacle_map_vis = self._last_sem_map_vis, self._last_sem_map_frame, self._last_obstacle_map_vis
                
                return action, (sem_map_vis, sem_map_frame, obstacle_map_vis)

            assert len(self.nav_indices) > 0, "Trying to activate pointnav policy but could not find any high level goals selected by the VLM. This implies high level goal verification should have failed and the task should have been stopped !"
            assert len(self._global_poses) == len(self._observations), "Global poses and observations should have the same length !"

            # Store the goal pose states for each valid subgoal
            if not self._set_nav_goals:
                # Append the goal image for each valid index
                for index in self._clipped_indices:
                    self.nav_goal_targets.append(self._global_poses[index])
                self._set_nav_goals = True
                
            # Start global pointnav to VLM selected image goal
            if self._cur_subgoal_idx is None:
                print("Beginning global pointnav to VLM selected image goal...")
                self._cur_subgoal_idx = 0
                
            # This implies that the pointnav has tried reaching all the subgoals so we just invoke STOP
            if self._cur_subgoal_idx >= len(self.nav_goal_targets):
                action = {
                    "action": ("pddl_intermediate_stop"),
                    "action_args": {},
                }
                return action, (self._last_sem_map_vis, self._last_sem_map_frame, self._last_obstacle_map_vis)
                
            # Get the pointnav global pose target for current subgoal
            target_pose = self.nav_goal_targets[self._cur_subgoal_idx]
            
            mapper_observation = self.mapper_env._preprocess_obs(obs)
            planner_action, mapper_info, _, _, planner_success = self.mapper_agent.act(mapper_observation, obs["can_take_action"], target_pose, self._global_obstacle_map)
            sem_map_vis, sem_map_frame, obstacle_map_vis = self.mapper_env._process_info(mapper_info)
            assert planner_success is not None, "Planner success flag is not set !"
            
            # If the agent is in rotate-in-place mode, we need to execute the rotate-in-place action
            if self._rotate_in_place_mode:
                rotate_action = self.turn_towards_goal(obs)
                
                if rotate_action != HabitatSimActions.stop:
                    print(f"[Global Pointnav] rotate-in-place action: name - {rotate_action}")
                    return rotate_action, (sem_map_vis, sem_map_frame, obstacle_map_vis)
                else:
                    print(f"[Global Pointnav] Ending rotate-in-place action !")
                    self._rotate_in_place_mode = False
                    self._rotate_in_place_executed = True
                    planner_action = DiscreteNavigationAction.STOP
                
            print(f"[Global Pointnav] Executing low level planner action: name - {planner_action.name}, value - {planner_action.value}")
                        
            if planner_action.value == 0:
                action = {
                    "action": (HabitatSimActions.stop),
                }
            elif planner_action.value == 1:
                action = {
                    "action": (HabitatSimActions.move_forward),
                }
            elif planner_action.value == 2:
                action = {
                    "action": (HabitatSimActions.turn_left),
                }
            elif planner_action.value == 3:
                action = {
                    "action": (HabitatSimActions.turn_right),
                }
            else:
                raise ValueError(f"Invalid action with name: {planner_action.name} and value: {planner_action.value}")
            
            if planner_action.value == HabitatSimActions.stop:
                # Return specific action denoting that the planner failed to reach the goal. But if in rotate-in-place mode, then we dont care about planner output
                if not planner_success and not self._rotate_in_place_mode:
                    self._cur_subgoal_idx += 1
                    print(f"Planner failed to reach the goal and switching to subgoal {self._cur_subgoal_idx} !")
                    self._rotate_in_place_executed = False
                    action = {
                        "action": ("low_level_policy_failure"),
                        "action_args": {},
                    }
                    
                else:
                    # Execute rotate-in-place sequence if not completed, otherwise switch to the next subgoal
                    if not self._rotate_in_place_executed:
                        self._rotate_in_place_mode = True
                        rotate_action = self.turn_towards_goal(obs)
                        print(f"Beginning rotate-in-place sequence with action {rotate_action}...")
                        action = {
                            "action": (rotate_action),
                        }
                        
                        # Switch to next subgoal if rotate-in-place action immediately returns STOP
                        if rotate_action == HabitatSimActions.stop:
                            print(f"Rotate-in-place sequence completed with action {rotate_action} !")
                            self._cur_subgoal_idx += 1
                            print("Switching pointnav subgoal idx to : ", self._cur_subgoal_idx)
                            self._rotate_in_place_executed = False
                            action = {
                                "action": ("low_level_subgoal_switch_action"),
                                "action_args": {},
                            }
                            self._rotate_in_place_executed = False
                            self._rotate_in_place_mode = False
                            
                    else:
                        self._cur_subgoal_idx += 1
                        print("Switching pointnav subgoal idx to : ", self._cur_subgoal_idx)
                        self._rotate_in_place_executed = False
                        action = {
                            "action": ("low_level_subgoal_switch_action"),
                            "action_args": {},
                        }

            self.action_index += 1
            
            # Low level policy timeout
            if self.action_index == self._pointnav_max_steps:
                print("Pointnav policy timed out !")
                action = {
                    "action": (HabitatSimActions.stop),
                }
                
            self._last_obstacle_map_vis = obstacle_map_vis
            self._last_sem_map_vis = sem_map_vis
            self._last_sem_map_frame = sem_map_frame
                    
        else:            
            # Cache the observations and agent states for subsequent high level goal success verification
            # We only add observations when data collection is being done as we dont need to cache the observations/states when the low level policy operates
            self._observations.append(obs)
            self._agent_states.append(agent_state)
            self._agent_in_nav_mode.append(not obs["manipulation_mode"])

            # Perform semantic mapping only in navigation mode as mapping in manipulation mode smears the map
            if not obs["manipulation_mode"]:
                mapper_observation = self.mapper_env._preprocess_obs(obs)
                planner_action, mapper_info, cur_global_pose, self._global_obstacle_map, _ = self.mapper_agent.act(mapper_observation, obs["can_take_action"])
                sem_map_vis, sem_map_frame, obstacle_map_vis = self.mapper_env._process_info(mapper_info)
                self._last_obstacle_map_vis = obstacle_map_vis
                self._last_sem_map_vis = sem_map_vis
                self._last_sem_map_frame = sem_map_frame
                self._global_poses.append(cur_global_pose)
            else:
                sem_map_vis, sem_map_frame, obstacle_map_vis = self._last_sem_map_vis, self._last_sem_map_frame, self._last_obstacle_map_vis
                self._global_poses.append(self._global_poses[-1])

            action = self.get_random_action()       # Return random action as the oracle agent action will override this during data collection phase

        return action, (sem_map_vis, sem_map_frame, obstacle_map_vis)
    
    def turn_towards_goal(self, obs):
        
        # Compute the goal rotation
        goal_rot = self._agent_states[self._clipped_indices[self._cur_subgoal_idx]].rotation
        snapped_goal_orientation = [goal_rot.x, goal_rot.y, goal_rot.z, goal_rot.w]

        current_rotation_yaw = get_agent_yaw(obs['agent_state'])
        goal_rotation_yaw = coeff_to_yaw(snapped_goal_orientation)

        angle_to_goal = goal_rotation_yaw - current_rotation_yaw
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        angle_to_goal = np.degrees(angle_to_goal)

        if angle_to_goal < -self.turn_angle/2:
            return HabitatSimActions.turn_right
        elif angle_to_goal > self.turn_angle/2:
            return HabitatSimActions.turn_left
        else:
            return HabitatSimActions.stop