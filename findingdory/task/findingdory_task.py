#!/usr/bin/env python3
import os.path as osp
from typing import Dict, List
from collections import defaultdict
import copy

import numpy as np
import pandas as pd
import random

import habitat_sim
from habitat.core.registry import registry
from habitat.core.spaces import EmptySpace
from habitat.core.logging import logger
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.utils import add_perf_timing_func
from habitat.tasks.ovmm.sub_tasks.nav_to_obj_task import OVMMDynNavRLEnv
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from findingdory.dataset.findingdory_dataset import FindingDoryEpisode
from findingdory.policies.heuristic.oracle_agent import OracleAgent
from findingdory.task.pddl.pddl import (
    ContPddlEntityProperty,
    DiscPddlEntityProperty,
    MultiDiscPddlEntityProperty,
    FindingDoryPddlEntity,
    MultiContPddlEntityProperty,
    TaskPosData,
)
from findingdory.task.pddl.pddl_logical_expr import FindingDoryLogicalExpr, FindingDoryLogicalQuantifierType
from findingdory.task.utils import iterate_action_space_recursively_with_keys, convert_discrete_actions_to_continuous, load_json_file
from findingdory.dataset.utils import teleport_agent_to_state_quat_assert, convert_to_ordinal
from findingdory.utils import ACTION_MAPPING
from findingdory.task.pddl.pddl import is_robot_at_position, is_robot_in_room
from findingdory.task.pddl.pddl_domain import get_pddl

@registry.register_task(name="FindingDoryTask-v0")
class FindingDoryTask(OVMMDynNavRLEnv):
    def __init__(
        self,
        *args,
        config,
        dataset=None,
        agent_name="robot_0",
        **kwargs
    ):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self.config = config
        self.agent_name = agent_name
        self.load_room_categories()
        self.pddl = get_pddl(
            config,
            agent_name,
            self.obj_categories,
            self.receptacle_categories,
            self.room_categories,
        )
        self.object_attributes = pd.read_csv(config.object_attributes_file_path)

        self._other_object_semantic_ids: Dict[int, int] = {}
        self._other_object_categories: Dict[str, str] = {}
        self._other_obj_category_to_other_obj_category_id = (
            dataset.other_obj_category_to_other_obj_category_id
        )
        self.load_other_object_categories_file(
            config.other_object_categories_file
        )
        self.num_steps_daily = config.num_steps_daily
        self.selected_instruction = config.selected_instruction

        self._goal_expr = None
        self._last_observation = None
        self._data_collection_phase = True
        self._current_oracle_goal = None
        self._current_oracle_goal_idx = 0

        self.task_pos_data = TaskPosData()
        self.region_name = None
        self.num_steps = 0
        self.lang_goal = ""
        self.last_action = None
        self.oracle_agent = None

        self.action_order, self.action_slices = [], []
        ptr = 0
        for action, action_name in iterate_action_space_recursively_with_keys(self.action_space):
            n_action = 1 if isinstance(action, EmptySpace) else action.shape[0]
            self.action_order.append(action_name)
            self.action_slices.append(slice(ptr, ptr + n_action))
            ptr += n_action
        
        self._invalid_nav_pick_goals = []
        self._last_goal_created = False

        # Initialize VLM related success/failure tracking variables
        self._high_level_goal_assigned = False
        self._high_level_goal_success = -1
        self._high_level_goal_success = -1
        self._high_level_dtg_success = -1
        self._high_level_sem_cov_success = -1
        self._vlm_misidentification = False
        self._vlm_response_error = False
        self._vlm_num_targets_error = False
        self._out_of_bounds_pred = False
        self._pddl_assigns_to_ignore = None
        self._low_level_policy_failure = False
        
        # Initialize high-level/low-level policy success/failure tracking variables
        self._ll_pddl_per_assign_metrics = {}
        self._hl_pddl_per_assign_metrics = {}
        self._hl_pddl_per_assign_list = []
        self._pddl_verification_obs = None
        self._num_data_collection_steps = 0
        self._full_success_metrics = None
        
        self.connected_recep_names = defaultdict(list)
        
        self._oracle_task_end_indices = {}

    def load_other_object_categories_file(
        self,
        other_object_categories_file: str,
    ):
        """
        Load Receptacle category mapping file to generate a dictionary of Receptacle.unique_names to their category.

        """
        self._loaded_other_object_categories = False
        if other_object_categories_file is not None:
            assert osp.exists(other_object_categories_file)
            df = pd.read_csv(other_object_categories_file)
            name_key = "id" if "id" in df else "name"
            category_key = (
                "main_category" if "main_category" in df else "clean_category"
            )

            df["category"] = (
                df[category_key]
                .fillna("")
                .apply(lambda x: x.replace(" ", "_").split(".")[0])
            )
            self._other_object_categories = dict(zip(df[name_key], df["category"]))
            # remove objects that are in object categories or receptacle categories
            self._other_object_categories = {
                k: v
                for k, v in self._other_object_categories.items()
                if k not in self._object_categories and k not in self._receptacle_categories
            }
            self._loaded_other_object_categories = True

    def load_room_categories(self):
        room_categories = list(load_json_file(self.config.room_objects_file_path).keys())
        # add region in front of the room category to disambiguate between rooms with other entities of the same name (eg tv (corresponds to tv_room))
        self._room_categories = [f"region_{room}" for room in room_categories]

    @property
    def obj_categories(self):
        return list(self._obj_category_to_obj_category_id.keys())

    @property
    def receptacle_categories(self):
        return list(self._recep_category_to_recep_category_id.keys())

    @property
    def other_object_semantic_ids(self):
        return self._other_object_semantic_ids

    @property
    def loaded_other_object_categories(self):
        return self._loaded_other_object_categories

    @property
    def room_categories(self):
        return self._room_categories

    def _cache_other_objects(self):
        rom = self._sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)

            # confirm object is not a receptacle
            user_attr_keys = obj.user_attributes.get_subconfig_keys()
            if any(key.startswith("receptacle_") for key in user_attr_keys):
                continue

            # confirm object is not a pickupable object
            obj_name = obj_handle[:-6]
            category = self._object_categories.get(obj_name)
            if category in self._obj_category_to_obj_category_id:
                continue

            category = self._other_object_categories.get(obj_name)

            if (
                category is None
                or category
                not in self._other_obj_category_to_other_obj_category_id
            ):
                continue
            category_id = self._other_obj_category_to_other_obj_category_id[
                category
            ]
            self._other_object_semantic_ids[obj.object_id] = category_id + 1

    def get_next_goal(self, episode: FindingDoryEpisode, idx):
        if idx >= self.num_candidate_objects(episode) and not self._last_goal_created:
            last_goal_position, last_goal_orientation = episode.nav_goal_pos, episode.nav_goal_rot
            self._last_goal_created = True
            self._current_oracle_goal_idx = -1
            return {
                "goal_position": last_goal_position,
                "goal_orientation": last_goal_orientation
            }
        elif idx >= self.num_candidate_objects(episode):
            return None

        pick_goal = episode.candidate_objects[idx]

        chosen_idx_pick = None
        for i in range(len(pick_goal.view_points)):
            if self._verify_path(
                self._get_agent_state.base_pos,
                pick_goal.view_points[i].agent_state.position
            ):
                chosen_idx_pick = i
                break

        if chosen_idx_pick is None:
            raise ValueError("No valid path to pick goal")

        goal_receptacle_name = episode.goal_receptacles[idx][0]
        pick_receptacle_name = episode.target_receptacles[idx][0]

        pick_receptacle = None
        for candidate_start_recep in episode.candidate_start_receps:
            if candidate_start_recep.object_name == pick_receptacle_name:
                pick_receptacle = candidate_start_recep
                break

        place_goal = None
        for candidate_goal_recep in episode.candidate_goal_receps:
            if candidate_goal_recep.object_name == goal_receptacle_name:
                place_goal = candidate_goal_recep
                break

        assert place_goal is not None, (
            "Goal receptacle not found in candidate goal receptacles"
        )

        if self._sim.ep_info.place_valid_receps is None:        # This is used with the modify_episodes_for_pick_place.py which is responsible for populating the place_valid_receps field in the episode dataset
            return {
                "pick_goal": pick_goal,
                "pick_goal_idx": chosen_idx_pick,
                "pick_receptacle": pick_receptacle, 
                "place_goal": place_goal,
                "place_goal_idx": None,
            }
        else:
            chosen_idx_place = self._sim.ep_info.place_valid_receps[str(idx)]

        if chosen_idx_place is None:
            raise ValueError("No valid receptacle was stored during offline place viewpoint selection")
        if chosen_idx_place == -1:
            logger.info("Offline place viewpoint locator was run but none of the viewpoints were successful")
            for i in range(len(place_goal.view_points)):
                if self._verify_path(
                    pick_goal.view_points[chosen_idx_pick].agent_state.position,
                    place_goal.view_points[i].agent_state.position
                ):
                    chosen_idx_place = i
                    break

        self.lang_goal = "Pick up the {} and place it on the {}".format(
            pick_goal.object_category, place_goal.object_category
        )

        return {
            "pick_goal": pick_goal,
            "pick_goal_idx": chosen_idx_pick,
            "pick_receptacle": pick_receptacle,
            "place_goal": place_goal,
            "place_goal_idx": chosen_idx_place,
        }

    @property
    def _get_agent_state(self, agent_id=0):
        return self._sim.get_agent_data(agent_id).articulated_agent

    def _verify_path(self, src, tgt):
        # check if start and end points are navigable
        if not self._sim.pathfinder.is_navigable(src):
            return False
        if not self._sim.pathfinder.is_navigable(tgt):
            return False

        # check if both points are on the same island
        if self._sim.pathfinder.get_island(
            src
        ) != self._sim.pathfinder.get_island(
            tgt
        ):
            return False

        path = habitat_sim.ShortestPath()
        path.requested_start = src
        path.requested_end = tgt
        path_found = self._sim.pathfinder.find_path(path)

        return path_found
    
    def get_alternate_nav_pick_goal(self):
        """
        If object is not visible during picking, then we try to navigate to a new viewpoint from which we can execute the heursitic pick skill
        """
        self._invalid_nav_pick_goals.append(
            self._current_oracle_goal['pick_goal_idx']
        )
        
        pick_goal = self._current_oracle_goal['pick_goal']        
        for i in range(len(pick_goal.view_points)):
            if i in self._invalid_nav_pick_goals:
                continue
            if self._verify_path(
                self._get_agent_state.base_pos,
                pick_goal.view_points[i].agent_state.position
            ):
                self._current_oracle_goal['pick_goal_idx'] = i
                break

    def num_candidate_objects(self, episode: FindingDoryEpisode):
        return len(episode.candidate_objects)

    def candidate_goal_receps_instances(self, episode: FindingDoryEpisode):
        return [r.object_name for r in episode.candidate_goal_receps]

    def candidate_start_receps_instances(self, episode: FindingDoryEpisode):
        return [r.object_name for r in episode.candidate_start_receps]

    def candidate_objects_instances(self, episode: FindingDoryEpisode):
        return list(self._sim._candidate_obj_idx_to_rigid_obj_handle.values())
    
    def candidate_objects_noninteracted_instances(self, episode: FindingDoryEpisode):
        return list(self._sim._candidate_obj_noninteracted_idx_to_rigid_obj_handle.values())

    def reset(self, episode: FindingDoryEpisode):
        self._other_object_semantic_ids = {}
        self._cache_other_objects()

        self._current_oracle_goal_idx = 0
        self._data_collection_phase = True
        self._last_goal_created = False
        self.num_steps = 0
        self.last_action = None
        self.lang_goal = ""
        self._current_oracle_goal_idx = 0
        self._data_collection_phase = True
        self._last_goal_created = False

        if self.selected_instruction >= 0:
            self._chosen_instr_idx = self.selected_instruction
        else:
            self._chosen_instr_idx = np.random.randint(len(episode.instructions))
        
        print(f"Episode id: {episode.episode_id}, \tChosen instruction idx: {episode.instructions[self._chosen_instr_idx].task_id} -----> lang: {episode.instructions[self._chosen_instr_idx].lang}")
        # assert episode.instructions[self._chosen_instr_idx].task_id == f"task_{self._chosen_instr_idx+1}", \
        #     f"Found episode with task_{self._chosen_instr_idx+1} missing!"

        super().reset(episode, fetch_observations=False)

        self._subgoals = list(self._get_subgoals(episode))

        self._sim.maybe_update_articulated_agent()

        self.oracle_agent = OracleAgent(self._sim, self, self.config)

        self._last_observation = self._get_observations(episode)
        
        if self.num_steps < self.num_steps_daily:
            self._current_oracle_goal = self.get_next_goal(episode, self._current_oracle_goal_idx)
            
        self.task_pos_data = TaskPosData()
        self.task_pos_data.old_start_pos = (
            self._get_agent_state.base_pos,
            self._get_agent_state.base_rot
        )
        
        self.pddl.bind_to_instance(self._sim, self, episode)
        # Initialize keyframe tracking dict if it doesn't exist
        self._entity_keyframes = defaultdict(list)
        self._step_manip_modes = {}
        
        # Flag variables used to manage interaction start/end time assignment
        self._is_pick_time_assigned = False
        self._is_nav_place_time_assigned = False
        self._is_place_time_assigned = False
        self._entity_interaction_times = defaultdict(lambda: defaultdict(list))
        self._per_obj_interaction_duration = defaultdict(float)
        
        # Reset metrics tracking variables 
        self._num_data_collection_steps = 0
        self._num_actual_targets = None
        self._oracle_agent_timeout = False
                
        # Store the original PDDL predicate thresholds (as we will modify them dynamcially to test for various success metrics)
        self._orig_pddl_predicates = copy.deepcopy(self.pddl.predicates)
        
        self._reset_pddl_verification_entities()
        
        # Add region tracking variables
        self._visited_region_names = set()
        self._start_room = None
        self._per_region_time_spent = defaultdict(float)
        self._region_to_start_obj_idx_map, self._region_to_start_obj_cat_map = self._map_obj_pos_to_regions()
        
        self._oracle_solution = None
        self._oracle_entities = None
        self._oracle_task_end_indices = {}
        
        self.connected_recep_names = defaultdict(list)
        
        return self._last_observation
    
    def _reset_pddl_verification_entities(self):
        '''
        Reset all the PDDL goal verification related variables
        '''
        # Reset task success related metrics
        self._high_level_goal_assigned = False
        self._high_level_goal_success = False
        self._high_level_dtg_success = False
        self._high_level_sem_cov_success = False
        self._vlm_misidentification = False
        self._vlm_response_error = False
        self._vlm_num_targets_error = False
        self._out_of_bounds_pred = False
        self._high_level_nav_indices = None
        self._low_level_policy_failure = False

        # Reset task success/failure mode metrics handled at the PDDL predicate level
        self._pddl_verification_obs = None
        self._pddl_assigns_to_ignore = None
        self._full_success_metrics = None
        self._oracle_solution = None
        self._oracle_entities = None

        # Reset high-level/low-level policy success/failure tracking variables
        self._hl_pddl_per_assign_list = []
        
    def within_episode_reset(self, episode: FindingDoryEpisode, observations):
        self._data_collection_phase = False
        self._current_oracle_goal_idx = -1
        self._num_data_collection_steps = self.num_steps
        self.measurements.reset_measures(
            episode=episode,
            task=self,
            observations=observations,
        )
        self.lang_goal = episode.instructions[self._chosen_instr_idx].lang        


    def step(self, *args, action, episode: FindingDoryEpisode, **kwargs):
        update_task_pos = False

        if self.num_steps < self.num_steps_daily and self._data_collection_phase:
            action, current_task_done, action_idx = self.oracle_agent.act(
                self._last_observation,
                self._current_oracle_goal,
                self._current_oracle_goal_idx,
                self.measurements.measures
            )

            self._update_entity_interaction_timestep(episode, current_task_done=current_task_done)
            # TODO: We should avoid having issues with the agent getting stuck

            # Check if object was not visible during pick sequence
            if self.oracle_agent.current_policy == "retry_nav_pick":
                assert ACTION_MAPPING[action_idx] == "navigation_mode"      # When retrying to navigate to an alternate pick object goal viewpoint, we first switch to nav_mode
                self.get_alternate_nav_pick_goal()
                logger.info(f"Switched to alternate Nav-Pick goal idx: {self._current_oracle_goal['pick_goal_idx']}")   
                self._is_pick_time_assigned = False         # Interaction times will need to be set again as we try to re-navigate to the pick object so that it is visible during picking  
                self.oracle_agent.current_policy = "nav_pick"
                    
            # Get region for current robot position
            robot_pos = self._sim.get_agent_state().position
            regions = self._sim.semantic_scene.get_regions_for_point(robot_pos)
            for region_id in regions:
                region_name = self._sim.semantic_scene.regions[region_id].id
                self._visited_region_names.add(region_name)
                self._per_region_time_spent[region_name] += 1.
            
            if self._start_room is None:
                # TODO: What if there are multiple regions?
                self._start_room = self._sim.semantic_scene.regions[regions[0]].id
            
            if current_task_done:
                if self._current_oracle_goal_idx == -1:
                    # If we have already reached the last goal, we need to reset the oracle agent
                    if not self._last_goal_created:
                        raise ValueError("Last goal was not created but current task is done!")
                    self._current_oracle_goal = None
                    self._invalid_nav_pick_goals = []
                else:
                    self._oracle_task_end_indices.update({self._current_oracle_goal_idx : self.num_steps})
                    self._current_oracle_goal_idx += 1
                    self._current_oracle_goal = self.get_next_goal(episode, self._current_oracle_goal_idx)
                    self._invalid_nav_pick_goals = []
                
            if self._current_oracle_goal is None or (self.num_steps + 1) == self.num_steps_daily:
                # This condition denotes the max num_steps_daily are reached but the oracle agent hasnt been able to perform all the required oracle pick-place interactions
                if self._current_oracle_goal is not None:
                    action = {'action': 0}                      # We just invoke STOP action in this case to end the episode
                    self._high_level_goal_success = False
                    self._high_level_goal_assigned = True
                    self._oracle_agent_timeout = True
                    
                else:
                    # self._validate_and_fill_keyframes()
                    
                    # Register all entities PDDL entities post data collection phase 
                    # This allows us to collect extra information online required for some tasks
                    self._load_start_goal(episode)
                    
                    self._check_for_pddl_assigns_to_ignore()
                                        
                    self.within_episode_reset(episode, self._last_observation)
                    
                    action = self.oracle_agent.get_navigation_mode_action()
                    update_task_pos = True

                    # NOTE: We need to reset the pddl truth cache because pddl values may become True during task collection phase
                    self.pddl.sim_info.reset_pred_truth_cache()

                    assert self._num_actual_targets > 0, "Finished setting up PDDL goal expr but found no entities for task verification !"

            # Force the task to continue even if the force termination conditions are met
            self.should_end = False
        
        else:
            # If current task does not require sequential goal checking, we reset the truth cache as the cache may store truth values from prior PDDL verification routines which we dont want to use
            if not isinstance(episode.instructions[self._chosen_instr_idx].sequential_goals, Dict):
                self.pddl.sim_info.reset_pred_truth_cache()

        # For place_viewpoint modification script: Force the task to continue even if the force termination conditions are met
        if self._sim.ep_info.place_valid_receps is None:     
            self.should_end = False
        
        # Check if we are testing the verification of the high level goals selected by the VLM. Assign the based on the verification result
        if action["action"] == "high_level_policy_action":
            assert self._data_collection_phase is False, "[ERROR] Trying to run high level policy subgoal verification before data collection ended !"
            self._run_high_level_policy_verification(action)
            
            # If high level goals lead to task success, then execute an empty action to shift flow to the agent's low level policy
            if self._high_level_goal_success:
                action = {
                    "action": ("arm_action"),
                    "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
                }
            # If high level goals lead to task failure, stop the task
            else:
                action = {
                    "action": (HabitatSimActions.stop),
                }

        # Convert action_dict for default habitat discrete actions (forward,left,right) into continuous BaseWaypointTeleport actions to work with articulated agents
        # Low level policies such as imagenav can have a different turn angle
        if 'override_turn_angle' in action:
            angle_step = action['override_turn_angle']
        else:
            angle_step = self._sim.habitat_config.turn_angle
        action = convert_discrete_actions_to_continuous(action, self._sim.habitat_config.forward_step_size, angle_step)

        self.num_steps += 1
        self.last_action = action
        self._last_observation = super().step(*args, action=action, episode=episode, **kwargs)
        
        if update_task_pos:
            # Update the task position data after observation is updated
            self.task_pos_data.new_start_pos = (
                self._get_agent_state.base_pos,
                self._get_agent_state.base_rot
            )

        # Track valid keyframes for objects and receptacles
        # if self._data_collection_phase:
        #     self.track_valid_keyframes()
            
        return self._last_observation

    def track_valid_keyframes(self):
        """
        Tracks which objects and receptacles satisfy the is_robot_at_position() predicate at the current step.
        This function should be called within the step() function to maintain a record of valid keyframes.
        """
        
        # Track manipulation mode for each step
        self._step_manip_modes[self.num_steps] = self._in_manip_mode
        
        # Helper function to create temporary PDDL entity
        def create_temp_pddl_entity(entity_name, is_receptacle=False):
            if is_receptacle:
                asset_name = _strip_instance_id(_strip_receptacle(entity_name))
            else:
                asset_name = _strip_instance_id(entity_name)
                
            if asset_name not in self._name_to_cls:
                return None
                    
            cls_name = self._name_to_cls[asset_name]
            entity_type = self.pddl.expr_types[cls_name]
            return FindingDoryPddlEntity(entity_name, entity_type)

        # Get list of candidate objects
        candidate_objects = self.candidate_objects_instances(self._sim.ep_info)

        # Track objects
        for obj_name in self.pddl.sim_info.obj_ids:
            temp_entity = create_temp_pddl_entity(obj_name, is_receptacle=False)
            if temp_entity is None:
                continue
                
            # For candidate objects, only check keyframe after place policy has started for that object
            if obj_name in candidate_objects:
                # Check if we've started tracking this object (i.e., its place policy has begun)
                should_track = (
                    obj_name in self._entity_interaction_times and 
                    'picked_obj_place_start_time' in self._entity_interaction_times[obj_name]
                )
                
                if should_track:
                    is_at_obj = is_robot_at_position(
                        at_entity=temp_entity,
                        sim_info=self.pddl.sim_info,
                        dist_thresh=self._orig_pddl_predicates['robot_at_object']._is_valid_fn.keywords['dist_thresh'],
                        semantic_cov_thresh=self._orig_pddl_predicates['robot_at_object']._is_valid_fn.keywords['semantic_cov_thresh'],
                        dist_measure="geodesic",
                        robot=None,
                        angle_thresh=self._orig_pddl_predicates['robot_at_object']._is_valid_fn.keywords['angle_thresh']
                    )
                    
                    if is_at_obj:
                        self._entity_keyframes[obj_name].append(self.num_steps)
            else:
                # For non-candidate objects, check keyframe at every step
                is_at_obj = is_robot_at_position(
                    at_entity=temp_entity,
                    sim_info=self.pddl.sim_info,
                    dist_thresh=self._orig_pddl_predicates['robot_at_object']._is_valid_fn.keywords['dist_thresh'],
                    semantic_cov_thresh=self._orig_pddl_predicates['robot_at_object']._is_valid_fn.keywords['semantic_cov_thresh'],
                    dist_measure="geodesic",
                    robot=None,
                    angle_thresh=self._orig_pddl_predicates['robot_at_object']._is_valid_fn.keywords['angle_thresh']
                )
                
                if is_at_obj:
                    self._entity_keyframes[obj_name].append(self.num_steps)

        # Track receptacles
        for receptacle_name in self.pddl.sim_info.receptacles:
            temp_entity = create_temp_pddl_entity(receptacle_name, is_receptacle=True)
            if temp_entity is None:
                continue
                    
            # Check if robot is at this receptacle position
            is_at_recep = is_robot_at_position(
                at_entity=temp_entity,
                sim_info=self.pddl.sim_info,
                dist_thresh=self._orig_pddl_predicates['robot_at_receptacle']._is_valid_fn.keywords['dist_thresh'],
                semantic_cov_thresh=self._orig_pddl_predicates['robot_at_receptacle']._is_valid_fn.keywords['semantic_cov_thresh'],
                dist_measure="geodesic",
                robot=None,
                angle_thresh=self._orig_pddl_predicates['robot_at_receptacle']._is_valid_fn.keywords['angle_thresh']
            )
            
            if is_at_recep:
                self._entity_keyframes[receptacle_name].append(self.num_steps)

        # track room visits
        for region in self._sim.semantic_scene.regions:
            entity_type = self.pddl.expr_types["room_region_entity"]
            region_name = f"region_{region.id}"
            room_entity = FindingDoryPddlEntity(region_name, entity_type)

            # Check if robot is in this room
            is_in_room = is_robot_in_room(
                at_entity=room_entity,
                sim_info=self.pddl.sim_info,
                robot=None,
            )

            if is_in_room:
                self._entity_keyframes[region_name].append(self.num_steps)
            
    def _validate_and_fill_keyframes(self):
        """
        Validates that all candidate objects have keyframes tracked between their place_start and end step indices.
        If any keyframes are missing within this range, they are added to maintain continuity.
        Candidate objects can have missing valid keyframes as the candidate object might not be visible during the initial arm extension and midway arm retraction in the placing routine
        """
        candidate_objects = self.candidate_objects_instances(self._sim.ep_info)
        
        for obj_name in candidate_objects:
            # Skip if object wasn't interacted with
            if obj_name not in self._entity_interaction_times:
                continue
                
            # Get the start and end indices for each interaction interval
            start_indices = self._entity_interaction_times[obj_name]['picked_obj_place_start_step_idx']
            end_indices = self._entity_interaction_times[obj_name]['picked_obj_end_step_idx']
            
            # Since we expect single-element lists, get the single start and end index
            start_idx = start_indices[0]
            end_idx = end_indices[0]

            expected_steps = set(range(start_idx, end_idx + 1))
            actual_steps = set(self._entity_keyframes.get(obj_name, []))
            
            # Find missing steps within this interval
            missing_steps = sorted(list(expected_steps - actual_steps))

            # Filter missing steps to only include those where agent was in manipulation mode
            missing_steps = [step for step in missing_steps if self._step_manip_modes.get(step, False)]
            
            if missing_steps:
                logger.info(f"Adding missing keyframes for {obj_name} between steps {start_idx}-{end_idx}")
                
                # Initialize keyframes list if it doesn't exist
                if obj_name not in self._entity_keyframes:
                    self._entity_keyframes[obj_name] = []
                
                # Add missing steps while maintaining sorted order
                current_keyframes = self._entity_keyframes[obj_name]
                for step in missing_steps:
                    insert_idx = 0
                    while insert_idx < len(current_keyframes) and current_keyframes[insert_idx] < step:
                        insert_idx += 1
                    current_keyframes.insert(insert_idx, step)
                    
    ## PDDL related functions
    @property
    def pddl_problem(self):
        return self.pddl

    @add_perf_timing_func()
    def _load_start_goal(self, episode):
        """
        Setup the start and goal PDDL conditions. Will change the simulator
        state to set the start state.
        """
        # Why are we setting up the PDDL entities after this function? 
        # This code was based on Llarp
        self.pddl.bind_to_instance(self._sim, self, episode)

        self._setup_pddl_entities(episode)

        self.pddl.sim_info.reset_pred_truth_cache()

        if 'farthest' in episode.instructions[self._chosen_instr_idx].task_type:
            self._update_goal_expr_for_distance_task(episode)
                        
        if 'room' in episode.instructions[self._chosen_instr_idx].task_type:
            self._update_goal_expr_for_room_task(episode)

        self._sim.internal_step(-1)
        self._goal_expr = self._load_goal_preds(episode)
        if self._goal_expr is not None:
            self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)
            
        # Update the instruction lang for temporal tasks with a query timestep
        if 'timestep' in episode.instructions[self._chosen_instr_idx].task_type:
            self._update_goal_expr_for_temporal_task(episode)
        
        print(f"Episode id: {episode.episode_id}, \tChosen instruction idx: {episode.instructions[self._chosen_instr_idx].task_id} -----> lang: {episode.instructions[self._chosen_instr_idx].lang}")

    @add_perf_timing_func()
    def _load_goal_preds(self, episode):
        """
        Load the goal PDDL expression from the episode and parse it into a PDDL logical expression.
        """
        return self.pddl.parse_only_logical_expr(
            episode.instructions[
                self._chosen_instr_idx
            ].goal_expr, self.pddl.all_entities
        )

    @property
    def goal_expr(self) -> FindingDoryLogicalExpr:
        return self._goal_expr

    @property
    def subgoals(self):
        return self._subgoals

    @property
    def _name_to_cls(self):
        # combine object and receptacle dictionaries
        return {
            **self._object_categories,
            **self._receptacle_categories
        }
        
    def _check_for_pddl_assigns_to_ignore(self):
        if self._pddl_assigns_to_ignore is None:
            assigns_to_ignore = []
            
            # For sequential tasks, we check if some of the assigns need to be ignored. Usually happens for unordered revisitation tasks where we need to revisit the same receptacle more than once
            for i,truth_val in enumerate(self.goal_expr._truth_vals):
                names = _extract_predicate_names(self.goal_expr.sub_exprs[i])
                names = [name for name in names if 'robot' not in name]
                assert len(names) == 1, "Found more than one unique PDDL assign for current sub expr" 
                if truth_val is None:       # Assigns to be ignored have None truth values
                    assigns_to_ignore.append(names[-1])
            self._pddl_assigns_to_ignore = assigns_to_ignore
            logger.info(f"Will ignore following assign while PDDL verification + failure analysis: {self._pddl_assigns_to_ignore}")

    def is_goal_satisfied(self):
        # We populate the PDDL entities post the data collection phase.
        # So we return False till the data collection is running.
        if self._data_collection_phase:
            return False

        assert self._goal_expr is not None and len(self._goal_expr._sub_exprs) > 0, \
            "Goal expression is not set or is empty."
            
        ret = self.pddl.is_expr_true(self._goal_expr)
        
        # Permanently return False in case of out of order evaluation or incorrect location PDDL stop
        if self._goal_expr._permanently_false:
            return False
        
        # If no additional subgoals were evaluated to True, but the pddl intermediate stop was called -> stop 
        # invoked at an incorrect location -> set PDDL truth to False permanently
        if isinstance(self._sim.ep_info.instructions[self._chosen_instr_idx].sequential_goals, Dict):
            # Filter out `None` values from _truth_vals when checking the number of satisfied subgoals
            current_satisfied_subgoals = sum(val for val in self._goal_expr._truth_vals if val is not None)     # truth_vals will have Nones in ordered revisitation tasks for receps that are not to be visited
            
            # Was a new subgoal satisfied?
            new_subgoal_reached = current_satisfied_subgoals != self._goal_expr._num_subgoals_satisfied
            if not new_subgoal_reached and self.pddl.sim_info.does_want_intermediate_stop:
                self._goal_expr._permanently_false = True
            self._goal_expr._num_subgoals_satisfied = current_satisfied_subgoals
        else:
            # For non sequential tasks, if none of the potential goals are satisfied and stop is invoked -> stop invoked at incorrect location -> set PDDL truth to False permanently
            if not ret and self.pddl.sim_info.does_want_intermediate_stop:
                self._goal_expr._permanently_false = True
        
        # Set the pddl intermediate stop to False so that it does not evaluate to True for the subsequent sub goals
        self.pddl.sim_info.does_want_intermediate_stop = False
        
        return ret

    @add_perf_timing_func()
    def _get_subgoals(self, episode) -> List[List[Predicate]]:
        subgoals = getattr(episode.instructions[self._chosen_instr_idx], "subgoals", None)
        if subgoals is None:
            return []

        ret_subgoals = []
        for subgoal in subgoals:
            subgoal_preds = []
            for subgoal_predicate in subgoal:
                pred = self.pddl.parse_predicate(
                    subgoal_predicate, self.pddl.all_entities
                )
                subgoal_preds.append(pred)
            if len(subgoal_preds) != 0:
                ret_subgoals.append(subgoal_preds)
        return ret_subgoals
    
    def _update_entity_interaction_timestep(self, episode, current_task_done=False):
        if self._current_oracle_goal_idx == -1:
            return

        # Extract current time_of_day [hours, minutes]
        cur_time_of_day = self._extract_time_of_day()

        picked_obj_name = self._sim._candidate_obj_idx_to_rigid_obj_handle[self._current_oracle_goal_idx]
        start_recep_name = episode.name_to_receptacle[picked_obj_name]
        goal_recep_name = episode.goal_receptacles[self._current_oracle_goal_idx][0]

        # Update interaction times based on current policy state
        if self.oracle_agent.current_policy == "pick" and not self._is_pick_time_assigned:
            self._entity_interaction_times[picked_obj_name]['picked_obj_start_time'] = [cur_time_of_day]
            self._entity_interaction_times[start_recep_name]['start_receptacle_start_time'].append(cur_time_of_day)
            self._is_pick_time_assigned = True
            logger.info("Set interaction start time for object and start recep")

        elif self.oracle_agent.current_policy == "nav_place" and not self._is_nav_place_time_assigned:
            self._entity_interaction_times[start_recep_name]['start_receptacle_end_time'].append(cur_time_of_day)
            self._is_nav_place_time_assigned = True
            logger.info("Set interaction end time for start recep")

        elif self.oracle_agent.current_policy == "place" and not self._is_place_time_assigned:
            self._entity_interaction_times[goal_recep_name]['goal_receptacle_start_time'].append(cur_time_of_day)
            self._entity_interaction_times[picked_obj_name]['picked_obj_place_start_time'] = [cur_time_of_day]
            self._entity_interaction_times[picked_obj_name]['picked_obj_place_start_step_idx'] = [self.num_steps]
            self._is_place_time_assigned = True
            logger.info("Set interaction start time for goal recep")

        # Set interaction end times for object/goal_recep
        if current_task_done:
            self._entity_interaction_times[picked_obj_name]['picked_obj_end_time'] = [cur_time_of_day]
            self._entity_interaction_times[picked_obj_name]['picked_obj_end_step_idx'] = [self.num_steps]
            self._entity_interaction_times[goal_recep_name]['goal_receptacle_end_time'].append(cur_time_of_day)
            logger.info("Set interaction end time for object and goal_recep")
            
            # Reset flags and interaction times
            self._is_pick_time_assigned = False
            self._is_nav_place_time_assigned = False
            self._is_place_time_assigned = False

    @add_perf_timing_func()
    def _setup_pddl_entities(self, episode):
        """
        Registers all entities (non-interactable/interactable) objects and receptacles and other entities in the scene as PDDL entities.
        
        Since we register all entities after data collection ends, we also register the interaction times and all other relevant proeprties (ex: interaction order).
        
        Args:
            episode: The current FindingDoryEpisode being run
        """
        candidate_goal_receptacles = self.candidate_goal_receps_instances(episode)
        candidate_start_receptacles = self.candidate_start_receps_instances(episode)
        candidate_objects = self.candidate_objects_instances(episode)
        
        start_recep_name_to_cat_map = {}
        for recep in episode.candidate_start_receps:
            start_recep_name_to_cat_map.update({recep.object_name : recep.object_category})

        goal_recep_name_to_order_map = defaultdict(list)
        for idx, recep in enumerate(episode.goal_receptacles):
            goal_recep_name_to_order_map[recep[0]].append(convert_to_ordinal(idx))

        start_recep_name_to_order_map = defaultdict(list)
        for idx, recep in enumerate(episode.target_receptacles):
            start_recep_name_to_order_map[recep[0]].append(convert_to_ordinal(idx))

        candidate_obj_to_order_map = {}
        for id,name in self._sim._candidate_obj_idx_to_rigid_obj_handle.items():
            candidate_obj_to_order_map.update({name : convert_to_ordinal(id)})

        start_recep_to_candidate_obj_cat_map = defaultdict(list)
        for idx,recep in enumerate(episode.target_receptacles):
            start_recep_to_candidate_obj_cat_map[recep[0]].append(episode.candidate_objects[idx].object_category)
        
        # Register the specific objects in this scene as simulator entities.
        for obj_name in self.pddl.sim_info.obj_ids:
            asset_name = _strip_instance_id(obj_name)
            if asset_name not in self._name_to_cls:
                # These belong to hab2
                continue
            
            # For interactable objects, we add relevant interaction properties
            if obj_name in candidate_objects:
                additional_properties = []
                
                # Add object attribute properties
                object_attribute = self.object_attributes[self.object_attributes['object_name'] == asset_name]
                if len(object_attribute) != 1:
                    logger.warning(f"Expected 1 entry for {obj_name} in the object_attributes file, but got {len(object_attribute)} entries")
                    object_attribute_properties = None
                    logger.warning("Object attribute properties set to None !")
                else:
                    object_attribute_properties = []
                    for attribute_name, attribute_value in object_attribute.items():
                        if attribute_name == 'object_name':
                            continue
                        object_attribute_properties.append(DiscPddlEntityProperty(attribute_name, str(attribute_value.iloc[0])))
                additional_properties.extend(object_attribute_properties)
                
                # Add entity property that denotes from which receptacle the object was picked up
                start_recep_name = self._sim.ep_info.name_to_receptacle[obj_name]
                additional_properties.append(DiscPddlEntityProperty("start_receptacle", start_recep_name_to_cat_map[start_recep_name]))
                
                # Add entity property that denotes the interaction order of the object
                additional_properties.append(DiscPddlEntityProperty("interaction_order", candidate_obj_to_order_map[obj_name]))
                
                self._register_pddl_entity(
                    obj_name,
                    interacted=True,
                    is_receptacle=False,
                    interaction_start_time=self._entity_interaction_times[obj_name]['picked_obj_start_time'],
                    interaction_end_time=self._entity_interaction_times[obj_name]['picked_obj_end_time'],
                    additional_properties=additional_properties
                )
            else:
                self._register_pddl_entity(obj_name, interacted=False, is_receptacle=False)
                
        # Check if this is a multi-goal task
        multi_goal_task = self._sim.ep_info.instructions[self._chosen_instr_idx].sequential_goals

        # Build a mapping from first part to actual interacted receptacle names. "First part" names are the 3D object names
        # This mapping will be used to assign times and properties to receptacles that are a part of the actual interacted receptacles and have slightly different mesh names
        candidate_asset_name_to_recep_name = {}
        self.connected_recep_names = defaultdict(list)
        for recep in episode.candidate_start_receps + episode.candidate_goal_receps:
            asset_name_with_instance = _strip_receptacle(recep.object_name)
            candidate_asset_name_to_recep_name[asset_name_with_instance] = recep.object_name
            
        for receptacle_name in self.pddl.sim_info.receptacles:
            asset_name = _strip_instance_id(_strip_receptacle(receptacle_name))
            if asset_name not in self._name_to_cls:
                continue

            asset_name_with_instance = _strip_receptacle(receptacle_name)
            if asset_name_with_instance in candidate_asset_name_to_recep_name:
                # Use the actual interacted receptacle name to fetch times and properties for receptacles with slightly different names
                interacted_name = candidate_asset_name_to_recep_name[asset_name_with_instance]
                interaction_start_time = []
                interaction_end_time = []
                additional_properties = []

                if interacted_name in candidate_start_receptacles:
                    interaction_start_time.extend(self._entity_interaction_times[interacted_name]['start_receptacle_start_time'])
                    interaction_end_time.extend(self._entity_interaction_times[interacted_name]['start_receptacle_end_time'])
                    additional_properties.append(DiscPddlEntityProperty("start_receptacle", True))
                    additional_properties.append(MultiDiscPddlEntityProperty("interaction_order", start_recep_name_to_order_map[interacted_name]))
                    additional_properties.append(MultiDiscPddlEntityProperty("initial_object", start_recep_to_candidate_obj_cat_map[interacted_name]))
                if interacted_name in candidate_goal_receptacles:
                    interaction_start_time.extend(self._entity_interaction_times[interacted_name]['goal_receptacle_start_time'])
                    interaction_end_time.extend(self._entity_interaction_times[interacted_name]['goal_receptacle_end_time'])
                    additional_properties.append(DiscPddlEntityProperty("goal_receptacle", True))
                    additional_properties.append(MultiDiscPddlEntityProperty("interaction_order", goal_recep_name_to_order_map[interacted_name]))

                # This is the case when we have the slightly different interacted receptacle name but we still mark as an interacted entity
                # Corresponds to having the same 3D asset but a different receptacle name. We want to mark all the receptacles of a 3D asset as 'interacted' if any one of them was interacted with.
                if interacted_name != receptacle_name and multi_goal_task:
                    # NOTE: In multi-goal tasks, we only register the actual interacted receptacle as registering multiple interacted entities requires sequential PDDL verification at each of them.
                    # Instead, during sequential goal PDDL verification, we will use the connected_recep_names dict to explicitly check for PDDL success at each of the connected receptacles.
                    self.connected_recep_names[interacted_name].append(receptacle_name)
                    self._register_pddl_entity(receptacle_name, interacted=False, is_receptacle=True)
                else:
                    self._register_pddl_entity(
                        receptacle_name,
                        interacted=True,
                        is_receptacle=True,
                        interaction_start_time=interaction_start_time,
                        interaction_end_time=interaction_end_time,
                        additional_properties=additional_properties
                    )
            else:
                self._register_pddl_entity(receptacle_name, interacted=False, is_receptacle=True)

        for region in self._sim.semantic_scene.regions:
            region_name = f"region_{region.id}"
            self.pddl.register_episode_entity(
                FindingDoryPddlEntity(region_name, self.pddl.expr_types["room_region_entity"])
            )

        # Register the robot.
        self.pddl.register_episode_entity(
            FindingDoryPddlEntity(self.agent_name, self.pddl.expr_types[self.agent_name])
        )
        
        # Add entity property that denotes shortest/longest interaction time. Skip if num_steps=-1 which implies we want to override the data collecor routine
        # TODO: This is a hacky fix to skip video collection when running pick_place viewpoint locator routine during dataset generation
        if self.num_steps_daily != -1:
            self._update_pddl_entity_with_shortest_longest_time()

        # Register the task related important positions.
        for k in self.task_pos_data.keys():
            self.pddl.register_episode_entity(FindingDoryPddlEntity(k, self.pddl.expr_types[k]))

    def _register_pddl_entity(
        self,
        entity_name,
        interacted,
        is_receptacle=False,
        interaction_start_time=None,
        interaction_end_time=None,
        additional_properties=None
    ):
        """Helper function to register a PDDL entity, handling objects and receptacles differently."""
        
        # Handle stripping differently for objects and receptacles
        if is_receptacle:
            asset_name = _strip_instance_id(_strip_receptacle(entity_name))
        else:
            asset_name = _strip_instance_id(entity_name)
        
        cls_name = self._name_to_cls[asset_name]
        entity_type = self.pddl.expr_types[cls_name]

        # Initialize the base entity properties
        if interacted:
            entity_properties = [DiscPddlEntityProperty("interacted", True)]

            if interaction_start_time is None or interaction_end_time is None:
                raise ValueError("Interaction times is not set for an interacted entity")

            entity_properties.append(MultiContPddlEntityProperty("interaction_start_time", interaction_start_time))
            entity_properties.append(MultiContPddlEntityProperty("interaction_end_time", interaction_end_time))
            
            # Log the interaction durations for each interacted object. Skip if num_steps=-1 which implies we want to override the data collecor routine
            # TODO: This is a hacky fix to skip video collection when running pick_place viewpoint locator routine during dataset generation
            if self.num_steps_daily != -1:
                if not is_receptacle:
                    assert len(interaction_end_time) == 1 and len(interaction_start_time) == 1, \
                        f"Object has non-unique interaction start/end times! " \
                        f"Start times: {interaction_start_time}, End times: {interaction_end_time}"
                    interaction_duration = interaction_end_time[0] - interaction_start_time[0]
                    self._per_obj_interaction_duration[entity_name] = interaction_duration
                    entity_properties.append(ContPddlEntityProperty("total_interaction_time", interaction_duration))
        else:
            entity_properties = [DiscPddlEntityProperty("interacted", False)]

        # Add any additional properties
        if additional_properties:
            entity_properties.extend(additional_properties)

        # Register the entity with its properties
        self.pddl.register_episode_entity(
            FindingDoryPddlEntity(entity_name, entity_type, frozenset(entity_properties))
        )
        
    def _find_shortest_longest_interaction_objs(self):
        '''
        Find the objects which took shortest/longest to rearrange based on interaction timesteps
        '''
        min_interaction_time = float('inf')
        max_interaction_time = -float('inf')
        for interacted_obj_name,interaction_time in self._per_obj_interaction_duration.items():
            if interaction_time < min_interaction_time:
                min_interaction_time = interaction_time
                min_interaction_obj = interacted_obj_name
            if interaction_time > max_interaction_time:
                max_interaction_time = interaction_time
                max_interaction_obj = interacted_obj_name
        
        return min_interaction_obj, max_interaction_obj  
    
    def _update_pddl_entity_with_shortest_longest_time(self):
        
        min_interaction_obj = None
        max_interaction_obj = None
        min_interaction_obj, max_interaction_obj = self._find_shortest_longest_interaction_objs()
        assert min_interaction_obj is not None and max_interaction_obj is not None        
        new_props = list(self.pddl.all_entities[min_interaction_obj].properties) + [DiscPddlEntityProperty("shortest_interaction_time", True)]
        self.pddl.register_episode_entity(
            FindingDoryPddlEntity(min_interaction_obj, self.pddl.all_entities[min_interaction_obj].expr_type, frozenset(new_props))
        )
        new_props = list(self.pddl.all_entities[max_interaction_obj].properties) + [DiscPddlEntityProperty("longest_interaction_time", True)]
        self.pddl.register_episode_entity(
            FindingDoryPddlEntity(max_interaction_obj, self.pddl.all_entities[max_interaction_obj].expr_type, frozenset(new_props))
        ) 

    def _update_goal_expr_for_temporal_task(self, episode):
        """
        Updates the current self.goal_expr with the exact temporal target entity specified in 
        episode.instructions.target_temporal_entity. Uses interaction start/end times from data collection
        phase to identify the target entity and update the episode.instruction.goal_expr. Replaces "XX:XX" placeholder
        in the original instruction with actual timestamp.
        """
        # Get target temporal entity name from instruction
        target_temporal_entity_name = self.goal_expr.sub_exprs[0].sub_exprs[0]._arg_values[0].name
        
        if episode.instructions[self._chosen_instr_idx].task_id == "task_67":   # Navigate to an object which took XX seconds to rearrange
            assert target_temporal_entity_name in self._per_obj_interaction_duration
            query_interaction_time = self._per_obj_interaction_duration[target_temporal_entity_name]
            
            if list(self._per_obj_interaction_duration.values()).count(query_interaction_time):
                print("[WARNING]: For task_67 --> Found multiple interacted objects with the exact same rearrangement duration but only!")
            
            # Update instruction language with sampled interaction duration
            query_interaction_time = str(int(query_interaction_time))
            instruction = episode.instructions[self._chosen_instr_idx]
            instruction.lang = instruction.lang.replace("XX", query_interaction_time)

            logger.info("Updated episode instruction with a query interaction time and removed XX !")
        else:
            # Find interaction times for target entity
            interaction_start_time = []
            interaction_end_time = []
            entity = self.pddl.all_entities[target_temporal_entity_name]

            for prop in entity.properties:
                if prop.name == "interaction_start_time":
                    interaction_start_time.extend(prop.value)
                elif prop.name == "interaction_end_time":
                    interaction_end_time.extend(prop.value)

            if len(interaction_start_time) == 0 or len(interaction_end_time) == 0:
                raise ValueError("Could not find target temporal entity interaction times")

            # First, randomly sample an interval
            interval_idx = random.randint(0, len(interaction_start_time) - 1)
            # Randomly sample a timestep based on the temporal target interaction start and end times

            random_timestep = random.randint(
                int(interaction_start_time[interval_idx]),
                int(interaction_end_time[interval_idx])
            )
            query_timestep = f"{random_timestep // 60}:{random_timestep % 60:02d}"

            # Update instruction language with sampled timestamp
            instruction = episode.instructions[self._chosen_instr_idx]
            instruction.lang = instruction.lang.replace("XX:XX", query_timestep)

            logger.info("Updated episode instruction with a query timestep and removed XX:XX !")

    def _extract_time_of_day(self):
        cur_time_of_day = [int(x) for x in self._last_observation['time_of_day'].split(':')]
        cur_time_min = float(cur_time_of_day[0]) * 60. + float(cur_time_of_day[1])
        return cur_time_min
    
    def _update_entity_with_farthest_property(self, query_entities, property_key, entity_type):
        
        max_dist = -float('inf')
        farthest_entity = None

        for i,entity in enumerate(query_entities):
            
            if entity_type == "receptacle":
                pddl_entity = self.pddl.all_entities[entity.object_name]
            elif entity_type == "object":
                pddl_entity = self.pddl.all_entities[self._sim._candidate_obj_idx_to_rigid_obj_handle[i]]
            else:
                raise ValueError("Unknown entity type !")
            
            viewpoints = self.pddl.sim_info.get_entity_viewpoints(pddl_entity)
            
            if viewpoints is None:  # This is the case when the query entity is an object
                targ_pos = self.pddl.sim_info.get_entity_pos(pddl_entity)
                obj_pos = self.pddl.sim_info.sim.safe_snap_point(targ_pos)
            else:
                obj_pos = viewpoints
                
            robot_obj = self.pddl.sim_info.sim.get_agent_data(0).articulated_agent
            cur_dist = self.pddl.sim_info.sim.geodesic_distance(robot_obj.base_pos, obj_pos)

            if cur_dist > max_dist:
                max_dist = cur_dist
                farthest_entity = pddl_entity.name
        
        new_props = list(self.pddl.all_entities[farthest_entity].properties) + [DiscPddlEntityProperty(property_key, True)]
        
        self.pddl.register_episode_entity(
            FindingDoryPddlEntity(farthest_entity, self.pddl.all_entities[farthest_entity].expr_type, frozenset(new_props))
        )
    
    def _update_goal_expr_for_distance_task(self, episode):
        '''
        This function updates the PDDL properties of the various farthest entities (interacted obj/receps) from the simulator state
        Then we update the self._goal_expr object so that the solution entities are matched for the PDDL verification
        '''
        
        # Find farthest goal_rec entity
        query_entities = self._sim.ep_info.candidate_goal_receps
        property_key = "farthest_goal_rec"
        self._update_entity_with_farthest_property(query_entities, property_key, entity_type='receptacle')
        
        # Find farthest start_rec entity
        query_entities = self._sim.ep_info.candidate_start_receps
        property_key = "farthest_start_rec"
        self._update_entity_with_farthest_property(query_entities, property_key, entity_type='receptacle')
        
        # Find farthest interacted_rec entity
        query_entities = self._sim.ep_info.candidate_start_receps + self._sim.ep_info.candidate_goal_receps
        property_key = "farthest_rec_interacted"
        self._update_entity_with_farthest_property(query_entities, property_key, entity_type='receptacle')

        # Find farthest noninteracted_rec entity
        query_entities = self._get_noninteracted_receps()
        property_key = "farthest_rec_non_interacted"
        self._update_entity_with_farthest_property(query_entities, property_key, entity_type='receptacle')
        
        # Find farthest interacted_obj entity
        query_entities = self._sim.ep_info.candidate_objects
        property_key = "farthest_obj_interacted"
        self._update_entity_with_farthest_property(query_entities, property_key, entity_type='object')
        
        logger.info("Updated episode with farthest entity properties !")
        
    def _update_goal_expr_for_room_task(self, episode):
        """
        Create PDDL entities for each room region in the scene and add relevant entity properties.
        Update the goal expr for the PDDL verification for room-based tasks
        """
        
        # Get all regions from semantic scene
        regions = self._sim.semantic_scene.regions
        
        # Create mapping between regions and the candidate object final locations
        region_to_final_obj_idx_map, region_to_final_obj_cat_map = self._map_obj_pos_to_regions()
        
        # Locate the final room where the agent ended data collection
        final_robot_pos = self._sim.get_agent_state().position
        final_region = self._sim.semantic_scene.get_regions_for_point(final_robot_pos)
        self._final_room = self._sim.semantic_scene.regions[final_region[0]].id
        
        # Locate region where agent spent maximum time
        max_time_region = max(self._per_region_time_spent, key=self._per_region_time_spent.get, default=None)
        assert max_time_region is not None, "Could not locate the region where agent spent the max time !"
                
        # Create a PDDL entity for each region
        for region in regions:
            region_name = f"region_{region.id}"
            
            # Check if this region was visited during data collection
            was_visited = region.id in self._visited_region_names
            
            # Create region entity with visited property
            entity_properties = [
                DiscPddlEntityProperty("visited", was_visited)
            ]
            
            # Mark the room where the robot started the data collection
            if region.id == self._start_room:
                entity_properties.append(DiscPddlEntityProperty("initial_room", True))
            else:
                entity_properties.append(DiscPddlEntityProperty("initial_room", False))
            
            # Mark the room where the robot ended the data collection
            if region.id == self._final_room:
                entity_properties.append(DiscPddlEntityProperty("final_room", True))
            else:
                entity_properties.append(DiscPddlEntityProperty("final_room", False))
            
            if region.id in self._region_to_start_obj_idx_map:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_idx_initial", self._region_to_start_obj_idx_map[region.id]))
            else:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_idx_initial", []))

            if region.id in region_to_final_obj_idx_map:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_idx_final", region_to_final_obj_idx_map[region.id]))
            else:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_idx_final", []))

            if region.id in self._region_to_start_obj_cat_map:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_cat_initial", self._region_to_start_obj_cat_map[region.id]))
            else:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_cat_initial", []))

            if region.id in region_to_final_obj_cat_map:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_cat_final", region_to_final_obj_cat_map[region.id]))
            else:
                entity_properties.append(MultiDiscPddlEntityProperty("obj_cat_final", []))
                
            # Mark the room where the robot spent most time
            if region.id == max_time_region:
                entity_properties.append(DiscPddlEntityProperty("longest_time_spent", True))
            else:
                entity_properties.append(DiscPddlEntityProperty("longest_time_spent", False))

            # Register the region entity
            self.pddl.register_episode_entity(
                FindingDoryPddlEntity(
                    region_name,
                    self.pddl.expr_types["room_region_entity"],
                    frozenset(entity_properties)
                )
            )
            
        logger.info("Updated episode with room entity properties !")
        
    def _get_noninteracted_receps(self):
        '''
        Retrieve non-interacted receptacles, ensuring they do not overlap with 
        interacted start or goal receptacles.
        '''
        cur_ep = self._sim.ep_info

        # Combine non-interacted start and goal receptacles
        noninteracted_receps = (
            cur_ep.candidate_start_receps_noninteracted +
            cur_ep.candidate_goal_receps_noninteracted
        )

        # Collect names of interacted start and goal receptacles
        interacted_recep_names = {
            recep.object_name for recep in cur_ep.candidate_start_receps
        }.union({
            recep.object_name for recep in cur_ep.candidate_goal_receps
        })
        interacted_asset_name = set(
            name.split("|")[0] for name in interacted_recep_names
        )

        # Filter out non-interacted receptacles that appear in interacted lists
        filtered_receps = [
            recep for recep in noninteracted_receps if recep.object_name not in interacted_recep_names
        ]
        filtered_receps = [
            recep for recep in filtered_receps
            if recep.object_name.split("|")[0] not in interacted_asset_name
        ]

        return filtered_receps  
        
    def _map_obj_pos_to_regions(self):
        '''
        Helper function that creates a mapping between the regions and the candidate objects (idxs and obj_categories)
        '''
        region_to_candidate_obj_idx_map = defaultdict(list)
        region_to_candidate_obj_cat_map = defaultdict(list)
        rom = self._sim.get_rigid_object_manager()
        
        for idx,obj_name in self._sim._candidate_obj_idx_to_rigid_obj_handle.items():
            obj = rom.get_object_by_handle(obj_name)
            region = self._sim.semantic_scene.get_regions_for_point(obj.com)
            region_name = self._sim.semantic_scene.regions[region[0]].id
            region_to_candidate_obj_idx_map[region_name].append(convert_to_ordinal(idx))
            region_to_candidate_obj_cat_map[region_name].append(self._sim.ep_info.candidate_objects[idx].object_category)
            
        return region_to_candidate_obj_idx_map, region_to_candidate_obj_cat_map

    def _run_high_level_policy_verification(self, action):
        
        nav_indices = action["action_args"]["nav_indices"]
        self._high_level_nav_indices = nav_indices
        goal_states = action["action_args"]["nav_goal_states"]
        nav_mode_flags = action["action_args"]["nav_mode_flag"]
        
        # Track an invalid response from VLM
        if len(nav_indices) == 1 and nav_indices[0] == -1:
            self._high_level_goal_success = False
            self._high_level_goal_assigned = True
            self._vlm_response_error = True
            self._full_success_metrics = -1
            print("Episode failed as VLM generated an invalid response !")
            
        # Check if the VLM predicted the correct number of targets
        elif self._num_actual_targets != len(nav_indices):
            self._high_level_goal_success = False
            self._high_level_goal_assigned = True
            self._vlm_num_targets_error = True
            self._full_success_metrics = -1
            print(f"Episode failed as VLM predicted incorrect number of goals -- Actual num_goals: {self._num_actual_targets} but predicted {len(nav_indices)} subgoals")
        
        # Iterate over each index in nav_indices to check for out-of-bound index
        else:
            for index in nav_indices:
                if index > self._num_data_collection_steps or index < 0:
                    # If any index is out-of-bounds, mark the task as unsuccessful and handle errors
                    self._high_level_goal_success = False
                    self._high_level_goal_assigned = True
                    self._out_of_bounds_pred = True
                    self._full_success_metrics = -1
                    print("Episode failed as VLM predicted out-of-bounds frame index!")

        if not self._high_level_goal_assigned:
            self.assign_and_validate_high_level_goal(goal_states, nav_mode_flags)
            print(f"--------------------------> Did VLM locate the image frame successfully ?: {self._high_level_goal_success}")
            print(f"Full Success Verification Metrics: {self._full_success_metrics}")

    def assign_and_validate_high_level_goal(self, high_level_goal_states, nav_mode_flags):
        
        dtg_sem_cov_success = []
        dtg_success = []
        sem_cov_success =[]
        
        # Run VLM PDDL verification based on all success criteria
        dtg_sem_cov_success = self._run_pddl_vlm_verification(high_level_goal_states, nav_mode_flags, dtg_sem_cov_success, full_success=True)
        
        # Run VLM PDDL verification only based on dist_to_goal
        _set_new_predicate_args(self._goal_expr, new_param_args=[{'semantic_cov_thresh': 0.}])
        dtg_success = self._run_pddl_vlm_verification(high_level_goal_states, nav_mode_flags, dtg_success)
        
        # Run VLM PDDL verification only based on semantic_coverage of the target object/receptacle
        _set_predicate_args_to_default(self._goal_expr, self._orig_pddl_predicates)
        _set_new_predicate_args(self._goal_expr, new_param_args=[{'dist_thresh': 10000.}])
        sem_cov_success = self._run_pddl_vlm_verification(high_level_goal_states, nav_mode_flags, sem_cov_success)
        
        # Set PDDL predicates back to default arguments and reset the PDDL goal expr for subsequent low level policy success verification
        _set_predicate_args_to_default(self._goal_expr, self._orig_pddl_predicates)
        self._goal_expr = self._load_goal_preds(self._sim.ep_info)
        if self._goal_expr is not None:
            self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)

        # Assign the success values from the result of the PDDL verification procedure
        self._high_level_goal_success = dtg_sem_cov_success[-1]
        self._high_level_dtg_success = dtg_success[-1]
        self._high_level_sem_cov_success = sem_cov_success[-1]
        
        self._high_level_goal_assigned = True
        self._pddl_verification_obs_obtained = False
        
    def _run_pddl_vlm_verification(self, high_level_goal_states, nav_mode_flags, success_tracker_list, full_success=False):
        
        current_state = self._sim.get_agent_state()
        agent = self._sim.get_agent_data(0).articulated_agent
        
        self.pddl.sim_info.does_want_intermediate_stop = True
        self.pddl.sim_info.reset_pred_truth_cache()

        # Cycle through all the high level goal states predicted by the VLM
        for goal_state, nav_mode_flag in zip(high_level_goal_states, nav_mode_flags):
            
            # Teleport the agent to the goal state selected by the high level policy (VLM)
            high_level_goal_rot = goal_state.rotation
            high_level_goal_quat = [
                high_level_goal_rot.x,
                high_level_goal_rot.y,
                high_level_goal_rot.z,
                high_level_goal_rot.w
            ]
            teleport_agent_to_state_quat_assert(
                self._sim,
                high_level_goal_quat,
                goal_state.position,
            )
            agent.update()
            
            self._pddl_verification_obs = self._get_observations_for_pddl_verification(nav_mode_flag)

            # Check if the selected high level goal leads to a PDDL success
            goal_success = self.is_goal_satisfied()
            success_tracker_list.append(goal_success)
            self._hl_pddl_per_assign_list.append(self._hl_pddl_per_assign_metrics)
            self._pddl_verification_obs = None
            
            # If evaluating multi-goals, we again set the intermediate stop to True to make sure each subgoal can evaluate to True
            self.pddl.sim_info.does_want_intermediate_stop = True
                        
        # Bring the agent to the original state after the high level goal has been validated
        cur_state_quat = [
            current_state.rotation.x,
            current_state.rotation.y,
            current_state.rotation.z,
            current_state.rotation.w
        ]
        teleport_agent_to_state_quat_assert(
            self._sim,
            cur_state_quat,
            current_state.position,
        )
        agent.update()
        
        # If we are testing for full succes (DTG + SemCov + AngleThresh), then save the valid PDDL entities metrics for posthoc analysis to check which metric failed/succeeded
        if full_success:
            self._full_success_metrics = copy.deepcopy(self._hl_pddl_per_assign_metrics)
        
        # Reset PDDL related structures to prevent issues with subsequent low level policy success validation
        self._goal_expr._permanently_false = False       # Set to False because it may have been set to True during the high level goal validation
        self.pddl.sim_info.reset_pred_truth_cache()
        self.pddl.sim_info.does_want_intermediate_stop = False
        
        # Also reset the goal_expr to make sure PDDL related things are reset
        self._goal_expr = self._load_goal_preds(self._sim.ep_info)
        if self._goal_expr is not None:
            self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)
                
        return success_tracker_list
    
    def _get_observations_for_pddl_verification(self, nav_mode_flag):
        '''
        Helper function to obtain the observations (for PDDL verification) from the goal state that we just teleported the agent to
        Flow is as follows: switch agent to manipulation mode if oracle agent collected image in manip_mode -> sim.step() and get pddl_obs -> switch agent back to navigation_mode
        '''
        
        # Since the image in the oracle agent video was originally collected in manip mode, we switch the agent to manip and obtain the current observations
        if not nav_mode_flag:
            action_dict = {
                "action": ("manipulation_mode"),
                "action_args": {"manipulation_mode": [1.0], "is_last_action": True},
            }
            new_args = ()
            new_kwargs = {}
            pddl_obs = super().step(*new_args, action=action_dict, episode=self._sim.ep_info, **new_kwargs)
            self._cur_episode_step -= 1
        
            # Switch agent back to navigation mode as PDDL predicates assume agent in navigation mode
            action_dict = {
                "action": ("navigation_mode"),
                "action_args": {"navigation_mode": [1.0], "is_last_action": True},
            }
            new_args = ()
            new_kwargs = {}
            _ = super().step(*new_args, action=action_dict, episode=self._sim.ep_info, **new_kwargs)
            self._cur_episode_step -= 1
        
        # Since agent is already in nav mode, we just create an empty action and step the environment to obtain the current observations
        else:
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
            }
            new_args = ()
            new_kwargs = {}
            pddl_obs = super().step(*new_args, action=action_dict, episode=self._sim.ep_info, **new_kwargs)
            self._cur_episode_step -= 1
            
        return pddl_obs

        
def _strip_instance_id(instance_id: str) -> str:
    # Strip off the unique instance ID of the object and only return the asset name.
    return "_".join(instance_id.split("_")[:-1])


def _strip_receptacle(obj_name: str) -> str:
    return obj_name.split("|")[0]


def _extract_predicate_names(expr):
    predicate_names = []

    # Check if the current expression is a logical_expr with sub-expressions
    if hasattr(expr, 'sub_exprs'):
        for sub_expr in expr.sub_exprs:
            # Recursively parse each sub-expression
            predicate_names.extend(_extract_predicate_names(sub_expr))

    # Check if the current expression is a predicate
    elif hasattr(expr, '_arg_values'):      # Only predicate expressions will have _arg_values
        # Extract the 'name' attribute from all 'arg_values'
        for arg in expr._arg_values:
            if hasattr(arg, 'name'):
                predicate_names.append(arg.name)

    return predicate_names

def _set_new_predicate_args(expr, new_param_args=[{'dist_thresh': 0.}, {'semantic_cov_thresh': 0.}]):
    '''Recursive function which parses the PDDL goal_expr and sets the _is_valid_fn to new arguments for calculating non standard success metrics'''
    predicate_info = []

    # Check if the current expression is a logical_expr with sub-expressions
    if hasattr(expr, 'sub_exprs'):
        for sub_expr in expr.sub_exprs:
            # Recursively parse each sub-expression
            predicate_info.extend(_set_new_predicate_args(sub_expr, new_param_args=new_param_args))

    # Check if the current expression is a predicate
    elif hasattr(expr, '_arg_values'):  # Only predicate expressions will have _arg_values
        if hasattr(expr, '_is_valid_fn'):
            predicate_name = expr.name
            
            if predicate_name == 'robot_at_receptacle' or predicate_name == 'robot_at_object':
                # Access the 'dist_thresh' key in _is_valid_fn.keyword if it exists
                if hasattr(expr._is_valid_fn, 'keywords'):
                    for param_dict in new_param_args:
                        param_name = list(param_dict.keys())[0]
                        param_value = list(param_dict.values())[0]
                        if param_name in expr._is_valid_fn.keywords:
                            expr._is_valid_fn.keywords[param_name] = param_value
                            # Collect information about the predicate and dist_thresh

    return predicate_info

def _set_predicate_args_to_default(expr, default_args):
    '''Recursive function which parses the PDDL goal_expr and sets the _is_valid_fn to default arguments.
    This uses the default arguments from the config/task/fp_domain_pddl.yaml file.'''
    predicate_info = []

    # Check if the current expression is a logical_expr with sub-expressions
    if hasattr(expr, 'sub_exprs'):
        for sub_expr in expr.sub_exprs:
            # Recursively parse each sub-expression
            predicate_info.extend(_set_predicate_args_to_default(sub_expr, default_args))

    # Check if the current expression is a predicate
    elif hasattr(expr, '_arg_values'):  # Only predicate expressions will have _arg_values
        if hasattr(expr, '_is_valid_fn'):
            predicate_name = expr.name
            
            if predicate_name == 'robot_at_receptacle':
                # Access the 'dist_thresh' key in _is_valid_fn.keyword if it exists
                if hasattr(expr._is_valid_fn, 'keywords'):
                    expr._is_valid_fn.keywords['dist_thresh'] = default_args[predicate_name]._is_valid_fn.keywords['dist_thresh']
                    expr._is_valid_fn.keywords['semantic_cov_thresh'] = default_args[predicate_name]._is_valid_fn.keywords['semantic_cov_thresh']

            if predicate_name =='robot_at_object':
                # Access the 'dist_thresh' key in _is_valid_fn.keyword if it exists
                if hasattr(expr._is_valid_fn, 'keywords'):
                    expr._is_valid_fn.keywords['dist_thresh'] = default_args[predicate_name]._is_valid_fn.keywords['dist_thresh']
                    expr._is_valid_fn.keywords['semantic_cov_thresh'] = default_args[predicate_name]._is_valid_fn.keywords['semantic_cov_thresh']
    return predicate_info
