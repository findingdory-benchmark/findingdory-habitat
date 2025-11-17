#!/usr/bin/env python3
from typing import Dict
import copy

import numpy as np

from habitat.core.registry import registry
from habitat.core.logging import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from findingdory.dataset.findingdory_dataset import FindingDoryEpisode
from findingdory.task.utils import convert_discrete_actions_to_continuous
from findingdory.dataset.utils import teleport_agent_to_state_quat_assert
from findingdory.utils import ACTION_MAPPING
from findingdory.task import FindingDoryTask


@registry.register_task(name="FindingDoryMultiTask-v0")
class FindingDoryMultiTask(FindingDoryTask):
    """
    FindingDoryMultiTask extends FindingDoryTask to support the evaluation of multiple task instructions without re-running the oracle agent video data collection.

    Unlike the base class (FindingDoryTask), which only evaluates a **single** task instruction and terminates,
    this class allows the robot to sequentially evaluate multiple task instructions based on the video generated during data_collection phase.
    """

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
                
    def reset(self, episode: FindingDoryEpisode):
        super().reset(episode)
        
        self._instructions_to_evaluate = list(self.config.instructions_to_evaluate)
        
        # Evaluate all tasks sequentially if user specifies [-1] as the instructions to evaluate
        if len(self._instructions_to_evaluate) == 1 and self._instructions_to_evaluate[0] == -1:
            self._instructions_to_evaluate = [x for x in range(len(episode.instructions))]
        
        next_instr_id = self._find_next_instruction_idx()
        self._chosen_instr_idx = next_instr_id
        self.metrics_per_task = {}
        
        return self._last_observation
    
    def _find_next_instruction_idx(self):
        """
        Cycles through user-specified instruction task_ids in `self._instructions_to_evaluate`
        until a valid instruction is found in `episode.instructions`. Returns the corresponding
        integer index of that instruction from `episode.instructions`.
        
        If no valid instruction is found, returns -1 as a flag to end evaluations.
        """
        while self._instructions_to_evaluate:
            query_task_id = self._instructions_to_evaluate.pop(0)

            if not isinstance(query_task_id, str):
                if isinstance(query_task_id, int):
                    query_task_id = "task_" + str(query_task_id)
                else:
                    print("Invalid task ID type. Expected a string or int.")
                    return -1  # Return flag instead of crashing

            # Check if query_task_id exists in episode instructions
            for idx, instruction in enumerate(self._sim.ep_info.instructions):
                if instruction.task_id == query_task_id:
                    print(f"Found instruction for specified {query_task_id} at list index {idx}")
                    return idx  # Return the found index immediately
            
            print(f"User-specified task_id '{query_task_id}' not found in current episode. Checking next in queue...")

        print("No more user-specified task instructions exist in the current episode. Ending evaluations !")
        return -1  # Return a flag value instead of an error
        
    def step(self, *args, action, episode: FindingDoryEpisode, **kwargs):
        update_task_pos = False
        self._switch_to_new_task = False

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
                print(f"Switched to alternate Nav-Pick goal idx: {self._current_oracle_goal['pick_goal_idx']}")   
                self._is_pick_time_assigned = False         # Interaction times will need to be set again as we try to re-navigate to the pick object so that it is visible during picking  
                self.oracle_agent.current_policy = "nav_pick"

            # Get region for current robot position
            robot_pos = self._sim.get_agent_state().position
            regions = self._sim.semantic_scene.get_regions_for_point(robot_pos)
            for region_id in regions:
                self.region_name = self._sim.semantic_scene.regions[region_id].id
                self._visited_region_names.add(self.region_name)
                self._per_region_time_spent[self.region_name] += 1.
            
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
                    # self._validate_and_fill_keyframes()     # TODO: This should be uncommented when generating oracle keyframe solutions
                    
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
                # Set to False to ensure control transfers to the low level policy
                self.should_end = False

            # If high level goals lead to task failure, stop the task
            else:
                action = {
                    "action": (HabitatSimActions.stop),
                }
        
        # In case of a multi-goal task, we may need to stop the task and switch to next instruction (by executing STOP) if the PDDL goal expr has become permanently false due to an earlier subgoal failing
        if isinstance(episode.instructions[self._chosen_instr_idx].sequential_goals, Dict):
            if not self._data_collection_phase and self._goal_expr._permanently_false:
                print("PDDL goal expr has become permanently false so stopping task!")
                action = {
                    "action": (HabitatSimActions.stop),
                }
        
        # If we encounter a stop action, it implies either the high level goals lead to task failure or the low level policy has invoked a STOP action
        # In either case, we move to the next task instruction (if there is one remaining) to be evaluated and reset the PDDL-related objects for the new evaluation
        if action["action"] == HabitatSimActions.stop or action["action"] == "pddl_intermediate_stop":                
            next_instr_id = self._find_next_instruction_idx()
            if next_instr_id != -1:
                self._switch_evaluation_to_new_task(episode, action, self._last_observation, next_instr_id)
                action = {
                    "action": ("arm_action"),
                    "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
                }
                self._switch_to_new_task = True
            else:
                print("All instructions exhausted -- ending task !")
                
        # Check if the low level policy needs to switch to a new subgoal (for multi-goal tasks) and change the action to a pddl_intermediate_stop for subgoal verification
        # Also, if the low level policy is unable to generate any valid action, we siwtch to next instruciton in queue
        if action["action"] == "low_level_subgoal_switch_action" or action["action"] == "low_level_policy_failure":
            if action["action"] == "low_level_policy_failure":
                print("Low level policy failed to reach the goal !")
                self._low_level_policy_failure = True            
            action = {
                "action": ("pddl_intermediate_stop"),
                "action_args": {},
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
        self._last_observation = super(FindingDoryTask, self).step(*args, action=action, episode=episode, **kwargs)
        
        if update_task_pos:
            # Update the task position data after observation is updated
            self.task_pos_data.new_start_pos = (
                self._get_agent_state.base_pos,
                self._get_agent_state.base_rot
            )
            self._oracle_agent_final_pose = self._sim.get_agent_state()

        # Track valid keyframes for objects and receptacles      # TODO: This should be uncommented when generating oracle keyframe solutions and will slow down significantly due to large number of PDDL verifications at each sim step
        # if self._data_collection_phase: 
        #     self.track_valid_keyframes()
            
        return self._last_observation
    
    def _switch_evaluation_to_new_task(self, episode: FindingDoryEpisode, action, observations, next_instr_id):
        """
        This function does the following:
        - Update and print out the current metrics
        - Teleports the agent back to the pose where the data collection terminated
        - Updates the PDDL goal expression with the new task instruction
        - Resets measurements for the new task
        """
        
        # Calculate and print out the measurements at the current task termination
        if action["action"] == "pddl_intermediate_stop":
            self.pddl.sim_info.does_want_intermediate_stop = True       # Set this to True, otherwise low level policy predicate success will not evaluate to True
        stop_action = {
            "action": (HabitatSimActions.stop),
        }
        self.measurements.update_measures(
            episode=episode,
            action=stop_action,
            task=self,
            observations=observations,
        )
        metrics = self.measurements.get_metrics()
        metrics_to_print = {}
        for m, v in metrics.items():
            if m == "top_down_map" or m == "fog_of_war_mask":
                continue
            metrics_to_print[m] = v
        self.metrics_per_task.update({episode.instructions[self._chosen_instr_idx].task_id : metrics_to_print})
        
        cur_task_id = episode.instructions[self._chosen_instr_idx].task_id
        print(f"Metrics for task_ID {cur_task_id}: {metrics_to_print}")
        
        # Teleport the agent to the goal state selected by the high level policy (VLM)
        agent = self._sim.get_agent_data(0).articulated_agent
        oracle_agent_final_quat = [
            self._oracle_agent_final_pose.rotation.x,
            self._oracle_agent_final_pose.rotation.y,
            self._oracle_agent_final_pose.rotation.z,
            self._oracle_agent_final_pose.rotation.w
        ]
        teleport_agent_to_state_quat_assert(
            self._sim,
            oracle_agent_final_quat,
            self._oracle_agent_final_pose.position,
        )
        agent.update()
        
        # Update the PDDL goal expr with the new task
        self._chosen_instr_idx = next_instr_id
        
        self.pddl.bind_to_instance(self._sim, self, episode)
        self._setup_pddl_entities(episode)
        self._oracle_solution = None
        self._oracle_entities = None

        if 'farthest' in episode.instructions[self._chosen_instr_idx].task_type:
            self._update_goal_expr_for_distance_task(episode)
                        
        if 'room' in episode.instructions[self._chosen_instr_idx].task_type:
            self._update_goal_expr_for_room_task(episode)

        self._goal_expr = self._load_goal_preds(episode)
        if self._goal_expr is not None:
            self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)
            
        # Update the goal expr for temporal tasks with a query timestep
        if 'timestep' in episode.instructions[self._chosen_instr_idx].task_type:
            self._update_goal_expr_for_temporal_task(episode)

        self.lang_goal = episode.instructions[self._chosen_instr_idx].lang
        self._check_for_pddl_assigns_to_ignore()
        print(f"Episode id: {episode.episode_id}, \tChosen instruction idx: {episode.instructions[self._chosen_instr_idx].task_id} -----> lang: {episode.instructions[self._chosen_instr_idx].lang}")

        # Reset measurements in preparation for the new instruction evaluation
        self._reset_pddl_verification_entities()
        self.pddl.sim_info.does_want_intermediate_stop = False
        self.measurements.reset_measures(
            episode=episode,
            task=self,
            observations=observations,
        )
        self.should_end = False
            
    @property
    def current_task_id(self):
        return self._sim.ep_info.instructions[self._chosen_instr_idx].task_id