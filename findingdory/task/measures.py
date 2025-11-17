import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import itertools
import networkx as nx
import os
import json

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.ovmm.sub_tasks.place_sensors import ObjAnywhereOnGoal
from habitat.tasks.rearrange.rearrange_sensors import NumStepsMeasure
from habitat.sims.habitat_simulator.actions import HabitatSimActions


@registry.register_measure
class PredicateTaskSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "predicate_task_success"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.is_goal_satisfied()
        
        if task._data_collection_phase:
            # the goal_expr is not available during the data collection phase
            return

        # _metric already checks if the PDDL intermediate stop action was invoked at the subgoal correctly 
        if self._metric:
            task.should_end = True
        
        # If the goal_expr is permanently False due to an out of order evaluation or invoking PDDL stop at
        # an incorrect location, we immediately terminate the episode unsuccessfully
        if task._goal_expr._permanently_false:
            task.should_end = True


@registry.register_measure
class PickedObjAnywhereOnGoal(ObjAnywhereOnGoal):
    '''
    This measure checks whether there exists any contact point between the picked object and the goal receptacle surface where the object is to be placed on
    We use this measure to find the specific point where the object touches the receptacle surface while lowering the robot arm (and the object)
    Once the first point of contact between the picked object and receptacle surface is detected, we slightly lift the arm back up and drop the object
    '''
    
    cls_uuid: str = "picked_obj_anywhere_on_goal"
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickedObjAnywhereOnGoal.cls_uuid

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._sim.perform_discrete_collision_detection()
        cps = self._sim.get_physics_contact_points()

        if task._current_oracle_goal_idx != -1:     # If idx = -1, data collection phase is over and we dont need this metric anymore
            query_label = self._sim._candidate_obj_idx_to_rigid_obj_handle[task._current_oracle_goal_idx]
            obj_id = self._sim.handle_to_object_id[query_label]
            abs_obj_id = self._sim.scene_obj_ids[obj_id]
            picked_idx = abs_obj_id
            self._metric = {str(picked_idx): False}
                
            for cp in cps:
                if cp.object_id_a == abs_obj_id or cp.object_id_b == abs_obj_id:
                    if cp.contact_distance < -0.01:
                        self._metric = {str(picked_idx): False}
                    else:
                        place_goal_id = int(task._current_oracle_goal['place_goal'].object_id)
                        
                        if place_goal_id == cp.object_id_a or place_goal_id == cp.object_id_b:   
                            # Get the contact point on the other object
                            contact_point = (
                                cp.position_on_a_in_ws
                                if place_goal_id == cp.object_id_a
                                else cp.position_on_b_in_ws
                            )
                            # Check if the other object has an id that is acceptable
                            self._metric = {
                                str(picked_idx): contact_point[1] >= self._config.max_floor_height  # ensure that the object is not on the floor
                            }
                        else:
                            continue        # Continue checking for the other ocntact points
                            
                        if self._metric[str(picked_idx)]:
                            return

        else:
            self._metric = -1


@registry.register_measure
class HighLevelGoalSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "high_level_goal_success"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        assert task._high_level_goal_success != -1, "Success metric was never assigned !"
        self._metric = task._high_level_goal_success
        
        
@registry.register_measure
class HighLevelDTGSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "high_level_dtg_success"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        assert task._high_level_dtg_success != -1, "Success metric was never assigned !"
        self._metric = task._high_level_dtg_success


@registry.register_measure
class HighLevelSemCovSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "high_level_sem_cov_success"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        assert task._high_level_sem_cov_success != -1, "Success metric was never assigned !"
        self._metric = task._high_level_sem_cov_success
         

@registry.register_measure
class SubGoalCount(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "subgoal_count"

    def reset_metric(self, *args, task, **kwargs):
        self._set_metric = False
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        # PDDL goal expr truth vals are only initialized when data collection ends so we return None as we dont know the number of subgoals
        if task._data_collection_phase:
            self._metric = -1
        else:
            if not self._set_metric:
                # Set the number of targets for the current task/episode. This will be used to verify if the VLM was successful or not
                multi_goal_task = task._sim.ep_info.instructions[task._chosen_instr_idx].sequential_goals
                if not multi_goal_task:
                    task._num_actual_targets = 1
                elif isinstance(multi_goal_task, dict):
                    if multi_goal_task['ordered']:
                        task._num_actual_targets = len([item for item in task._goal_expr._truth_vals if item is not None])
                    else:
                        task._num_actual_targets = len(task._goal_expr._truth_vals)
                self._set_metric = True
                
            self._metric = task._num_actual_targets


@registry.register_measure
class VLMFailureModeSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "vlm_failure_modes"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        
        metrics = {
            "misidentification": False,
            "response_error": False,
            "out_of_bounds_pred": False,
            "num_targets_error": False,
            "oracle_agent_timeout": False,
        }

        if task._high_level_goal_assigned and not task._high_level_goal_success:
            metrics = {
                "misidentification": not (task._vlm_response_error or task._out_of_bounds_pred or task._vlm_num_targets_error or task._oracle_agent_timeout),
                "response_error": task._vlm_response_error,
                "out_of_bounds_pred": task._out_of_bounds_pred,
                "num_targets_error": task._vlm_num_targets_error,
                "oracle_agent_timeout": task._oracle_agent_timeout,
            }
        
        # If oracle agent timeout, ensure that time out is the only reported failure mode to prevent double counting
        if metrics["oracle_agent_timeout"]:
            metrics = {
                "misidentification": False,
                "response_error": False,
                "out_of_bounds_pred": False,
                "num_targets_error": False,
                "oracle_agent_timeout": True,
            }
        
        self._metric = metrics

def _find_assign_with_min_d2g(assign_metrics, success_flag):
    """Finds ssignment with the minimum distance to goal (d2g)."""
    min_d2g = float('inf')
    selected_assign = None
    
    for assign, metrics in assign_metrics.items():
        assert metrics["dist_to_goal"] != -1, "DTG is -1 -> Metric was never set in PDDL verification"
        if metrics["success"] is success_flag and metrics["dist_to_goal"] < min_d2g:
            min_d2g = metrics["dist_to_goal"]
            selected_assign = assign
            
    return selected_assign

@registry.register_measure
class LowLevelPolicyFailureModeSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._low_level_policy_max_steps  = config.low_level_policy_max_steps
        self._relaxed_dtg_thresh = config.relaxed_dtg_thresh

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "c"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):

        metrics = {
            "low_level_policy_timeout": False,
            "incorrect_stop": False,
            "tgt_in_view_but_too_far": False,
            "tgt_in_view_slightly_far": False,
            "tgt_not_in_view_slightly_far": False,
            "tgt_not_in_view_too_far": False,
            "low_level_policy_failure": False,
        }
        
        # Was the low level policy successful or not ?
        low_level_success = task.measurements.measures[
            PredicateTaskSuccess._get_uuid()
        ].get_metric()
        
        high_level_policy_success = task.measurements.measures[
            HighLevelGoalSuccess._get_uuid()
        ].get_metric()
        
        # low level policy steps
        low_level_steps = task.measurements.measures[
            NumStepsMeasure.cls_uuid
        ].get_metric()
        
        if not task._data_collection_phase:
            # Delete assigns which have None truth values as they are not validated within the PDDL setup
            if task._pddl_assigns_to_ignore is not None:
                for assign_to_delete in task._pddl_assigns_to_ignore:
                    del task._ll_pddl_per_assign_metrics[assign_to_delete]
            assert len(task._ll_pddl_per_assign_metrics) > 0, "No per assign metrics remain for failure analysis !"
            
            # Only check for failure modes if low level policy has dailed and the intermediate STOP was called
            is_stop_called = False
            if 'action' in kwargs.keys():
                cur_action = kwargs['action']['action']
                if cur_action == "pddl_intermediate_stop" or cur_action == HabitatSimActions.stop:
                    is_stop_called = True
                    
            if high_level_policy_success and not low_level_success and is_stop_called:                
                metrics = {
                    "low_level_policy_timeout": False,
                    "incorrect_stop": False,
                    "tgt_in_view_but_too_far": False,
                    "tgt_in_view_slightly_far": False,
                    "tgt_not_in_view_slightly_far": False,
                    "tgt_not_in_view_too_far": False,
                    "high_level_policy_success": False,
                    "low_level_policy_failure": False,
                }

                if low_level_steps == self._low_level_policy_max_steps + 1:
                    metrics["low_level_policy_timeout"] = True
                
                # Low level policy executed to generate an action
                elif task._low_level_policy_failure:
                    print("Marking low level policy failure in metrics !")
                    metrics["low_level_policy_failure"] = True
                
                # Low level policy executed a STOP action which caused episode to Fail permanently
                else:                       
                    metrics["incorrect_stop"] = True
                    
                    min_dist_assign = _find_assign_with_min_d2g(task._ll_pddl_per_assign_metrics, success_flag=False)
                    
                    if min_dist_assign is not None:
                        print("Min dist assign: ", min_dist_assign)
                        print("Low level pddl assign metrics: ", task._ll_pddl_per_assign_metrics)
                        print("is stop called: ", is_stop_called)
                        
                    assign = task._ll_pddl_per_assign_metrics[min_dist_assign]
                    
                    if not assign["tgt_not_in_view"]:
                        if assign["dist_to_goal"] < self._relaxed_dtg_thresh:
                            metrics["tgt_in_view_slightly_far"] = True
                        else:
                            metrics["tgt_in_view_but_too_far"] = True
                    else:                        
                        if assign["dist_to_goal"] < self._relaxed_dtg_thresh:
                            metrics["tgt_not_in_view_slightly_far"] = True
                        else:
                            metrics["tgt_not_in_view_but_too_far"] = True
                                        
        self._metric = metrics
        

@registry.register_measure
class OracleSolutionGenerator(Measure):
    """
    Calculate the average temporal deviation between the keyframe selected by the VLM and the nearest keyframe corresponding to the goal entity.
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._output_path = config.result_save_path

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "oracle_solution_generator"

    def reset_metric(self, *args, task, **kwargs):
        self._set_metric = False
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        # PDDL goal expr truth vals are only initialized when data collection ends so we return None as we dont know the number of subgoals
        if not task._high_level_goal_assigned:
            self._metric = -1
        else:
            if not self._set_metric and task._full_success_metrics is not None and task._oracle_solution is not None:
                total_frames = task._num_data_collection_steps
                multi_goal_task = task._sim.ep_info.instructions[task._chosen_instr_idx].sequential_goals
                
                assert total_frames > 0, "Oracle video has no frames !"

                # Store keyframe data for each assign
                keyframe_data = {
                    "episode_id": task._sim.ep_info.episode_id,
                    "total_frames": total_frames,
                    "assigns": {}
                }
                
                print("Oracle solution: ", task._oracle_solution)

                # VLM predicted correct number of goals/non out-of-bounds subgoals -? PDDL verification as done -> we have the assign metrics
                if task._full_success_metrics != -1:
                    valid_keyframes = []
                    for assign_name, assign_metrics in task._full_success_metrics.items():
                        keyframe_data["assigns"][assign_name] = {
                            "keyframes": list(set(task._entity_keyframes[assign_name])),
                            "metrics": assign_metrics
                        }
                        valid_keyframes.extend(task._entity_keyframes[assign_name])
                
                # VLM predicted either incorrect num subgoals / out-of-bounds subgoals -> we didnt perform PDDL verification -> so just save the oracle task solution
                else:
                    valid_keyframes = []
                    for assign_name in task._oracle_entities:
                        assert assign_name in task._entity_keyframes, "Assignment not found in task keyframes dict which implies it was never tracked !"
                        if task._entity_keyframes[assign_name] is None:
                            assign_keyframes = [-1]  # Assignment has None value
                        elif len(task._entity_keyframes[assign_name]) == 0:
                            assign_keyframes = [-1]  # Assignment has empty list
                        else:
                            assign_keyframes = task._entity_keyframes[assign_name]  # Use actual keyframes
                        
                        keyframe_data["assigns"][assign_name] = {
                            "keyframes": list(set(assign_keyframes)),
                        }
                        valid_keyframes.extend(assign_keyframes)

                if not multi_goal_task:
                    valid_keyframes = sorted(list(set(valid_keyframes)))
                    print("Valid entity keyframes: ", valid_keyframes)
                    
                    res = min_deviation(valid_keyframes, task._high_level_nav_indices)
                    
                    self._metric = res[task._high_level_nav_indices[0]] / total_frames
                
                else:
                    if multi_goal_task['ordered']:
                        correct_assign_order = []
                        for idx in task.pddl.sim_info.sequential_goals['sub_expr_sequence']:
                            correct_assign_order.append(task.goal_expr.sub_exprs[idx].sub_exprs[0]._arg_values[0].name)
                        keyframe_data["assign_traversal_order"] = correct_assign_order
                        print("Valid assign traversal: ", correct_assign_order)
                        
                    print("Valid entity keyframes: ", keyframe_data["assigns"])
                    self._metric = -1       # TODO: We use -1 for multi goal tasks for now. Decide how to calculate this in future
                                
                # Store the predicted keyframe for this episode
                keyframe_data["vlm_selected_keyframes"] = task._high_level_nav_indices

                # Store the task instruction for this episode
                keyframe_data["task_instruction"] = task._sim.ep_info.instructions[task._chosen_instr_idx].lang

                # Create directory if it doesn't exist
                dir_name = os.path.join(self._output_path, "ep_id_" + task._sim.ep_info.episode_id)
                os.makedirs(dir_name, exist_ok=True)
                
                cur_task_id = task._sim.ep_info.instructions[task._chosen_instr_idx].task_id
                fname = os.path.join(dir_name, cur_task_id + ".json")
                with open(fname, 'w') as f:
                    json.dump(keyframe_data, f, indent=2)
                
                self._set_metric = True

def min_deviation(elements, query_list):
    return {q: min((abs(q - x) for x in elements), default=-1) for q in query_list}

@registry.register_measure
class FindingDorySPL(Measure):
    """SPL (Success weighted by Path Length) for VLM-selected navigation goals"""

    def __init__(
        self, sim: "Simulator", config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "findingdory_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        
        self._start_end_episode_distance = 0.0    
        self._set_shortest_goal_dist = False
            
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def _find_closest_target(self, metrics_dict):
        print("[SPL Calc] PDDL Low Level per assign metrics:")
        for assign_name, values in metrics_dict.items():
            print(f"  - {assign_name}: dist_to_goal = {values['dist_to_goal']}")
        
        min_item = min(metrics_dict.items(),
                    key=lambda x: x[1]['dist_to_goal'])
        
        print(f"[SPL Calc] Assignment with minimum dist_to_goal: {min_item[0]} = {min_item[1]['dist_to_goal']}")
        
        return min_item[0], min_item[1]['dist_to_goal']
    
    def update_metric(
        self, episode, task: "EmbodiedTask", *args: Any, **kwargs: Any
    ):
        
        if not self._set_shortest_goal_dist and task._high_level_goal_assigned and task._high_level_goal_success:
            multi_goal_task = task._sim.ep_info.instructions[task._chosen_instr_idx].sequential_goals
            
            if not multi_goal_task:
                _, self._start_end_episode_distance = self._find_closest_target(
                    task._ll_pddl_per_assign_metrics
                )
            
            elif isinstance(multi_goal_task, dict):
                if multi_goal_task['ordered']:
                    self._start_end_episode_distance = self._shortest_dist_ordered_revisitation(task)
                else:
                    self._start_end_episode_distance = self._shortest_dist_unordered_revisitation(task)
            else:
                raise ValueError("Unknown task instruction type -- SPL calculation not supported !")
            
            self._set_shortest_goal_dist = True
              
        # Get success from high level goal success measure
        ep_success = task.measurements.measures[
            PredicateTaskSuccess._get_uuid()
        ].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += task._sim.geodesic_distance(
            current_position, self._previous_position
        )
        
        self._previous_position = current_position

        if self._start_end_episode_distance <= 0. and self._agent_episode_distance <= 0.:
            if ep_success:
                self._metric = 1
            else:
                self._metric = 0
        else:
            self._metric = ep_success * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance, self._agent_episode_distance
                )
            )
    
    def _shortest_dist_ordered_revisitation(self, task):
        shortest_dist = 0.
        prev_pos = task._sim.get_agent_state().position
        rom = task._sim.get_rigid_object_manager()

        for i in task.pddl.sim_info.sequential_goals['sequence']:
            obj_name = task._sim._candidate_obj_idx_to_rigid_obj_handle[i]
            obj_pos = task.pddl.sim_info.sim.safe_snap_point(rom.get_object_by_handle(obj_name).transformation.translation)
            obj_dist = task.pddl.sim_info.sim.geodesic_distance(prev_pos, obj_pos)
            shortest_dist += obj_dist
            prev_pos = obj_pos
        
        print("Min distance for ordered revisitation: ", shortest_dist)
        
        return shortest_dist

    def _shortest_dist_unordered_revisitation(self, task):
        """
        Calculates the shortest path for unordered revisitation tasks
        by finding the minimal path distance to visit all goals in any order.
        """
                
        num_goals = len(task._ll_pddl_per_assign_metrics.keys())
        
        current_position = task._sim.get_agent_state().position
        rom = task._sim.get_rigid_object_manager()

        # Extract goal positions
        goal_positions = []
        for obj_name in task._ll_pddl_per_assign_metrics.keys():
            if '|' in obj_name:             # NOTE: receptacle objects names have ".....|receptacle_mesh..." so we need to extract the part of the name before pipe character
                obj_name = obj_name.split("|")[0]
            goal_positions.append([
                task.pddl.sim_info.sim.safe_snap_point(
                    rom.get_object_by_handle(obj_name).transformation.translation
                ),
                obj_name
            ])
        
        if num_goals <= 6:      # Upto 6 cities, brute force has comparable speed and guarantees the optimal shortest path
            min_distance = self._tsp_solution_brute(task, current_position, goal_positions)
        else:
            min_distance = self._tsp_solution_christofide(task, current_position, goal_positions)
            
        print("Min distance for unordered revisitation: ", min_distance)
            
        return min_distance

    def _tsp_solution_christofide(self, task, current_position, goal_positions):
        """
        Calculates the TSP solution using the Christofide algorithm from the networkX library. O(n^3) time complexity
        Not guaranteed to compute the optimal solution but much faster than the brute force approach for optimal solutions 
        """
        
        print("Calculating TSP solution using Christofide...")

        # Extract separate lists of positions and names
        goal_names = [goal[1] for goal in goal_positions]
        goal_positions = [goal[0] for goal in goal_positions]

        # Number of goals
        n = len(goal_positions)

        # If there's only one goal, just return the geodesic distance
        if n == 1:
            return task.pddl.sim_info.sim.geodesic_distance(current_position, goal_positions[0])

        # Construct the distance matrix (including the agent's current position)
        all_positions = [current_position] + goal_positions  # Add agent start position at index 0
        distance_matrix = np.zeros((n + 1, n + 1))

        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    distance_matrix[i][j] = task.pddl.sim_info.sim.geodesic_distance(all_positions[i], all_positions[j])

        # Solve TSP using Christofides algorithm
        tsp_path = tsp_christofides(distance_matrix)

        # Compute total path distance along the computed tour
        min_distance = sum(distance_matrix[tsp_path[i]][tsp_path[i + 1]] for i in range(len(tsp_path) - 1))
        
        return min_distance

    def _tsp_solution_brute(self, task, current_position, goal_positions):
        """
        Calculates the TSP solution using brute force enumeration algorithm. O(n!) time complexity
        Guaranteed to compute the optimal solution
        """
        
        print("Calculating TSP solution using brute force...")

        # Separate out the goal_pos vectors and goal_names
        goal_names = []
        goal_pos = []
        for goal_info in goal_positions:
            goal_pos.append(goal_info[0])
            goal_names.append(goal_info[1])
        goal_positions = goal_pos

        # Compute the geodesic distance from the current position to each goal
        start_to_goals = [
            task.pddl.sim_info.sim.geodesic_distance(current_position, goal_pos)
            for goal_pos in goal_positions
        ]

        # Compute all pairwise distances between goals (store both directions)
        goal_distances = {}
        for i in range(len(goal_positions)):
            for j in range(i + 1, len(goal_positions)):
                dist = task.pddl.sim_info.sim.geodesic_distance(goal_positions[i], goal_positions[j])
                goal_distances[(i, j)] = dist
                goal_distances[(j, i)] = dist  # Add the reverse pair

        # Initialize variables to track the minimum distance and best order
        min_distance = float("inf")
        best_order = None

        # Find the shortest path using permutations
        for perm in itertools.permutations(range(len(goal_positions))):
            # Calculate total distance for this permutation
            total_distance = start_to_goals[perm[0]]  # From start to the first goal
            for i in range(len(perm) - 1):
                total_distance += goal_distances[(perm[i], perm[i + 1])]

            # Update minimum distance and best order if the current permutation is better
            if total_distance < min_distance:
                min_distance = total_distance
                best_order = perm

        return min_distance
    
def tsp_christofides(distance_matrix):
    """
    Solves TSP approximately using the Christofides algorithm.
    Returns the optimal visit order for the shortest path.
    """
    n = len(distance_matrix)
    G = nx.complete_graph(n)

    # Assign edge weights based on the distance matrix
    for i in range(n):
        for j in range(i + 1, n):
            G[i][j]['weight'] = distance_matrix[i][j]
            G[j][i]['weight'] = distance_matrix[i][j]  # Ensure undirected graph

    # Use NetworkX's built-in approximation TSP solver
    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)

    return tsp_path

@registry.register_measure
class FindingDoryHighLevelSPL(Measure):
    """SPL (Success weighted by Path Length) for VLM-selected navigation goals without considering the low-level policy"""

    def __init__(
        self, sim: "Simulator", config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "findingdory_high_level_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._set_high_level_spl = False
            
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _find_closest_target(self, metrics_dict):
        print("[SPL Calc HIGH Level] PDDL Low Level per assign metrics:")
        for assign_name, values in metrics_dict.items():
            print(f"  - {assign_name}: dist_to_goal = {values['dist_to_goal']}")

        # Filter out elements where dist_to_goal == -1 --> these PDDL entities correspond to the ones that were not tested during high level PDDL verification
        # This happens due to early termination in PDDL logical expr evaluation (ex: for OR logical expr -> PDDL ignores any assign checking after encountering first True assign )
        filtered_metrics = {k: v for k, v in metrics_dict.items() if v['dist_to_goal'] != -1}

        # Ensure there are valid entries left
        if not filtered_metrics:
            print("[SPL Calc HIGH Level] No valid assignments found.")
            return None, None

        # Find the assignment with the minimum dist_to_goal
        min_item = min(filtered_metrics.items(), key=lambda x: x[1]['dist_to_goal'])

        print(f"[SPL Calc HIGH Level] Assignment with minimum dist_to_goal: {min_item[0]} = {min_item[1]['dist_to_goal']}")

        return min_item[0], min_item[1]['dist_to_goal']
    
    def _check_min_d2g_valid_assign(self, assign_metrics):
        """Finds ssignment with the minimum distance to goal (d2g)."""
        min_d2g = float('inf')
        selected_assign = None

        for assign, metrics in assign_metrics.items():
            if metrics["dist_to_goal"] == -1:  # Skip entries with dist_to_goal == -1
                continue
            
            if metrics["success"] is True and metrics["dist_to_goal"] < min_d2g:
                min_d2g = metrics["dist_to_goal"]
                selected_assign = assign
                
        return selected_assign
    
    def update_metric(
        self, episode, task: "EmbodiedTask", *args: Any, **kwargs: Any
    ):
        
        if not self._set_high_level_spl and task._high_level_goal_assigned:
            if not task._high_level_goal_success:
                self._metric = 0
            
            else:
                multi_goal_task = task._sim.ep_info.instructions[task._chosen_instr_idx].sequential_goals
                
                if not multi_goal_task:
                    # Find the distance to nearest valid assign
                    _, shortest_dist = self._find_closest_target(
                        task._ll_pddl_per_assign_metrics
                    )
                    
                    # Find the distance to the assign that was selected to satsify the high level success criteria
                    chosen_assign = self._check_min_d2g_valid_assign(task._full_success_metrics)
                    actual_dist = task._ll_pddl_per_assign_metrics[chosen_assign]["dist_to_goal"]
                    if actual_dist == 0:
                        self._metric = 1.0  # Perfect SPL when agent is already at the goal
                    else:
                        self._metric = shortest_dist / max(shortest_dist, actual_dist)
                    print("High Level SPL: ", self._metric)

                # Multi-goal SPL calc for high-level goal not supported yet
                else:
                    self._metric = -1
            
            self._set_high_level_spl = True