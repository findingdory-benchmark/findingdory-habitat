#!/usr/bin/env python3
from typing import Any
import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.spaces import EmptySpace
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.ovmm.sub_tasks.nav_to_obj_sensors import MultiObjectSegmentationSensor
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from findingdory.task.utils import iterate_action_space_recursively_with_keys
from findingdory.task.text_space_gymnasium import Text, AgentStateSpace, KeyframesStateSpace
from findingdory.utils import ACTION_MAPPING


@registry.register_sensor
class LangMemoryGoalSensor(Sensor):
    cls_uuid = "lang_memory_goal"

    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return Text(
            min_length=1,
            max_length=256,
        )

    def get_observation(self, task, *args, **kwargs):
        return task.lang_goal


@registry.register_sensor
class TimeOfDaySensor(Sensor):
    cls_uuid: str = "time_of_day"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._wake_up_time = config.wake_up_time
        self._sleep_time = config.sleep_time

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return Text(
            min_length=1,
            max_length=64,
        )

    def get_observation(self, task, *args, **kwargs):
        
        # Return dummy time of day if num_steps_daily is set to -1. This happens when we need to override the data collector agent actions such as during place viewpoint locator execution
        if task.num_steps_daily == -1:
            return "00:00"
        
        current_steps = task.num_steps
        num_steps_daily = task.num_steps_daily
        total_hours = self._sleep_time - self._wake_up_time
        mins_elapsed = current_steps / num_steps_daily * total_hours * 60

        # time will be in hh:mm format starting from wake up time (hh:00) and ending at sleep time (hh:00)
        current_hour = self._wake_up_time + int(mins_elapsed // 60)
        current_minute = np.floor(mins_elapsed % 60).astype(int)

        return f"{current_hour:02d}:{current_minute:02d}"


@registry.register_sensor
class OtherObjectSegmentationSensor(MultiObjectSegmentationSensor):
    cls_uuid: str = "other_object_segmentation"

    def _loaded_object_categories(self, task):
        return task.loaded_other_object_categories

    def _get_semantic_ids(self, task):
        return task.other_object_semantic_ids


@registry.register_sensor
class LastActionSensor(Sensor):
    cls_uuid: str = "last_action"

    def __init__(self, *args, config, **kwargs):
        self._max_action_len = config.max_action_len
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return LastActionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, task, **kwargs):
        return spaces.Box(
            low=0,
            high=np.inf,
            shape=(self._max_action_len,),
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        last_action = np.zeros(self._max_action_len, dtype=np.float32)
        if task.last_action is None:
            return last_action

        action_idx = 0
        for action, action_name in iterate_action_space_recursively_with_keys(
            task.action_space
        ):
            if type(task.last_action['action']) == str:
                task_action_name = task.last_action['action']
            else:
                task_action_name = ACTION_MAPPING[task.last_action['action']]

            if isinstance(action, EmptySpace):
                if action_name == task_action_name:
                    last_action[action_idx] = 1.0
                    break
                action_idx += 1
            elif isinstance(action, spaces.Box):
                if action_name == task_action_name:
                    last_action[
                        action_idx:action_idx + action.shape[0]
                    ] = task.last_action['action_args'][action_name]
                    break
                action_idx += action.shape[0]
            else:
                raise NotImplementedError

        return last_action

@registry.register_sensor
class CanTakeActionSensor(Sensor):
    cls_uuid: str = "can_take_action"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return CanTakeActionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, task, **kwargs):
        return spaces.Discrete(2)

    def get_observation(self, *args, task, **kwargs):
        if task.num_steps < task.num_steps_daily and task._data_collection_phase:
            return 0
        return 1

@registry.register_sensor
class ManipulationModeSensor(Sensor):
    cls_uuid: str = "manipulation_mode"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return ManipulationModeSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, task, **kwargs):
        return spaces.Discrete(2)

    def get_observation(self, *args, task, **kwargs):
        return task._in_manip_mode
    
@registry.register_sensor
class AgentStateSensor(Sensor):
    cls_uuid: str = "agent_state"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return AgentStateSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return AgentStateSpace()

    def get_observation(self, *args, task, **kwargs):
        return task._sim.get_agent_state()
    
@registry.register_sensor
class OracleKeyframesSensor(Sensor):
    cls_uuid: str = "oracle_keyframes"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return OracleKeyframesSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return KeyframesStateSpace()

    def get_observation(self, *args, task, **kwargs):
        
        if not task._data_collection_phase:
            valid_entities = []
            for sub_exprs in task.goal_expr.sub_exprs:
                valid_entities.append(sub_exprs.sub_exprs[0]._arg_values[0].name)
            
            valid_frames = []                    
            multi_goal_task = task._sim.ep_info.instructions[task._chosen_instr_idx].sequential_goals
            
            # Return a single list element which just combines all possible keyframes
            # Return list [-1] in case no keyframe would give task success for that entity
            if not multi_goal_task:
                for entity in valid_entities:
                    valid_frames.extend(task._entity_keyframes[entity] or [-1])
                valid_frames = [valid_frames]

            elif isinstance(multi_goal_task, dict):
                
                # If ordered revisitation task, then we return the oracle keyframes for each entity as sublists arranged in the order specified in task instruction  
                if multi_goal_task['ordered']:
                    valid_entities_ordered = []
                    for idx in task.pddl.sim_info.sequential_goals['sub_expr_sequence']:
                        entity = valid_entities[idx]
                        
                        # If the subgoal entity has connected receptacles, then we return the oracle keyframes for all connected receptacles
                        if entity in task.connected_recep_names:
                            all_valid_frames = self._get_all_valid_frames(task, entity)
                            valid_frames.append(sorted(all_valid_frames) or [-1])
                        else:
                            valid_frames.append(task._entity_keyframes[entity] or [-1])
                            
                        valid_entities_ordered.append(entity)
                    valid_entities = valid_entities_ordered
                
                # If unordered revisitation task, then we return the oracle keyframes for each entity as sublists (no explicit order)
                else:
                    for entity in valid_entities:
                        # If the subgoal entity has connected receptacles, then we return the oracle keyframes for all connected receptacles
                        if entity in task.connected_recep_names:
                            all_valid_frames = self._get_all_valid_frames(task, entity)
                            valid_frames.append(sorted(all_valid_frames) or [-1])
                        else:
                            valid_frames.append(task._entity_keyframes[entity] or [-1])
                        
            task._oracle_solution = valid_frames
            task._oracle_entities = valid_entities
            
            return valid_frames
    
        else:
            task._oracle_solution = None
            return []
        
    def _get_all_valid_frames(self, task, entity):
        '''
        Cycle through all connected receptacles and add their keyframes to the list of valid frames
        '''
        all_valid_frames = task._entity_keyframes[entity] or []
        for connected_receps in task.connected_recep_names[entity]:
            all_valid_frames.extend(task._entity_keyframes[connected_receps] or [])
        return list(set(all_valid_frames))