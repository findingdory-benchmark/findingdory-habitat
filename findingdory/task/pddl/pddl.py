from dataclasses import dataclass
from enum import Enum
import inspect
import os.path as osp
from typing import Dict, Optional, cast, Union, List, Set, Tuple

import numpy as np
import magnum as mn
from habitat.tasks.nav.object_nav_task import ObjectGoal
from habitat.datasets.rearrange.samplers.receptacle import Receptacle, TriangleMeshReceptacle
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.pddl_defined_predicates import is_inside
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import PddlSimInfo, ExprType, SimulatorObjectType
from habitat.tasks.rearrange.utils import get_aabb
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.articulated_agents.robots.stretch_robot import StretchRobot

from findingdory.dataset.findingdory_dataset import FindingDoryEpisode
from findingdory.task.utils import get_closest_dist_to_region


class FindingDorySimulatorObjectType(Enum):
    """
    Predefined entity types for which default predicate behavior is defined.
    """
    BASE_ENTITY = SimulatorObjectType.BASE_ENTITY.value
    MOVABLE_ENTITY = SimulatorObjectType.MOVABLE_ENTITY.value
    STATIC_RECEPTACLE_ENTITY = SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
    ARTICULATED_RECEPTACLE_ENTITY = SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
    GOAL_ENTITY = SimulatorObjectType.GOAL_ENTITY.value
    ROBOT_ENTITY = SimulatorObjectType.ROBOT_ENTITY.value
    POSITIONAL_ENTITY = "pos_entity_type"
    ROOM_ENTITY = "room_region_entity"


class TaskPosData:
    new_start_pos: Tuple[np.ndarray, float] = None
    old_start_pos: Tuple[np.ndarray, float] = None

    def keys(self):
        return [k for k in TaskPosData.__dict__.keys() if not k.startswith("__") and not inspect.isfunction(getattr(TaskPosData, k))]


class DiscPddlEntityProperty:
    def __init__(self, name: str, value: Union[str, bool]):
        assert isinstance(name, str)
        assert isinstance(value, str) or isinstance(value, bool)
        self.name = name
        self.value = value
    
    def __eq__(self, other):
        if not isinstance(other, DiscPddlEntityProperty):
            return False

        return self.name == other.name and self.value == other.value

    def __repr__(self):
        if isinstance(self.value, bool):
            if self.value:
                return f"{self.name}"
            else:
                return f"not {self.name}"
        return f"{self.name} = {self.value}"
    
    def __hash__(self):
        return hash(self.name)


class ContPddlEntityProperty:
    def __init__(self, name: str, value: float):
        assert isinstance(name, str)
        assert isinstance(value, float)
        self.name = name
        self.value = value
    
    def __eq__(self, other):
        if not isinstance(other, ContPddlEntityProperty):
            return False

        return self.name == other.name and self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, ContPddlEntityProperty):
            return False

        return self.name == other.name and self.value < other.value

    def __gt__(self, other):
        if not isinstance(other, ContPddlEntityProperty):
            return False

        return self.name == other.name and self.value > other.value

    def __repr__(self):
        return f"{self.name} = {self.value}"
    
    def __hash__(self):
        return hash(self.name)

class MultiDiscPddlEntityProperty:
    def __init__(self, name: str, values: list[Union[bool, str]]):
        assert isinstance(name, str), "Property name must be a string"
        assert isinstance(values, list), "Values must be a list"
        assert all(isinstance(v, (bool, str)) for v in values), "Values must be a list of bools or strings"
        
        self.name = name
        self.values = values  # List of discrete values
    
    def __eq__(self, other):
        """Compares with MultiDiscPddlEntityProperty and checks if the value exists in the list of values."""

        # If the comaprison is made with a DiscPddlEntityProperty instance which has single value
        if isinstance(other, DiscPddlEntityProperty):            
            # Return True if the names match and the discrete value exists in the list of values
            return self.name == other.name and other.value in self.values
        elif isinstance(other, MultiDiscPddlEntityProperty):
            # Return True if the names match, all values match, and lengths are the same
            return self.name == other.name and self.values == other.values and len(self.values) == len(other.values)
        
        return False

    def __repr__(self):
        return f"{self.name} = {self.values}"
    
    def __hash__(self):
        return hash(self.name)

class MultiContPddlEntityProperty:
    def __init__(self, name: str, value: list[float]):
        assert isinstance(name, str), "Property name must be a string"
        assert isinstance(value, list) and all(isinstance(v, float) for v in value), "Values must be a list of floats"

        self.name = name
        self.value = value  # List of continuous values

    def __eq__(self, other):
        """Compares with MultiContPddlEntityProperty and checks if the value exists in the list of values."""
        # If the comaprison is made with a ContPddlEntityProperty instance which has single value
        if isinstance(other, ContPddlEntityProperty):
            # Return True if the names match and the value exists in the list of values
            return self.name == other.name and other.value in self.value
        elif isinstance(other, MultiContPddlEntityProperty):
            # Return True if the names match, all values match, and lengths are the same
            return self.name == other.name and self.value == other.value and len(self.value) == len(other.value)

        return False

    def __repr__(self):
        return f"{self.name} = {self.value}"

    def __hash__(self):
        """Hash based on name."""
        return hash(self.name)


@dataclass(frozen=True)
class FindingDoryPddlEntity:
    """
    Abstract PDDL entity. This is linked to simulator via `PddlSimInfo`.
    """

    name: str
    expr_type: ExprType
    properties: Optional[Set[Union[DiscPddlEntityProperty, ContPddlEntityProperty, MultiContPddlEntityProperty]]] = None

    def __repr__(self):
        if self.properties is not None:
            return f"{self.name}-{self.expr_type}-{self.properties}"
        else:
            return f"{self.name}-{self.expr_type}"

    def __eq__(self, other):
        if not isinstance(other, FindingDoryPddlEntity):
            return False

        # All other types must be in our types.
        return (self.name == other.name) and (
            self.expr_type.name == other.expr_type.name
        )


@dataclass
class FindingDoryPddlSimInfo(PddlSimInfo):
    receptacles_viewpoints: Dict[str, ObjectGoal] = None
    episode: FindingDoryEpisode = None
    sequential_goals: Dict = None
    does_want_intermediate_stop: bool = False

    def get_entity_viewpoints(self, entity: FindingDoryPddlEntity) -> np.ndarray:
        """
        Gets the viewpoints associated with an entity.
        """
        ename = entity.name
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ):
            recep = self.receptacles_viewpoints.get(ename, None)
            # Receptacle didnt have any viewpoints
            if recep is None:
                return None
            episode_view_points = [
                view_point.agent_state.position
                for view_point in recep.view_points
            ]

            return np.array(episode_view_points)

        # NOTE: Do not use the viewpoints of the Movable entities since they will not be
        # valid after the object moves.
        
        return None
    
    def get_entity_angle(self, entity: FindingDoryPddlEntity) -> np.ndarray:
        """
        Gets the angle associated with an entity.
        """
        ename = entity.name
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.POSITIONAL_ENTITY.value
        ):
            pos = getattr(self.env.task_pos_data, ename, None)
            return pos[1] if pos is not None else None

        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.ROBOT_ENTITY.value
        ):
            robot_id = self.robot_ids[ename]
            robot = self.sim.get_agent_data(robot_id).articulated_agent
            curr_angle = float(robot.sim_obj.rotation.angle())
            # between 0 and 2pi (TODO(KY): COnfirm this is okay)
            curr_angle = curr_angle % (2 * np.pi)
            return curr_angle
        
        return None
    
    def get_entity_bounds(self, entity: FindingDoryPddlEntity) -> np.ndarray:
        """
        Gets the bounds of an entity.
        """
        ename = entity.name
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.MOVABLE_ENTITY.value
        ):
            idx = self.obj_ids[ename]
            abs_obj_id = self.sim.scene_obj_ids[idx]
            return get_aabb(abs_obj_id, self.sim, transformed=True)
            
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ):
            recep = self.receptacles[ename]
            return np.array(recep.bounds)


    def get_entity_pos(self, entity: FindingDoryPddlEntity) -> np.ndarray:
        """
        Gets a simulator 3D point for an entity.
        """
        ename = entity.name
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.ROBOT_ENTITY.value
        ):
            robot_id = self.robot_ids[ename]
            return self.sim.get_agent_data(robot_id).articulated_agent.base_pos
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ):
            marker_info = self.marker_handles[ename]
            return marker_info.get_current_position()
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.GOAL_ENTITY.value
        ):
            idx = self.target_ids[ename]
            targ_idxs, pos_targs = self.sim.get_targets()
            rel_idx = targ_idxs.tolist().index(idx)
            return pos_targs[rel_idx]
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ):
            recep = self.receptacles[ename]
            return np.array(recep.get_surface_center(self.sim))

            # return np.array(recep.get_global_transform(self.sim).translation)
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.MOVABLE_ENTITY.value
        ):
            rom = self.sim.get_rigid_object_manager()
            idx = self.obj_ids[ename]
            abs_obj_id = self.sim.scene_obj_ids[idx]
            cur_pos = rom.get_object_by_id(
                abs_obj_id
            ).transformation.translation
            return cur_pos
        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.POSITIONAL_ENTITY.value
        ):
            pos = getattr(self.env.task_pos_data, ename, None)
            return pos[0] if pos is not None else None
        raise ValueError()

    def search_for_entity(
        self, entity: FindingDoryPddlEntity
    ) -> Union[int, str, MarkerInfo, Receptacle]:
        """
        Returns underlying simulator information associated with a PDDL entity.
        Helper to match the PDDL entity to something from the simulator.
        """

        ename = entity.name

        if self.check_type_matches(
            entity, FindingDorySimulatorObjectType.ROBOT_ENTITY.value
        ):
            return self.robot_ids[ename]
        elif self.check_type_matches(
            entity, FindingDorySimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ):
            return self.marker_handles[ename]
        elif self.check_type_matches(
            entity, FindingDorySimulatorObjectType.GOAL_ENTITY.value
        ):
            return self.target_ids[ename]
        elif self.check_type_matches(
            entity, FindingDorySimulatorObjectType.MOVABLE_ENTITY.value
        ):
            return self.obj_ids[ename]
        elif self.check_type_matches(
            entity, FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ):
            return self.receptacles[ename]
        else:
            raise ValueError(f"No type match for {entity}")


def is_robot_at_position(
    at_entity,
    sim_info,
    dist_thresh: float,
    semantic_cov_thresh: float,
    dist_measure: str = "geodesic",
    robot=None,
    angle_thresh: Optional[float] = None,
):
    """
    Check if the robot is at a position within a distance threshold.
    """
    
    # Metrics dict for tracking detailed failure modes for the low level policy
    sim_info.env._ll_pddl_per_assign_metrics[at_entity.name] = {
        "success": False,
        "dist_to_goal": -1.,
        "angle_to_goal": -1.,
        "semantic_cov": -1.,
        "d2g_exceed": False,
        "a2g_exceed": False,
        "tgt_not_in_view": False,
    }
    
    predicate_result = True
    
    if robot is None:
        robot_obj = sim_info.sim.get_agent_data(None).articulated_agent
    else:
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot),
        )
        robot_obj = sim_info.sim.get_agent_data(robot_id).articulated_agent

    targ_pos = sim_info.get_entity_pos(at_entity)

    if targ_pos is None:
        return False
    
    # Extract the target entity instance mask and instance ID
    instance_mask = None
    entity_instance_id = None
    
    # If _pddl_verification_obs is not None, it implies we are checking high level VLM PDDL success for the state selected by the VLM which is captured in sim_info.env._pddl_verification_obs 
    if sim_info.env._pddl_verification_obs is not None:
        instance_mask = sim_info.env._pddl_verification_obs['head_panoptic']
    # Otherwise, just use the current observation from simulator for verifying PDDL success for the low level action policy
    else:
        instance_mask = sim_info.env._last_observation['head_panoptic']
    
    # Retrieve the instance ID of the target entity
    if at_entity.expr_type.parent.name == "static_receptacle_entity_type":            
        rom = sim_info.sim.get_rigid_object_manager()
        entity_obj_id = rom.get_object_id_by_handle(at_entity.name.split("|")[0])        
        entity_instance_id = entity_obj_id + sim_info.sim._object_ids_start
    elif at_entity.expr_type.parent.name == "movable_entity_type":
        obj_id = sim_info.sim.handle_to_object_id[at_entity.name]
        scene_obj_id = sim_info.sim.scene_obj_ids[obj_id]
        entity_instance_id = scene_obj_id + sim_info.sim._object_ids_start
    
    # Calculate the instance coverage of the target entity to ensure the target entity is actually in view of the agent
    if instance_mask is not None and entity_instance_id is not None:
        entity_instance_mask = instance_mask == entity_instance_id
        entity_semantic_coverage = entity_instance_mask.sum() / np.prod(entity_instance_mask.shape)
        
        _log_metric(sim_info, at_entity, "semantic_cov", entity_semantic_coverage)
        if entity_semantic_coverage < semantic_cov_thresh:
            _log_metric(sim_info, at_entity, "tgt_not_in_view", True)
            predicate_result = False
        
    # Get the base transformation
    T = robot_obj.base_transformation
    # Do transformation
    pos = T.inverted().transform_point(targ_pos)
    # Project to 2D plane (x,y,z=0)
    pos[2] = 0.0

    # Compute distance
    if dist_measure == "geodesic":
        viewpoints = sim_info.get_entity_viewpoints(at_entity)
        if viewpoints is not None:
            obj_pos = viewpoints
            dist = sim_info.sim.geodesic_distance(robot_obj.base_pos, obj_pos)
        else:
            try:
                obj_pos = sim_info.sim.safe_snap_point(targ_pos)
                dist = sim_info.sim.geodesic_distance(robot_obj.base_pos, obj_pos)
            except:
                interacted_entities = sim_info.env.candidate_start_receps_instances(sim_info.sim.ep_info) + \
                    sim_info.env.candidate_objects_instances(sim_info.sim.ep_info) + \
                    sim_info.env.candidate_goal_receps_instances(sim_info.sim.ep_info) + \
                    sim_info.env.candidate_objects_noninteracted_instances(sim_info.sim.ep_info)
                assert (
                    at_entity.name not in interacted_entities
                ), f"Failed PDDL verification for entity: {at_entity.name}. Interacted entities: {interacted_entities}"
                dist = float('inf')
    elif dist_measure == "euclidean":
        dist = np.linalg.norm(pos)
    else:
        raise ValueError(f"Unknown distance measure: {dist_measure}")

    # Updated high-level/low-level metric based on if the vlm_predicted_obs is supplied or not
    _log_metric(sim_info, at_entity, "dist_to_goal", dist)
    if dist > dist_thresh:
        _log_metric(sim_info, at_entity, "d2g_exceed", True)
        predicate_result = False

    # Unit vector of the pos
    targ_angle = sim_info.get_entity_angle(at_entity)
    if targ_angle is None:
        targ_angle = pos.normalized() if np.linalg.norm(pos) > 0 else pos
        # Define the coordinate of the robot
        robot_angle = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.dot(targ_angle, robot_angle))
        
        # Stretch URDF forward direction is offset by 90 with respect actual forward camera direction 
        if isinstance(robot_obj, StretchRobot) and sim_info.env._in_manip_mode:
            angle -= np.pi / 2
    else:
        robot_angle = sim_info.get_entity_angle(robot)
        angle = (targ_angle - robot_angle) 

    # Updated high-level/low-level metric based on if the vlm_predicted_obs is supplied or not
    _log_metric(sim_info, at_entity, "angle_to_goal", np.abs(angle))
    if angle_thresh is not None and np.abs(angle) > angle_thresh:
        _log_metric(sim_info, at_entity, "a2g_exceed", True)
        predicate_result = False

    # Updated high-level/low-level metric based on if the vlm_predicted_obs is supplied or not
    _log_metric(sim_info, at_entity, "success", predicate_result)

    return predicate_result

def is_robot_in_room(
    at_entity,
    sim_info,
    robot=None
):
    
    assert at_entity.expr_type.name == FindingDorySimulatorObjectType.ROOM_ENTITY.value, "Supplied entity for predicate verification should be a room entity !"
    
    sim_info.env._ll_pddl_per_assign_metrics[at_entity.name] = {
        "success": False,
        "dist_to_goal": -1.,
        "angle_to_goal": -1.,
        "semantic_cov": -1.,
        "d2g_exceed": False,
        "a2g_exceed": False,
        "tgt_not_in_view": False,
    }
    sim_info.env._hl_pddl_per_assign_metrics[at_entity.name] = {
        "success": False,
        "dist_to_goal": -1.,
        "angle_to_goal": -1.,
        "semantic_cov": -1.,
        "d2g_exceed": False,
        "a2g_exceed": False,
        "tgt_not_in_view": False,
    }

    
    entity_region_id = at_entity.name.split("region_")[1]
        
    if robot is None:
        robot_obj = sim_info.sim.get_agent_data(None).articulated_agent
    else:
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot),
        )
        robot_obj = sim_info.sim.get_agent_data(robot_id).articulated_agent

    robot_pos = robot_obj.base_pos
    regions = sim_info.env._sim.semantic_scene.get_regions_for_point(robot_pos)

    if len(regions) == 0:
        sim_info.env._ll_pddl_per_assign_metrics[at_entity.name]["success"] = False
        sim_info.env._hl_pddl_per_assign_metrics[at_entity.name]["success"] = False
        _log_metric(sim_info, at_entity, "dist_to_goal", 10000.)
        _log_metric(sim_info, at_entity, "angle_to_goal", np.pi)
        _log_metric(sim_info, at_entity, "semantic_cov", 0.)
        _log_metric(sim_info, at_entity, "d2g_exceed", True)
        _log_metric(sim_info, at_entity, "a2g_exceed", True)
        _log_metric(sim_info, at_entity, "tgt_not_in_view", True)
        return False

    # Check if the robot is in any of the regions returned by get_regions_for_point
    for region_id in regions:
        region_name = sim_info.env._sim.semantic_scene.regions[region_id].id

        # If the robot is in the target region, return True
        if region_name == entity_region_id:
            _log_metric(sim_info, at_entity, "success", True)
            _log_metric(sim_info, at_entity, "dist_to_goal", 0)
            _log_metric(sim_info, at_entity, "angle_to_goal", 0)
            _log_metric(sim_info, at_entity, "semantic_cov", 1.0)
            _log_metric(sim_info, at_entity, "d2g_exceed", False)
            _log_metric(sim_info, at_entity, "a2g_exceed", False)
            _log_metric(sim_info, at_entity, "tgt_not_in_view", False)
            return True
    
    # If we've checked all regions and none match, the robot is not in the target region
    dist = get_closest_dist_to_region(
        sim_info.env._sim,
        entity_region_id,
        robot_pos,
    )
    _log_metric(sim_info, at_entity, "success", False)
    _log_metric(sim_info, at_entity, "dist_to_goal", dist)
    _log_metric(sim_info, at_entity, "angle_to_goal", np.pi)
    _log_metric(sim_info, at_entity, "semantic_cov", 0)
    _log_metric(sim_info, at_entity, "d2g_exceed", False)
    _log_metric(sim_info, at_entity, "a2g_exceed", True)
    _log_metric(sim_info, at_entity, "tgt_not_in_view", True)
    return False

def is_object_at(
    obj: FindingDoryPddlEntity,
    at_entity: FindingDoryPddlEntity,
    sim_info: PddlSimInfo,
    dist_thresh: float,
) -> bool:
    """
    Checks if an object entity is logically at another entity. At an object
    means within a threshold of that object. At a receptacle means on the
    receptacle. At a articulated receptacle means inside of it.
    """
    entity_pos = sim_info.get_entity_pos(obj)

    if sim_info.check_type_matches(
        at_entity, FindingDorySimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
    ):
        # Object is rigid and target is receptacle, we are checking if
        # an object is inside of a receptacle.
        return is_inside(obj, at_entity, sim_info)
    elif sim_info.check_type_matches(
        at_entity, FindingDorySimulatorObjectType.GOAL_ENTITY.value
    ) or sim_info.check_type_matches(
        at_entity, FindingDorySimulatorObjectType.MOVABLE_ENTITY.value
    ):
        # Is the target `at_entity` a movable or goal entity?
        targ_idx = cast(
            int,
            sim_info.search_for_entity(at_entity),
        )
        idxs, pos_targs = sim_info.sim.get_targets()
        targ_pos = pos_targs[list(idxs).index(targ_idx)]

        dist = float(np.linalg.norm(entity_pos - targ_pos))
        return dist < dist_thresh
    elif sim_info.check_type_matches(
        at_entity, FindingDorySimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
    ):
        receptacle = sim_info.search_for_entity(at_entity)
        if isinstance(receptacle, TriangleMeshReceptacle):
            object_aabb = sim_info.get_entity_bounds(obj)
            object_corners = [
                object_aabb.back_bottom_left,
                object_aabb.back_bottom_right,
                object_aabb.front_bottom_left,
                object_aabb.front_bottom_right,
                object_aabb.back_top_left,
                object_aabb.back_top_right,
                object_aabb.front_top_left,
                object_aabb.front_top_right,
            ]
            are_points_on_surface = [
                receptacle.check_if_point_on_surface(
                    sim_info.sim, corner, threshold=dist_thresh
                )
                for corner in object_corners
            ]
            return bool(sum(are_points_on_surface)[0] >= 4)
        else:
            recep = cast(mn.Range3D, receptacle)
            return recep.contains(entity_pos)
    else:
        raise ValueError(
            f"Got unexpected combination of {obj} and {at_entity}"
        )


def do_entity_lists_match(
    to_set: List[FindingDoryPddlEntity], set_value: List[FindingDoryPddlEntity]
) -> bool:
    """
    Returns if the two predicate lists match in count and argument types.
    """

    if len(to_set) != len(set_value):
        return False
    # Check types are compatible
    return all(
        set_arg.expr_type.is_subtype_of(arg.expr_type)
        for arg, set_arg in zip(to_set, set_value)
    )


def ensure_entity_lists_match(
    to_set: List[FindingDoryPddlEntity], set_value: List[FindingDoryPddlEntity]
) -> None:
    """
    Checks if the two predicate lists match in count and argument types. If
    they don't match, an exception is thrown.
    """

    if len(to_set) != len(set_value):
        raise ValueError(
            f"Set arg values are unequal size {to_set} vs {set_value}"
        )
    # Check types are compatible
    for arg, set_arg in zip(to_set, set_value):
        if not set_arg.expr_type.is_subtype_of(arg.expr_type):
            raise ValueError(
                f"Arg type is incompatible \n{to_set}\n vs \n{set_value}"
            )
            
def _log_metric(sim_info, at_entity, key, value):
    """
    Update the metric for the entity.
    If sim_info.env._pddl_verification_obs is not None, update the high-level metrics;
    otherwise, update the low-level metrics.
    """
    metrics = (sim_info.env._hl_pddl_per_assign_metrics 
               if sim_info.env._pddl_verification_obs is not None 
               else sim_info.env._ll_pddl_per_assign_metrics)
    if at_entity.name in metrics:
        metrics[at_entity.name][key] = value
