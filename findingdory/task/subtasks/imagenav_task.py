#!/usr/bin/env python3

import random
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
import time

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.articulated_agents.robots.stretch_robot import StretchJointStates, StretchRobot
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.utils import cartesian_to_polar
from habitat.tasks.rearrange.utils import set_agent_base_via_obj_trans, rearrange_logger
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions


@dataclass
class NavToInfo:
    """
    :property nav_goal_pos: Where the articulated_agent should navigate to. This is likely
    on a receptacle and not a navigable position.
    """

    nav_goal_pos: np.ndarray
    nav_goal_rot: float
    articulated_agent_start_pos: np.ndarray
    articulated_agent_start_angle: float
    start_hold_obj_idx: Optional[int]


@registry.register_task(name="ImageNavTask-v0")
class ImageNavTask(RearrangeTask):
    """
    :property _nav_to_info: Information about the next skill we are navigating to.
    """

    _nav_to_info: Optional[NavToInfo]

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=False,
            **kwargs,
        )
        self.force_obj_to_idx = None
        self.force_recep_to_name = None

        # Set config options
        self._object_in_hand_sample_prob = (
            self._config.object_in_hand_sample_prob
        )
        self._min_start_distance = self._config.min_start_distance

        self._nav_to_info = None
        self._robot_start_position = None
        self._robot_start_rotation = None

        self._camera_tilt = config.camera_tilt

    @property
    def nav_goal_pos(self):
        return self._nav_to_info.nav_goal_pos
    
    @property
    def nav_goal_rot(self):
        return self._nav_to_info.nav_goal_rot

    @property
    def _get_agent_state(self, agent_id=0):
        return self._sim.get_agent_data(agent_id).articulated_agent

    @property
    def should_end(self) -> bool:
        does_want_terminate = False
        if "stop" in self.actions:
            does_want_terminate = self.is_stop_called
        else:
            does_want_terminate = self.actions["rearrange_stop"].does_want_terminate
        return (
            self._should_end
            or does_want_terminate
        )

    @should_end.setter
    def should_end(self, new_val: bool):
        self._should_end = new_val

    def set_args(self, obj, **kwargs):
        self.force_obj_to_idx = obj
        self.force_kwargs = kwargs

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the articulated_agent hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_to_pos(self, episode):
        goal_position = getattr(episode, "goal_position", [])
        goal_rotation = getattr(episode, "goal_rotation", [])
        if len(goal_position) > 0 and len(goal_rotation) > 0:
            nav_to_pos = goal_position
            nav_to_rot = goal_rotation
        else:
            nav_to_pos = self._sim.pathfinder.get_random_navigable_point(
                island_index=self._sim._largest_indoor_island_idx
            )
            nav_to_pos = self._sim.safe_snap_point(nav_to_pos)

            nav_to_rot = np.random.uniform(-np.pi, np.pi)
        
        return nav_to_pos, nav_to_rot

    def _generate_nav_start_goal(
        self,
        episode,
        nav_to_pos,
        nav_to_rot,
        start_hold_obj_idx=None,
    ) -> NavToInfo:
        """
        Returns the starting information for a navigate to object task.
        """
        # Hack: If sum of abs is zero, it means the start position is not set, so we need to set it
        if np.sum(np.abs(episode.start_position)) != 0.0:
            start_quat = quaternion_from_coeff(
                episode.start_rotation
            )
            direction_vector = np.array([0, 0, -1])
            heading_vector = quaternion_rotate_vector(start_quat, direction_vector)
            articulated_agent_pos = np.array(episode.start_position)
            articulated_agent_angle = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        else:
            def filter_func(start_pos, _):
                if len(nav_to_pos.shape) == 1:
                    goals = np.expand_dims(nav_to_pos, axis=0)
                else:
                    goals = nav_to_pos
                distance = self._sim.geodesic_distance(start_pos, goals, episode)
                return distance != np.inf and distance > self._min_start_distance

            (
                articulated_agent_pos,
                articulated_agent_angle,
            ) = self._sim.set_articulated_agent_base_to_random_point(
                filter_func=filter_func
            )

        return NavToInfo(
            nav_goal_pos=nav_to_pos,
            nav_goal_rot=nav_to_rot,
            articulated_agent_start_pos=articulated_agent_pos,
            articulated_agent_start_angle=articulated_agent_angle,
            start_hold_obj_idx=start_hold_obj_idx,
        )

    def reset(self, episode: Episode, fetch_observations: bool = True):

        # Generate a new random seed based on the current time
        new_random_seed = int(time.time() * 1000) % (2**32 - 1)
        
        # Set the random seed for sampling navigable points
        self._sim.pathfinder.seed(new_random_seed)

        super().reset(episode, fetch_observations=False)

        # in the case of Stretch, force the agent to look down and retract arm with the gripper pointing downwards
        if isinstance(self._sim.articulated_agent, StretchRobot):
            joints = StretchJointStates.NAVIGATION.copy()
            # set camera tilt, which is the the last joint of the arm
            joints[-1] = self._camera_tilt
            self._sim.articulated_agent.arm_joint_pos = joints
            self._sim.articulated_agent.arm_motor_pos = joints

        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        if (
            self.force_obj_to_idx is None
            and random.random() < self._config.object_in_hand_sample_prob
        ):
            start_hold_obj_idx = self._generate_snap_to_obj()

        nav_to_pos, nav_to_rot = self._generate_nav_to_pos(episode)

        self._nav_to_info = self._generate_nav_start_goal(
            episode,
            nav_to_pos, 
            nav_to_rot,
            start_hold_obj_idx=start_hold_obj_idx
        )

        set_agent_base_via_obj_trans(
            self._nav_to_info.articulated_agent_start_pos,
            self._nav_to_info.articulated_agent_start_angle,
            self._sim.articulated_agent
        )
        
        # Required to be in this format for the distance to goal computation
        episode.nav_goal_pos = [NavigationGoal(position=self._nav_to_info.nav_goal_pos)]
        episode.nav_goal_rot = [self._nav_to_info.nav_goal_rot]

        self._robot_start_position = self._sim.articulated_agent.sim_obj.translation
        start_quat = self._sim.articulated_agent.sim_obj.rotation
        self._robot_start_rotation = np.array(
            [
                start_quat.vector.x,
                start_quat.vector.y,
                start_quat.vector.z,
                start_quat.scalar,
            ]
        )

        if self._nav_to_info.start_hold_obj_idx is not None:
            if self._sim.grasp_mgr.is_grasped:
                raise ValueError(
                    f"Attempting to grasp {self._nav_to_info.start_hold_obj_idx} even though object is already grasped"
                )
            rearrange_logger.debug(
                f"Forcing to grasp object {self._nav_to_info.start_hold_obj_idx}"
            )
            self._sim.grasp_mgr.snap_to_obj(
                self._nav_to_info.start_hold_obj_idx, force=True
            )

        if self._sim.habitat_config.debug_render:
            rom = self._sim.get_rigid_object_manager()
            # Visualize the position the agent is navigating to.
            self._sim.viz_ids["nav_targ_pos"] = self._sim.visualize_position(
                #self._nav_to_info.nav_goal_rot,
                np.array(rom.get_object_by_id(int(episode.candidate_objects[0].object_id)).translation),
                self._sim.viz_ids["nav_targ_pos"],
                r=0.2,
            )
 
        if fetch_observations:
            self._sim.maybe_update_articulated_agent()
            return self._get_observations(episode)
        else:
            return None