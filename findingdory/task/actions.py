#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import Optional
from collections import defaultdict

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.articulated_agents.robots.stretch_robot import (
    StretchJointStates,
    StretchRobot,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)
from habitat.tasks.rearrange.actions.grip_actions import MagicGraspAction

from habitat.core.logging import logger
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import (
    is_accessible,
)
from habitat.datasets.rearrange.samplers.receptacle import (
    OnTopOfReceptacle,
    Receptacle,
    TriangleMeshReceptacle,
)
from habitat.tasks.rearrange.utils import get_aabb
from habitat.core.embodied_task import SimulatorTaskAction


if not HabitatSimActions.has_action("navigation_mode"):
    HabitatSimActions.extend_action_space("navigation_mode")


@registry.register_task_action
class NavigationModeAction(ArticulatedAgentAction):
    """
    This action reverts the effect of the ManipulationModeAction to bring back the robot to navigation mode.
    """

    def __init__(self, *args, config, **kwargs):
        self._threshold = config.threshold
        super().__init__(self, *args, config=config, **kwargs)

    def step(self, task, *args, is_last_action, **kwargs):
        manip_mode = kwargs.get("navigation_mode", [-1.0])
        if manip_mode[0] > self._threshold and task._in_manip_mode:
            if isinstance(self._sim.articulated_agent, StretchRobot):
                # Turn the head to face the arm
                task._in_manip_mode = False
                self._sim.articulated_agent.arm_motor_pos = (
                    StretchJointStates.NAVIGATION
                )
                self._sim.articulated_agent.arm_joint_pos = (
                    StretchJointStates.NAVIGATION
                )
                # now turn the robot's base left by -90 degrees
                obj_trans = self.cur_articulated_agent.sim_obj.transformation
                turn_angle = -np.pi / 2  # Turn left by -90 degrees
                rot_quat = mn.Quaternion(
                    mn.Vector3(0, np.sin(turn_angle / 2), 0),
                    np.cos(turn_angle / 2),
                )
                # Get the target rotation
                target_rot = rot_quat.to_matrix() @ obj_trans.rotation()
                target_trans = mn.Matrix4.from_(
                    target_rot,
                    obj_trans.translation,
                )
                self.cur_articulated_agent.sim_obj.transformation = target_trans
                if self.cur_grasp_mgr.snap_idx is not None:
                    # Holding onto an object, also kinematically update the object.
                    self.cur_grasp_mgr.update_object_to_grasp()

        if is_last_action:
            return self._sim.step(HabitatSimActions.navigation_mode)
        else:
            return {}


@registry.register_task_action
class PointGraspAction(MagicGraspAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.min_dist, self.max_dist = config.gaze_distance_range
        self._translation_up_offset = config.translation_up_offset
        self._object_ids_start = self._sim.habitat_config.object_ids_start
        self.auto_grasp = config.auto_grasp

    @property
    def action_space(self):
        if self.auto_grasp:
            return None
        else:
            return spaces.Box(shape=(1,), high=1.0, low=-1.0)

    def _determine_object_at_pixel(self, pixels):
        """Determine if an object is at the center of the frame and in range"""
        if not isinstance(self.cur_articulated_agent, StretchRobot):
            raise NotImplementedError(
                "This robot does not have PointGraspAction."
            )

        cam_pos = (
            self._sim.agents[0]
            .get_state()
            .sensor_states["head_rgb"]
            .position
        )

        panoptic_img = self._sim._sensor_suite.get_observations(
            self._sim.get_sensor_observations()
        )["head_panoptic"]

        height, width = panoptic_img.shape[:2]

        assert pixels[0] >= 0 and pixels[0] < height, \
            f"PointGraspAction: Invalid height for pixel: {pixels[0]}, image height is: {height}"
        assert pixels[1] >= 0 and pixels[1] < width, \
            f"PointGraspAction: Invalid width for pixel: {pixels[1]}, image width is: {width}"

        # Note that panoptic_img is a 3D array
        center_obj_id = (
            panoptic_img[pixels[0], pixels[1], 0]
            - self._object_ids_start
        )

        rom = self._sim.get_rigid_object_manager()
        if center_obj_id in self._sim.scene_obj_ids:
            obj_pos = rom.get_object_by_id(center_obj_id).translation

            # Skip if not in distance range
            dist = np.linalg.norm(obj_pos - cam_pos)
            if dist < self.min_dist or dist > self.max_dist:
                return None

            return center_obj_id

        return None

    def _grasp(self, pixels):
        # Check if there is an object at the given pixel
        selected_obj_idx = self._determine_object_at_pixel(
            pixels
        )

        # If there is nothing to grasp, then we return
        if selected_obj_idx is None:
            return

        keep_T = mn.Matrix4.translation(mn.Vector3(0.1, 0.0, 0.0))

        self.cur_grasp_mgr.snap_to_obj(
            selected_obj_idx,
            force=True,
            rel_pos=mn.Vector3(0.1, 0.0, 0.0),
            keep_T=keep_T,
        )
        return

    def _determine_receptacle_at_pixel(self, pixels):
        """Determine if an object is at the center of the frame and in range"""
        if not isinstance(self.cur_articulated_agent, StretchRobot):
            raise NotImplementedError(
                "This robot does not have PointGraspAction."
            )

        cam_pos = (
            self._sim.agents[0]
            .get_state()
            .sensor_states["head_rgb"]
            .position
        )

        panoptic_img = self._sim._sensor_suite.get_observations(
            self._sim.get_sensor_observations()
        )["head_panoptic"]

        height, width = panoptic_img.shape[:2]

        assert pixels[0] >= 0 and pixels[0] < height, \
            f"PointGraspAction: Invalid height for pixel: {pixels[0]}, image height is: {height}"
        assert pixels[1] >= 0 and pixels[1] < width, \
            f"PointGraspAction: Invalid width for pixel: {pixels[1]}, image width is: {width}"

        # Note that panoptic_img is a 3D array
        center_obj_id = (
            panoptic_img[pixels[0], pixels[1], 0]
            - self._object_ids_start
        )

        rom = self._sim.get_rigid_object_manager()
        if center_obj_id in self._sim.scene_recep_ids:
            recep_obj = rom.get_object_by_id(center_obj_id)
            all_receptacles = self._sim._obj_handle_to_receps[recep_obj.handle]

            reachable_receptacles = []
            for receptacle in all_receptacles:
                recep_pos = receptacle.get_surface_center(self._sim)
                # Skip if not in distance range
                dist = np.linalg.norm(recep_pos - cam_pos)
                if dist >= self.min_dist or dist <= self.max_dist:
                    reachable_receptacles.append(receptacle)

            return reachable_receptacles

        return reachable_receptacles

    def _ungrasp(self, pixels):
        # Check if there is a receptacle object at the given pixel
        reachable_receptacles = self._determine_receptacle_at_pixel(
            pixels
        )

        # If there is nothing to put the object on, then we return
        if len(reachable_receptacles) == 0:
            return

        rom = self._sim.get_rigid_object_manager()
        pick_obj = rom.get_object_by_id(self.cur_grasp_mgr.snap_idx)

        for receptacle in reachable_receptacles:
            if self.sample_placement(
                self._sim,
                pick_obj,
                receptacle,
                snap_down=True,
            ):
                break

        return

    def sample_placement(
        self,
        sim: habitat_sim.Simulator,
        object: str,
        receptacle: Receptacle,
        snap_down: bool = False,
        max_placement_attempts = 10,
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        """
        Attempt to sample a valid placement of the object in/on a receptacle given an object handle and receptacle information.

        :param sim: The active Simulator instance.
        :param object_handle: The handle of the object template for instantiation and attempted placement.
        :param receptacle: The Receptacle instance on which to sample a placement position.
        :param snap_down: Whether or not to use the snap_down utility to place the object.

        :return: The True if the object was successfully placed, False otherwise.
        """
        num_placement_tries = 0

        # Note: we cache the largest island ID to reject samples which are primarily accessible from disconnected navmesh regions.
        # This assumption limits sampling to the largest navigable component of any scene.
        current_agent_island_id = sim.pathfinder.get_island(
            sim.agents[0].get_state().position
        )

        cam_pos = (
            self._sim.agents[0]
            .get_state()
            .sensor_states["head_rgb"]
            .position
        )

        rec_up_global = (
            receptacle.get_global_transform(sim)
            .transform_vector(receptacle.up)
            .normalized()
        )

        # fail early if it's impossible to place the object
        cumulative_bb = object.root_scene_node.cumulative_bb
        new_object_base_area = (
            cumulative_bb.size_x() * cumulative_bb.size_z()
        )
        if new_object_base_area > receptacle.total_area:
            logger.info(
                f"Failed to sample placement. {object.handle} was too large to place on {receptacle.name}"
            )
            return False

        while num_placement_tries < max_placement_attempts:
            num_placement_tries += 1

            # sample the object location
            target_object_position = (
                receptacle.sample_uniform_global(
                    sim, defaultdict(lambda: 1.0)[receptacle.name] #self.sample_region_ratio[receptacle.name]
                )
                + self._translation_up_offset * rec_up_global
            )

            # check distance from the camera center
            dist = np.linalg.norm(target_object_position - cam_pos)
            if dist < self.min_dist or dist > self.max_dist:
                logger.info(
                    f"Failed to sample placement. {object.handle} was too far from the camera center ({dist})"
                )
                continue

            # try to place the object
            object.translation = target_object_position
            # rotate the object around the gravity direction
            rot = random.uniform(0, math.pi * 2.0)
            object.rotation = mn.Quaternion.rotation(
                mn.Rad(rot), mn.Vector3.y_axis()
            )

            if isinstance(receptacle, TriangleMeshReceptacle):
                object.translation = object.translation + mn.Vector3(
                    0, 0.05, 0
                )
            if isinstance(receptacle, OnTopOfReceptacle):
                snap_down = False

            if snap_down:
                support_object_ids = [habitat_sim.stage_id]
                # add support object ids for non-stage receptacles
                if receptacle.is_parent_object_articulated:
                    ao_instance = sim.get_articulated_object_manager().get_object_by_handle(
                        receptacle.parent_object_handle
                    )
                    for (
                        object_id,
                        link_ix,
                    ) in ao_instance.link_object_ids.items():
                        if receptacle.parent_link == link_ix:
                            support_object_ids = [
                                object_id,
                                ao_instance.object_id,
                            ]
                            break
                elif receptacle.parent_object_handle is not None:
                    support_object_ids = [
                        sim.get_rigid_object_manager()
                        .get_object_by_handle(receptacle.parent_object_handle)
                        .object_id
                    ]
                snap_success = sutils.snap_down(
                    sim,
                    object,
                    support_object_ids,
                )
                if snap_success:
                    logger.info(
                        f"Successfully sampled (snapped) object placement in {num_placement_tries} tries."
                    )
                    if not is_accessible(
                        sim=sim,
                        point=object.translation,
                        height=sim.agents[0].agent_config.height,
                        nav_to_min_distance=2.0,
                        nav_island=current_agent_island_id,
                        target_object_id=object.object_id,
                    ):
                        logger.warning(
                            f"Failed to navigate to {object.handle} on {receptacle.name} in {num_placement_tries} tries."
                        )
                        continue
                    object_aabb = get_aabb(
                        object.object_id, sim, transformed=True
                    )
                    object_corners = [
                        object_aabb.back_bottom_left,
                        object_aabb.back_bottom_right,
                        object_aabb.front_bottom_left,
                        object_aabb.front_bottom_right,
                    ]
                    if not all(
                        receptacle.check_if_point_on_surface(
                            sim, corner, threshold=0.05
                        )
                        for corner in object_corners
                    ):
                        logger.warning(
                            f"Failed to place {object.handle} within bounds of {receptacle.name} in {num_placement_tries} tries."
                        )
                        continue

                    object.angular_velocity = mn.Vector3.zero_init()
                    object.linear_velocity = mn.Vector3.zero_init()
                    sim.internal_step(-1)
                    object.angular_velocity = mn.Vector3.zero_init()
                    object.linear_velocity = mn.Vector3.zero_init()

                    obs = sim._sensor_suite.get_observations(sim.get_sensor_observations())

                    panoptic = obs["head_panoptic"]
                    unique_ids_cur_frame = set(np.unique(panoptic))
                    put_obj_in_panoptic = (object.object_id + self._object_ids_start) in unique_ids_cur_frame

                    if put_obj_in_panoptic:
                        self.cur_grasp_mgr.desnap(True)
                        # self.cur_grasp_mgr.update_object_to_grasp()
                        sim.internal_step(-1)

                        return True


        logger.info(
            f"Failed to sample {object.handle} placement on {receptacle.unique_name} in {max_placement_attempts} tries."
        )

        return False


    def step(self, grip_action, should_step=True, *args, **kwargs):
        grip = grip_action[0]
        pixels = np.array(grip_action[1:]).astype(int)
        if self.auto_grasp and not self.cur_grasp_mgr.is_grasped:
            self._grasp(pixels)
            return

        if grip is None:
            return

        if grip > 0 and not self.cur_grasp_mgr.is_grasped:
            self._grasp(pixels)
        elif grip < 0 and self.cur_grasp_mgr.is_grasped:
            self._ungrasp(pixels)


@registry.register_task_action
class PddlIntermediateStopAction(SimulatorTaskAction):
    '''
    Used for tasks (ordered/unordered) in which the agent needs to invoke this action to "stop" at each subgoal in the sequence
    '''
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

    def step(self, task, *args, **kwargs):
        task.pddl.sim_info.does_want_intermediate_stop = True