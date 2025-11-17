# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import copy

from findingdory.policies.heuristic import place_utils
from findingdory.policies.heuristic.heuristic_place_policy import HeuristicPlacePolicy

RETRACTED_ARM_APPROX_LENGTH = 0.15
STRETCH_STANDOFF_DISTANCE = 0.6
STRETCH_GRASP_DISTANCE = 0.8
STRETCH_EXTENSION_OFFSET = 0.0
ANGLE_ADJUSTMENT = -0.01
HARDCODED_ARM_EXTENSION_OFFSET = 4.0


camera_pose_transform = lambda habitat_camera_pose: place_utils.convert_pose_to_real_world_axis(
    habitat_camera_pose
)


class HeuristicPickPolicy(HeuristicPlacePolicy):
    """
    Heuristic policy for picking objects.
    Mainly used for visualizing the agent's arm reaching the object.
    First determines the pick point using object point cloud, then turns to orient towards the object, then moves the arm to the pick point and snaps the object.
    """

    def __init__(
        self, config, device, debug_visualize_xyz: bool = False, verbose: bool = False
    ):
        self.timestep = 0
        super().__init__(config, device, debug_visualize_xyz)
        
        self.verbose = verbose
        self.erosion_kernel = np.ones((2, 2), np.uint8)
        self.sub_skill_num_steps = 10

    def get_object_pick_point(
        self,
        obs, #: Observations
        pick_goal_id: int,
        vis_inputs: Optional[Dict] = None,
        arm_reachability_check: bool = False,
        visualize: bool = True,
    ):

        goal_object_mask = (
            (obs['all_object_segmentation']
            == pick_goal_id) * place_utils.valid_depth_mask(obs["head_depth"])
        ).astype(np.uint8)

        goal_object_mask = goal_object_mask.squeeze(-1)

        # Get dilated, then eroded mask (for cleanliness)
        goal_object_mask = place_utils.smooth_mask(
            goal_object_mask, self.erosion_kernel, num_iterations=2
        )[1]
        # Convert to booleans
        goal_object_mask = goal_object_mask.astype(bool)

        # if visualize:
        #     cv2.imwrite(f"{self.object_name}_semantic.png", goal_object_mask * 255)

        if not goal_object_mask.any():
            if self.verbose:
                print("Goal object not visible!")
            return None
        else:
            pcd_base_coords = self.get_target_point_cloud_base_coords(
                obs, goal_object_mask, arm_reachability_check
            )

            pcd_base_coords = pcd_base_coords.cpu().numpy()
            flat_array = pcd_base_coords.reshape(-1, 3)
            index = flat_array[:, 2].argmax()
            best_voxel = pcd_base_coords[
                index // pcd_base_coords.shape[-2], index % pcd_base_coords.shape[-2]
            ]

            if index == 0:
                # We recompute another best_voxel based on where the z-axis value is non-zero
                non_zero_coords = np.where(flat_array[:,2] != 0)
                index_nonzero = flat_array[non_zero_coords, 2].argmax()
                index = non_zero_coords[0][index_nonzero]
                best_voxel = pcd_base_coords[
                    index // pcd_base_coords.shape[-2], index % pcd_base_coords.shape[-2]
                ]        

            rgb_vis = obs["head_rgb"].copy()
            rgb_vis_tmp = cv2.circle(
                rgb_vis,
                (index % pcd_base_coords.shape[-2], index // pcd_base_coords.shape[-2]),
                4,
                (0, 255, 0),
                thickness=2,
            )
            if vis_inputs is not None:
                vis_inputs["semantic_frame"][..., :3] = rgb_vis_tmp
            
            return best_voxel, vis_inputs

    def generate_plan(
        self,
        obs, # : Observations,
        pick_goal_id: int,
        vis_inputs: Optional[Dict] = None
    ) -> None:
        """Hardcode the following plan:
        1. Find a grasp point.
        2. Turn to orient towards the object.
        3. Raise the arm.
        4. Move the arm to the object.
        5. Snap the object.
        6. Raise the arm.
        7. Close gripper."""
        self.du_scale = 1
        # self.object_name = obs.task_observations["goal_name"].split(" ")[1]

        self.sub_timestep = 0
        self.arm_joint_to_extend = 0
        self.prev_arm_ext = 0
        self._lift_up_actions = []
        self._arm_extension_actions = []

        found = self.get_object_pick_point(obs, pick_goal_id, vis_inputs)
        if found is None:
            # if not found, we retry after tilt is lowered. Otherwise, we just snap the object
            self.orient_turn_angle = 0
            fwd_dist = 0
            self.fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
            self.t_turn_to_orient = -1
            self.t_move_to_reach = -1
            self.t_manip_mode = 0
            self.t_turn_to_orient_post_manip_mode = 1
            self.t_relative_back = np.inf
            self.t_relative_standoff = np.inf
            self.t_relative_grasp = np.inf
            self.t_relative_snap_object = 0
            self.t_start_pick = np.inf
        else:
            self.grasp_voxel, vis_inputs = found

            center_voxel_trans = np.array(
                [
                    self.grasp_voxel[1],
                    self.grasp_voxel[2],
                    self.grasp_voxel[0],
                ]
            )

            delta_heading = place_utils.get_angle_to_pos(center_voxel_trans)
            self.orient_turn_angle = delta_heading

            self.nav_orient_num_turns = np.rad2deg(abs(self.orient_turn_angle)) // self.config.ENVIRONMENT.turn_angle
            self.nav_orient_num_turns = np.clip(self.nav_orient_num_turns,0,2)      # NOTE: We clip this to max number of allowed turns for the robot as large number of rotation may cause the object to move out of sight
            self.orient_turn_direction = np.sign(self.orient_turn_angle)

            # This gets the Y-coordiante of the center voxel
            # Base link to retracted arm - this is about 15 cm
            fwd_dist = self.grasp_voxel[1]
            self.fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
            self.nav_forward_num_turns = fwd_dist // self.config.ENVIRONMENT.forward

            # self.t_turn_to_orient = 0
            self.t_move_to_reach = self.timestep + self.nav_orient_num_turns
            self.t_manip_mode = self.t_move_to_reach + self.nav_forward_num_turns
            self.t_turn_to_orient_post_manip_mode = self.t_manip_mode + 1
            
            # timesteps relative to the time when orientation finishes
            self.t_relative_back = 0
            self.t_relative_standoff = 1
            self.t_relative_grasp = 2
            self.t_relative_snap_object = 3
            self.t_start_pick = np.inf
            if self.verbose:
                print("-" * 20)

    def get_action(
        self,
        obs, #: Observations
        pick_goal_id,
        vis_inputs: Optional[Dict] = None
    ) -> Tuple[
        Union[
            place_utils.ContinuousFullBodyAction,
            place_utils.ContinuousNavigationAction,
            place_utils.DiscreteNavigationAction,
        ],
        Dict,
    ]:
        """Get the action to execute at the current timestep using the plan generated in generate_plan.
        Before actual picking starts (i.e. before t_start_pick), the agent turns and moves to orient towards the pick point.
        Recalibrates the pick point after switching to manipulation mode.
        After t_start_pick, the agent moves the arm to the pick point and snaps the object.
        """
        action = None
        
        if self.sub_timestep == 0:

            if self.timestep < self.t_move_to_reach:
                # at first turn to face the object and move forward
                self.orient_turn_angle = 0
                self.action_desc = "init orient"
                if self.orient_turn_direction == -1:
                    action = place_utils.DiscreteNavigationAction.TURN_RIGHT
                elif self.orient_turn_direction == +1:
                    action = place_utils.DiscreteNavigationAction.TURN_LEFT
                
            elif self.timestep < self.t_manip_mode:
                action = place_utils.DiscreteNavigationAction.MOVE_FORWARD
                self.action_desc = "move forward"

            elif self.timestep == self.t_manip_mode:
                action = place_utils.DiscreteNavigationAction.MANIPULATION_MODE
                self._in_manip_mode = True
                self.action_desc = "goto manip orig"

            elif self.timestep == self.t_turn_to_orient_post_manip_mode and self._in_manip_mode:
                grasp_voxel = self.get_object_pick_point(obs, pick_goal_id, vis_inputs)
                # recalibrate the grasp voxel (since the agent may have moved a bit and is looking down)
                if grasp_voxel is not None:
                    self.grasp_voxel, vis_inputs = grasp_voxel
                    center_voxel_trans = np.array(
                        [
                            self.grasp_voxel[1],
                            self.grasp_voxel[2],
                            self.grasp_voxel[0],
                        ]
                    )
                    self.orient_turn_angle = (
                        place_utils.get_angle_to_pos(center_voxel_trans) + ANGLE_ADJUSTMENT
                    )
                
                # If object to be picked is still not visible, then we will try to re-navigate to another viewpoint
                else:
                    action = place_utils.DiscreteNavigationAction.EMPTY_ACTION
                    self.action_desc = "recovery rotate"
                    return action, vis_inputs, self.action_desc
                
                assert self.orient_turn_angle != 0, "Orient angle equal zero. Pick goal is not visible !"
                
                self.initial_orient_num_turns = np.rad2deg(abs(self.orient_turn_angle)) // self.config.ENVIRONMENT.turn_angle
                self.orient_turn_direction = np.sign(self.orient_turn_angle)
                
                self.orient_turn_angle = 0
                self.t_start_pick = self.timestep + self.initial_orient_num_turns
                self.t_relative_back = 0
                self.t_relative_standoff = 1
                self.t_relative_grasp = 2
                self.t_relative_snap_object = 3
                self.t_relative_retract = 4
                self.t_relative_lower = 5
                    
            if action is not None:
                self.timestep += 1
                return action, vis_inputs, self.action_desc

            # This if condition tries to perform the initial orientation until the pick action has to be started
            elif self.timestep < self.t_start_pick:

                self.action_desc = "final orient"

                if self._in_manip_mode:     # We need to exit manip_mode as the right/left turns are defined wrt navigation_mode
                    action = place_utils.DiscreteNavigationAction.NAVIGATION_MODE
                    self.timestep -= 1
                    self._in_manip_mode = False
                    self.action_desc = "goto nav"

                elif self.orient_turn_direction == -1:
                    action = place_utils.DiscreteNavigationAction.TURN_RIGHT
                elif self.orient_turn_direction == +1:
                    action = place_utils.DiscreteNavigationAction.TURN_LEFT
                if self.verbose:
                    print("[Placement] Turning to orient towards object")
            
            elif not self._in_manip_mode:
                action = place_utils.DiscreteNavigationAction.MANIPULATION_MODE
                self._in_manip_mode = True
                self.timestep -= 1
                self.action_desc = "goto manip"

            elif self.timestep == self.t_start_pick + self.t_relative_back:
                # final recalibration of the grasp voxel
                grasp_voxel = self.get_object_pick_point(obs, pick_goal_id, vis_inputs)
                if grasp_voxel is not None:
                    self.grasp_voxel, vis_inputs = grasp_voxel
                standoff_lift = np.min(
                    [self.max_arm_height, self.grasp_voxel[2] + STRETCH_STANDOFF_DISTANCE]
                )
                current_arm_lift = obs["joint"][4]
                delta_arm_lift = standoff_lift - current_arm_lift
                joints = np.array([0] * 4 + [delta_arm_lift] + [0] * 5)
                action = place_utils.ContinuousFullBodyAction(joints)
                self._lift_up_actions.append(copy.deepcopy(action))

                self.action_desc = "lift to calibrate"

            elif self.timestep == self.t_start_pick + self.t_relative_standoff:
                placement_extension = self.grasp_voxel[1]
                # current_arm_ext = obs["joint"][:4].sum()
                # delta_arm_ext = (
                #     target_extension - current_arm_ext - STRETCH_EXTENSION_OFFSET
                #     + HARDCODED_ARM_EXTENSION_OFFSET
                # )

                current_arm_ext = obs["joint"][:4].sum()

                self.delta_arm_lift_per_step = 0.
                self.delta_arm_ext_per_step = (
                    placement_extension
                    - STRETCH_STANDOFF_DISTANCE
                    - RETRACTED_ARM_APPROX_LENGTH
                    - current_arm_ext
                    + HARDCODED_ARM_EXTENSION_OFFSET
                ) / self.sub_skill_num_steps
                self.delta_gripper_yaw_per_step = 0.

                # joints = np.array([delta_arm_ext] + [0] * 9)
                # action = place_utils.ContinuousFullBodyAction(joints)

                action = self._move_arm_into_position(obs)
                self._arm_extension_actions.append(copy.deepcopy(action))
                self.action_desc = "extend arm"

                return action, vis_inputs, self.action_desc

            elif self.timestep == self.t_start_pick + self.t_relative_grasp:
                grasp_lift = np.min([self.max_arm_height, self.grasp_voxel[2] + STRETCH_GRASP_DISTANCE])
                current_arm_lift = obs["joint"][4]
                delta_arm_lift = grasp_lift - current_arm_lift
                joints = np.array([0] * 4 + [delta_arm_lift] + [0] * 5)
                action = place_utils.ContinuousFullBodyAction(joints)

                self._lift_up_actions.append(copy.deepcopy(action))

                self.action_desc = "lift to grasp"

            elif self.timestep == self.t_start_pick + self.t_relative_snap_object:
                # snap to pick the object
                if self.verbose:
                    print("[Pick] Snapping object")
                action = place_utils.DiscreteNavigationAction.SNAP_OBJECT
                self.action_desc = "snap obj"

            elif self.timestep == self.t_start_pick + self.t_relative_retract:
                action = self._retract(obs)
                self.action_desc = "retract arm"
                return action, vis_inputs, self.action_desc

            elif self.timestep == self.t_start_pick + self.t_relative_lower:
                action = self._lift_down(obs)
                self.action_desc = "goto low"
                return action, vis_inputs, self.action_desc

            else:
                if self.verbose:
                    print("[Pick] Stopping")
                self.action_desc = "stopping"
                action = place_utils.DiscreteNavigationAction.STOP

        else:
            if self.verbose:
                print("Executing pick skill substep...")

            if self.timestep == self.t_start_pick + self.t_relative_standoff:
                action = self._move_arm_into_position(obs)
                self._arm_extension_actions.append(copy.deepcopy(action))
            elif self.timestep == self.t_start_pick + self.t_relative_retract:
                action = self._retract(obs)
            elif self.timestep == self.t_start_pick + self.t_relative_lower:
                action = self._lift_down(obs)
            return action, vis_inputs, self.action_desc
        
        self.timestep += 1
        return action, vis_inputs, self.action_desc

    def forward(
        self,
        obs, #: Observations,
        pick_goal_id: int,
        vis_inputs: Optional[Dict] = None
    ):
        self.timestep = self.timestep

        # OVMM specific observation preprocessing
        obs["camera_pose"] = camera_pose_transform(np.asarray(obs["camera_pose"]))
        obs["head_depth"] = place_utils.preprocess_depth(obs["head_depth"])

        if self.timestep == 0:
            self.generate_plan(obs, pick_goal_id, vis_inputs)
        if self.verbose:
            print("-" * 20)
            print("Timestep", self.timestep)
        action, vis_inputs, action_desc = self.get_action(obs, pick_goal_id, vis_inputs)

        if self.verbose:
            print("Timestep", self.timestep)
        return action, vis_inputs, action_desc