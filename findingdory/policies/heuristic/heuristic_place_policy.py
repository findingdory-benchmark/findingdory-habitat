# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh.transformations as tra

from findingdory.policies.heuristic import place_utils
import copy

STRETCH_STANDOFF_DISTANCE = 0.05
RETRACTED_ARM_APPROX_LENGTH = 0.15
# HARDCODED_ARM_EXTENSION_OFFSET = 0.15
HARDCODED_ARM_EXTENSION_OFFSET = 3.0
HARDCODED_YAW_OFFSET = 0.25
MAX_ARM_HEIGHT = 4.0
STRETCH_RECEPTACLE_CLEARANCE = 1.0

# Set the random seed for reproducibility
random.seed(42)

camera_pose_transform = lambda habitat_camera_pose: place_utils.convert_pose_to_real_world_axis(
    habitat_camera_pose
)


class HeuristicPlacePolicy(nn.Module):
    """
    Policy to place object on end receptacle using depth and point-cloud-based heuristics. Objects will be placed nearby, on top of the surface, based on point cloud data. Requires segmentation to work properly.
    """

    # TODO: read these values from the robot kinematic model
    look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
    max_arm_height = MAX_ARM_HEIGHT

    def __init__(
        self,
        config,
        device,
        placement_drop_distance: float = 0.4,
        debug_visualize_xyz: bool = False,
        verbose: bool = False,
    ):
        """
        Parameters:
            config
            device
            placement_drop_distance: distance from placement point that we add as a margin
            debug_visualize_xyz: whether to display point clouds for debugging
            verbose: whether to print debug statements
        """
        super().__init__()
        self.timestep = 0
        self.config = config
        self.device = device
        self.debug_visualize_xyz = debug_visualize_xyz
        self.erosion_kernel = np.ones((5, 5), np.uint8)
        self.placement_drop_distance = placement_drop_distance
        self.verbose = verbose

        self.sub_skill_num_steps = self.config.sub_skill_num_steps
        self.fall_wait_steps = self.config.fall_wait_steps
        
        self._ready_to_drop_object = False
        self._num_lift_actions_after_desnap = 0
        self._is_obj_dropped = False

    def reset(self):
        self.timestep = 0

    def get_target_point_cloud_base_coords(
        self,
        obs,  #: Observations,
        target_mask: np.ndarray,
        arm_reachability_check: bool = False,
    ):
        """Get point cloud coordinates in base frame"""
        goal_rec_depth = (
            torch.tensor(obs["head_depth"], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .squeeze(-1)
        )

        camera_matrix = place_utils.get_camera_matrix(
            self.config.ENVIRONMENT.frame_width,
            self.config.ENVIRONMENT.frame_height,
            self.config.ENVIRONMENT.hfov,
        )
        # Get object point cloud in camera coordinates
        pcd_camera_coords = place_utils.get_point_cloud_from_z_t(
            goal_rec_depth, camera_matrix, self.device, scale=self.du_scale
        )

        # get point cloud in base coordinates
        camera_pose = np.expand_dims(obs["camera_pose"], 0)
        angles = [tra.euler_from_matrix(p[:3, :3], "rzyx") for p in camera_pose]
        tilt = angles[0][1]  # [0][1]

        # Agent height comes from the environment config
        agent_height = torch.tensor(camera_pose[0, 2, 3], device=self.device)

        # Object point cloud in base coordinates
        pcd_base_coords = place_utils.transform_camera_view_t(
            pcd_camera_coords, agent_height, np.rad2deg(tilt), self.device
        )

        if self.debug_visualize_xyz:
            # Remove invalid points from the mask
            xyz = (
                pcd_base_coords[0]
                .cpu()
                .numpy()
                .reshape(-1, 3)[target_mask.reshape(-1), :]
            )
            from home_robot.utils.point_cloud import show_point_cloud

            rgb = (obs.rgb).reshape(-1, 3) / 255.0
            show_point_cloud(xyz, rgb, orig=np.zeros(3))

        # Whether or not I can extend the robot's arm in order to reach each point
        if arm_reachability_check:
            # filtering out unreachable points based on Y and Z coordinates of voxels (Z is up)
            height_reachable_mask = (pcd_base_coords[0, :, :, 2] < agent_height).to(int)
            height_reachable_mask = torch.stack([height_reachable_mask] * 3, axis=-1)
            pcd_base_coords = pcd_base_coords * height_reachable_mask

            length_reachable_mask = (pcd_base_coords[0, :, :, 1] < agent_height).to(int)
            length_reachable_mask = torch.stack([length_reachable_mask] * 3, axis=-1)
            pcd_base_coords = pcd_base_coords * length_reachable_mask

        non_zero_mask = torch.stack(
            [torch.from_numpy(target_mask).to(self.device)] * 3, axis=-1
        )
        pcd_base_coords = pcd_base_coords * non_zero_mask

        return pcd_base_coords[0]

    def get_receptacle_placement_point(
        self,
        obs,  #: Observations,
        target_receptacle_id: int,
        vis_inputs: Optional[Dict] = None,
        arm_reachability_check: bool = False,
        visualize: bool = False,
    ):
        """
        Compute placement point in 3d space.

        Parameters:
            obs: Observation object; describes what we've seen.
            vis_inputs: optional dict; data used for visualizing outputs
        """
        NUM_POINTS_TO_SAMPLE = 50  # number of points to sample from receptacle point cloud to find best placement point
        SLAB_PADDING = 0.2  # x/y padding around randomly selected points
        SLAB_HEIGHT_THRESHOLD = 0.015  # 1cm above and below, i.e. 2cm overall
        ALPHA_VIS = 0.5

        goal_rec_mask = (
            (obs["receptacle_segmentation"]
            == target_receptacle_id) * place_utils.valid_depth_mask(obs["head_depth"])
        ).astype(np.uint8)

        goal_rec_mask = goal_rec_mask.squeeze(-1)

        # Get dilated, then eroded mask (for cleanliness)
        goal_rec_mask = place_utils.smooth_mask(
            goal_rec_mask, self.erosion_kernel, num_iterations=5
        )[1]
        # Convert to booleans
        goal_rec_mask = goal_rec_mask.astype(bool)

        if visualize:
            cv2.imwrite(f"{self.end_receptacle}_semantic.png", goal_rec_mask * 255)

        if not goal_rec_mask.any():
            if self.verbose:
                print("End receptacle not visible.")
            return None
        else:
            rgb_vis = obs["head_rgb"]
            pcd_base_coords = self.get_target_point_cloud_base_coords(
                obs, goal_rec_mask, arm_reachability_check=arm_reachability_check
            )
            ## randomly sampling NUM_POINTS_TO_SAMPLE of receptacle point cloud – to choose for placement
            reachable_point_cloud = pcd_base_coords.cpu().numpy()
            flat_array = reachable_point_cloud.reshape(-1, 3)

            # find the indices of the non-zero elements in the first two dimensions of the matrix
            nonzero_indices = np.nonzero(flat_array[:, :2].any(axis=1))[0]
            # create a list of tuples containing the non-zero indices in the first two dimensions
            nonzero_tuples = [
                (
                    index // reachable_point_cloud.shape[-2],
                    index % reachable_point_cloud.shape[-2],
                )
                for index in nonzero_indices
            ]

            # Set the random seed for reproducibility when we run the heuristic place policy online
            random.seed(42)

            # select a random subset of the non-zero indices
            random_indices = random.sample(
                nonzero_tuples, min(NUM_POINTS_TO_SAMPLE, len(nonzero_tuples))
            )

            x_values = pcd_base_coords[:, :, 0]
            y_values = pcd_base_coords[:, :, 1]
            z_values = pcd_base_coords[:, :, 2]

            max_surface_points = 0
            # max_height = 0

            max_surface_mask, best_voxel_ind, best_voxel = None, None, None

            ## iterating through all randomly selected voxels and choosing one with most XY neighboring surface area within some height threshold
            for ind in random_indices:
                sampled_voxel = pcd_base_coords[ind[0], ind[1]]
                sampled_voxel_x, sampled_voxel_y, sampled_voxel_z = (
                    sampled_voxel[0],
                    sampled_voxel[1],
                    sampled_voxel[2],
                )

                # sampling plane of pcd voxels around randomly selected voxel (with height tolerance)
                slab_points_mask_x = torch.bitwise_and(
                    (x_values >= sampled_voxel_x - SLAB_PADDING),
                    (x_values <= sampled_voxel_x + SLAB_PADDING),
                )
                slab_points_mask_y = torch.bitwise_and(
                    (y_values >= sampled_voxel_y - SLAB_PADDING),
                    (y_values <= sampled_voxel_y + SLAB_PADDING),
                )
                slab_points_mask_z = torch.bitwise_and(
                    (z_values >= sampled_voxel_z - SLAB_HEIGHT_THRESHOLD),
                    (z_values <= sampled_voxel_z + SLAB_HEIGHT_THRESHOLD),
                )

                slab_points_mask = torch.bitwise_and(
                    slab_points_mask_x, slab_points_mask_y
                ).to(torch.uint8)
                slab_points_mask = torch.bitwise_and(
                    slab_points_mask, slab_points_mask_z
                ).to(torch.uint8)

                # ALTERNATIVE: choose slab with maximum (area x height) product
                # TODO: remove dead code
                # slab_points_mask_stacked = torch.stack(
                #     [
                #         slab_points_mask * 255,
                #         slab_points_mask,
                #         slab_points_mask,
                #     ],
                #     axis=-1,
                # )
                # height = (slab_points_mask_stacked * pcd_base_coords)[..., 2].max()
                # if slab_points_mask.sum() * height >= max_surface_points * max_height:
                if slab_points_mask.sum() >= max_surface_points:
                    max_surface_points = slab_points_mask.sum()
                    max_surface_mask = slab_points_mask
                    # max_height = height
                    best_voxel_ind = ind
                    best_voxel = sampled_voxel

            slab_points_mask_vis = torch.stack(
                [
                    max_surface_mask * 255,
                    max_surface_mask,
                    max_surface_mask,
                ],
                axis=-1,
            )  # for visualization
            rgb_vis_tmp = cv2.addWeighted(
                rgb_vis, ALPHA_VIS, slab_points_mask_vis.cpu().numpy(), 1 - ALPHA_VIS, 0
            )

            rgb_vis_tmp = cv2.circle(
                rgb_vis_tmp,
                (best_voxel_ind[1], best_voxel_ind[0]),
                4,
                (0, 255, 0),
                thickness=2,
            )

            if vis_inputs is not None and vis_inputs["semantic_frame"] is not None:
                vis_inputs["semantic_frame"][..., :3] = rgb_vis_tmp

            # Add placement margin to the best voxel that we chose
            best_voxel[2] += self.placement_drop_distance

            if self.debug_visualize_xyz:
                from home_robot.utils.point_cloud import show_point_cloud

                show_point_cloud(
                    pcd_base_coords.cpu().numpy(),
                    rgb=obs.rgb / 255.0,
                    orig=best_voxel.cpu().numpy(),
                )

            return best_voxel.cpu().numpy(), vis_inputs

    def forward(
        self,
        obs,  #: Observations,
        recep_sem_id: int,
        has_obj_recep_collided,
        safe_to_drop,
        vis_inputs: Optional[Dict] = None,
    ):
        """
        1. Get estimate of point on receptacle to place object on.
        2. Orient towards it.
        3. Move forward to get close to it.
        4. Rotate 90º to have arm face the object. Then rotate camera to face arm.
        5. (again) Get estimate of point on receptacle to place object on.
        6. With camera, arm, and object (hopefully) aligned, set arm lift and
        extension based on point estimate from 4.

        Returns:
            action: what the robot will do - a hybrid action, discrete or continuous
            vis_inputs: dictionary containing extra info for visualizations
        """

        turn_angle = self.config.ENVIRONMENT.turn_angle
        fwd_step_size = self.config.ENVIRONMENT.forward

        # OVMM specific observation preprocessing
        obs["camera_pose"] = camera_pose_transform(np.asarray(obs["camera_pose"]))
        obs["head_depth"] = place_utils.preprocess_depth(obs["head_depth"])

        if self.timestep == 0:
            self.sub_timestep = 0
            self.arm_joint_to_extend = 0
            self.prev_arm_ext = 0
            self._lift_up_actions = []
            self._arm_extension_actions = []

            self.du_scale = 1  # TODO: working with full resolution for now
            # self.end_receptacle = obs.task_observations["goal_name"].split(" ")[-1]
            found = self.get_receptacle_placement_point(obs, recep_sem_id, vis_inputs)

            if found is None:
                if self.verbose:
                    print("Receptacle not visible. Execute hardcoded place.")
                self.total_turn_and_forward_steps = 0
                self.initial_orient_num_turns = -1
                self.fall_wait_steps = 0
                self.t_go_to_top = 1
                self.t_extend_arm = 2
                self.t_release_object = 3
                self.t_lift_arm = 4
                self.t_lift_back_and_retract = 5
                self.t_go_to_place = -1
                self.t_go_to_low = 6
                self.t_done_waiting = 6 + self.fall_wait_steps
            
            else:
                self.placement_voxel, vis_inputs = found

                center_voxel_trans = np.array(
                    [
                        self.placement_voxel[1],
                        self.placement_voxel[2],
                        self.placement_voxel[0],
                    ]
                )

                delta_heading = np.rad2deg(place_utils.get_angle_to_pos(center_voxel_trans))

                self.initial_orient_num_turns = abs(delta_heading) // turn_angle
                self.orient_turn_direction = np.sign(delta_heading)
                # This gets the Y-coordiante of the center voxel
                # Base link to retracted arm - this is about 15 cm
                fwd_dist = (
                    self.placement_voxel[1]
                    - STRETCH_STANDOFF_DISTANCE
                    - RETRACTED_ARM_APPROX_LENGTH
                )

                fwd_dist = np.clip(fwd_dist, 0, np.inf)  # to avoid negative fwd_dist
                self.forward_steps = fwd_dist // fwd_step_size
                self.total_turn_and_forward_steps = (
                    self.forward_steps + self.initial_orient_num_turns
                )
                # self.fall_wait_steps = 5
                self.t_go_to_top = self.total_turn_and_forward_steps + 1
                self.t_go_to_place = self.total_turn_and_forward_steps + 2
                self.t_release_object = self.total_turn_and_forward_steps + 3
                self.t_lift_arm = -1 # self.total_turn_and_forward_steps + 4
                self.t_lift_back_and_retract = self.total_turn_and_forward_steps + 4
                self.t_go_to_low = self.total_turn_and_forward_steps + 5
                self.t_extend_arm = -1
                self.t_done_waiting = (
                    self.total_turn_and_forward_steps + 5 + self.fall_wait_steps
                )
                if self.verbose:
                    print("-" * 20)
                    print(f"Turn to orient for {self.initial_orient_num_turns} steps.")
                    print(f"Move forward for {self.forward_steps} steps.")

        if self.verbose:
            print("-" * 20)
            print("Timestep", self.timestep)

        if self.sub_timestep == 0:
            
            if self.verbose:
                print("Executing new skill step...")

            if self.timestep < self.initial_orient_num_turns:
                if self.orient_turn_direction == -1:
                    action = place_utils.DiscreteNavigationAction.TURN_RIGHT
                if self.orient_turn_direction == +1:
                    action = place_utils.DiscreteNavigationAction.TURN_LEFT
                if self.verbose:
                    print("[Placement] Turning to orient towards object")
                self.action_desc = "turn to object"
            
            elif self.timestep < self.total_turn_and_forward_steps:
                if self.verbose:
                    print("[Placement] Moving forward")
                self.action_desc = "move to object"
                action = place_utils.DiscreteNavigationAction.MOVE_FORWARD
            
            elif self.timestep == self.total_turn_and_forward_steps:
                action = place_utils.DiscreteNavigationAction.MANIPULATION_MODE
                self.action_desc = "goto manip"
            
            elif self.timestep == self.t_go_to_top:
                # We should move the arm to the top position to prevent collision with receptacles while extending the arm to place the object on target receptacle
                current_arm_lift = obs["joint"][4]
                self.lift_delta_per_step = (self.max_arm_height - current_arm_lift) / self.sub_skill_num_steps

                action = self._lift(obs)
                self.action_desc = "goto top"
                self._lift_up_actions.append(copy.deepcopy(action))

                return action, vis_inputs, self.action_desc
                
            elif self.timestep == self.t_go_to_place:

                placement_height, placement_extension = (
                    self.placement_voxel[2],
                    self.placement_voxel[1],
                )

                current_arm_lift = obs["joint"][4]
                self.delta_arm_lift_per_step = (placement_height - current_arm_lift + STRETCH_RECEPTACLE_CLEARANCE) / self.sub_skill_num_steps

                current_arm_ext = obs["joint"][:4].sum()
                self.delta_arm_ext_per_step = (
                    placement_extension
                    - STRETCH_STANDOFF_DISTANCE
                    - RETRACTED_ARM_APPROX_LENGTH
                    - current_arm_ext
                    + HARDCODED_ARM_EXTENSION_OFFSET
                ) / self.sub_skill_num_steps

                center_voxel_trans = np.array(
                    [
                        self.placement_voxel[1],
                        self.placement_voxel[2],
                        self.placement_voxel[0],
                    ]
                )
                delta_heading = np.rad2deg(place_utils.get_angle_to_pos(center_voxel_trans))
                self.delta_gripper_yaw_per_step = (delta_heading / 90 - HARDCODED_YAW_OFFSET) / self.sub_skill_num_steps

                if self.verbose:
                    print("[Placement] Move arm into position")
                    
                action = self._move_arm_into_position(obs)
                self.action_desc = "lift/extend arm"
                self._arm_extension_actions.append(copy.deepcopy(action))

                return action, vis_inputs, self.action_desc

            elif self.timestep == self.t_release_object:
                
                if not has_obj_recep_collided and not self._ready_to_drop_object:
                    # keep lowering the arm until the object collides with the receptacle
                    action = self._lift_down(obs)
                    self.timestep -= 1
                    self.action_desc = "goto low b4 desnap"
                    
                else:
                    if safe_to_drop:
                        # Enough clearance to drop object so drop immediately
                        action = place_utils.DiscreteNavigationAction.DESNAP_OBJECT
                        self._is_obj_dropped = True
                        self.action_desc = "desnap obj"
                    else:
                        if self._num_lift_actions_after_desnap > 0:
                            # Not enough clearance to drop object so lifting back up.....
                            joints = self._lift_up_actions[len(self._lift_up_actions) - self._num_lift_actions_after_desnap - 1].joints
                            action = place_utils.ContinuousFullBodyAction(joints)
                            self._num_lift_actions_after_desnap -= 1
                            self.timestep -= 1
                        else:
                            # Dropping Object even though we dont have enough clearance from the receptacle surface because we maxed out arm lift
                            action = place_utils.DiscreteNavigationAction.DESNAP_OBJECT
                            self._is_obj_dropped = True
                            self.action_desc = "desnap obj"                            
                    
            elif self.timestep == self.t_lift_back_and_retract:
                
                assert self._is_obj_dropped is True

                if self._num_lift_actions_after_desnap > 0:
                    # Object was DROPPED so need to lift back arm to max position for a clean retraction....
                    joints = self._lift_up_actions[len(self._lift_up_actions) - self._num_lift_actions_after_desnap - 1].joints
                    action = place_utils.ContinuousFullBodyAction(joints)
                    self._num_lift_actions_after_desnap -= 1
                    self.timestep -= 1
                else:
                    # Retracting arm...
                    if self.t_go_to_place != -1:        # NOTE: If t_go_to_place = -1 -> a placement_voxel was never found (since robot cant see the receptacle) and the robot never actually extended its arm so we dont need a retraction
                        action = self._retract(obs)
                        self.action_desc = "retract arm"
                        return action, vis_inputs, self.action_desc
                    else:
                        action = place_utils.DiscreteNavigationAction.EMPTY_ACTION
                        self.action_desc = "skip retraction"
                            
            elif self.timestep == self.t_go_to_low:
                action = self._lift_down(obs)
                self.action_desc = "goto low"
                return action, vis_inputs, self.action_desc
                        
            elif self.timestep <= self.t_done_waiting:
                if self.verbose:
                    print("[Placement] Empty action")  # allow the object to come to rest
                self.action_desc = "waiting"
                action = place_utils.DiscreteNavigationAction.EMPTY_ACTION
            
            else:
                if self.verbose:
                    print("[Placement] Stopping")
                self.action_desc = "stopping"    
                action = place_utils.DiscreteNavigationAction.STOP

        else:
            if self.verbose:    
                print("Executing place skill substep...")
            
            if self.timestep == self.t_go_to_top:
                action = self._lift(obs)
                self._lift_up_actions.append(copy.deepcopy(action))
            elif self.timestep == self.t_go_to_place:
                action = self._move_arm_into_position(obs)
                self._arm_extension_actions.append(copy.deepcopy(action))
            elif self.timestep == self.t_lift_back_and_retract:
                action = self._retract(obs)
            elif self.timestep == self.t_go_to_low:
                action = self._lift_down(obs)
            
            elif self.timestep == self.t_release_object:
                if not has_obj_recep_collided:
                    # We can still lower the arm further as object has not touched receptacle surface
                    action = self._lift_down(obs)
                    
                    if self.timestep == self.t_release_object + 1:
                        # Could not find point to just lift back up as object never touched receptacle
                        self.timestep = self.t_release_object
                        self._ready_to_drop_object  = True
                        self._num_lift_actions_after_desnap = len(self._lift_up_actions) - 1
                else:
                    # lift the arm back up just slightly to desnap as we just found the point to lift back up
                    
                    self.sub_timestep -= 1
                    joints = self._lift_up_actions[len(self._lift_up_actions) - self.sub_timestep - 1].joints
                    action = place_utils.ContinuousFullBodyAction(joints)
                    
                    self._ready_to_drop_object = True
                    self._num_lift_actions_after_desnap = self.sub_timestep
                    self.sub_timestep = 0
                    
            return action, vis_inputs, self.action_desc

        debug_texts = {
            self.total_turn_and_forward_steps: "[Placement] Aligning camera to arm",
            self.t_go_to_top: "[Placement] Raising the arm before placement.",
            self.t_go_to_place: "[Placement] Move arm into position",
            self.t_release_object: "[Placement] Desnapping object",
            self.t_lift_arm: "[Placement] Lifting the arm after placement.",
            self.t_lift_back_and_retract: "[Placement] Lifting up and Retracting the arm after placement.",
            self.t_extend_arm: "[Placement] Extending the arm out for placing.",
            self.t_done_waiting: "[Placement] Empty action",
        }
        if self.verbose and self.timestep in debug_texts:
            print(debug_texts[self.timestep])

        self.timestep += 1
 
        return action, vis_inputs, self.action_desc


    def _move_arm_into_position(self, obs):

        current_arm_ext = obs["joint"][:4].sum()

        joints = np.array(
            [0] * 4 + [self.delta_arm_lift_per_step] + [self.delta_gripper_yaw_per_step] + [0] * 4
        )
        if current_arm_ext == self.prev_arm_ext and self.arm_joint_to_extend < 3:
            self.arm_joint_to_extend += 1
            joints[self.arm_joint_to_extend] = self.delta_arm_ext_per_step * 2
        else:
            joints[self.arm_joint_to_extend] = self.delta_arm_ext_per_step

        joints = self._look_at_ee(joints)
        action = place_utils.ContinuousFullBodyAction(joints)

        # Update sub-timestep
        self.sub_timestep += 1
        if self.sub_timestep >= self.sub_skill_num_steps:
            self.sub_timestep = 0  # Reset for next full operation
            self.timestep += 1     # Move to the next main timestep
            self.arm_joint_to_extend = 0

        self.prev_arm_ext = current_arm_ext
        
        return action

    def _lift(self, obs) -> place_utils.ContinuousFullBodyAction:
        """Compute a high-up lift position to avoid collisions when releasing"""

        # Hab sim dimensionality for arm == 10
        joints = np.zeros(10)

        joints[4] = self.lift_delta_per_step
        joints = self._look_at_ee(joints)
        action = place_utils.ContinuousFullBodyAction(joints)
        
        # Update sub-timestep
        self.sub_timestep += 1
        if self.sub_timestep >= self.sub_skill_num_steps:
            self.sub_timestep = 0  # Reset for next full operation
            self.timestep += 1     # Move to the next main timestep
        
        return action

    def _look_at_ee(self, joints: np.ndarray) -> np.ndarray:
        """Make sure it's actually looking at the end effector."""
        joints[8] = self.look_at_ee[0] / self.sub_skill_num_steps
        joints[9] = self.look_at_ee[1] / self.sub_skill_num_steps
        return joints

    def _retract(self, obs) -> place_utils.ContinuousFullBodyAction:
        """Retract the extended arm before lowering to prevent collision with nearby receptacles"""
        
        joints = -1 * self._arm_extension_actions[len(self._arm_extension_actions) - self.sub_timestep - 1].joints
        joints[4:] = 0.
        action = place_utils.ContinuousFullBodyAction(joints)

        self.sub_timestep += 1
        if self.sub_timestep > len(self._arm_extension_actions):
            self.sub_timestep = 0  # Reset for next full operation
            self.timestep += 1     # Move to the next main timestep

        return action

    def _lift_down(self, obs) -> place_utils.ContinuousFullBodyAction:
        """Lower the lifted arm after the object has been placed on the target receptacle"""

        joints = -1 * self._lift_up_actions[len(self._lift_up_actions) - self.sub_timestep - 1].joints
        action = place_utils.ContinuousFullBodyAction(joints)

        self.sub_timestep += 1
        if self.sub_timestep > len(self._lift_up_actions):
            self.sub_timestep = 0  # Reset for next full operation
            self.timestep += 1     # Move to the next main timestep

        return action