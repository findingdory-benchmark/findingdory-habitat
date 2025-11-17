#!/usr/bin/env python3

import numpy as np
from omegaconf import OmegaConf

from habitat.sims.habitat_simulator.actions import HabitatSimActions
import habitat_sim
from habitat_sim.physics import MotionType
import magnum as mn

import findingdory.config.default_structured_configs as default_configs
from findingdory.policies.heuristic.heuristic_place_policy import HeuristicPlacePolicy
from findingdory.policies.heuristic.heuristic_pick_policy import HeuristicPickPolicy
from findingdory.policies.heuristic.place_utils import ContinuousFullBodyAction
from findingdory.dataset.utils import teleport_agent_to_state_quat_assert
from findingdory.utils import get_device, ACTION_MAPPING
from findingdory.policies.agent import Agent

from findingdory.dataset.utils import get_agent_yaw, coeff_to_yaw


MAX_PLACE_BUDGET = 100
ARM_PICK_EXTENSION = 1.0


class DummyAgent:
    def __init__(self, sim, agent_id=0):
        self.sim = sim
        self.agent_id = agent_id
        self.agent = self.sim.get_agent(agent_id)

    def __getattr__(self, attr):
        return getattr(self.agent, attr)

    @property
    def state(self):
        return self.sim.get_agent_state(self.agent_id)


class OracleAgent(Agent):
    def __init__(
        self,
        sim,
        task,
        config,
    ):
        self.sim = sim
        self.task = task
        self.pick_task_step = 0
        self.forward_step_size = sim.habitat_config.forward_step_size
        self.turn_angle = sim.habitat_config.turn_angle
        self.follower = self.sim.make_greedy_follower(
            agent_id=0,
            goal_radius=self.forward_step_size,
            stop_key=HabitatSimActions.stop,
            forward_key=HabitatSimActions.move_forward,
            left_key=HabitatSimActions.turn_left,
            right_key=HabitatSimActions.turn_right,
        )
        self.follower.agent = DummyAgent(self.sim)
        self.assert_counter = 0

        rgb_camera_config = sim.habitat_config.agents.main_agent.sim_sensors.head_rgb_sensor
        self.heursitic_place_conf = OmegaConf.create(
            {
                "ENVIRONMENT": {
                    "frame_width": rgb_camera_config.width,
                    "frame_height": rgb_camera_config.height,
                    "hfov": rgb_camera_config.hfov,
                    "turn_angle": self.turn_angle,
                    "forward": self.forward_step_size,
                },
                "sub_skill_num_steps": 10,
                "fall_wait_steps": 5,
            }
        )

        # TODO: extract the device name form the sim/agent object
        self.place_policy = HeuristicPlacePolicy(
            config=self.heursitic_place_conf, device=get_device(), verbose=False
        )
        self.pick_policy = HeuristicPickPolicy(
            config=self.heursitic_place_conf, device=get_device(), verbose=False
        )

        self.grasp_manager = self.sim.agents_mgr[0].grasp_mgrs[0]
        self.current_goal = None
        self.current_policy = "nav_pick"

        # Flag variables for reproducing oracle place policy routine from cached action sequence
        self.place_first_step = True
        self.place_setup_arm_orientation = True
        self._place_orientation_check = True
        self.ready_to_desnap = False
        self.pre_drop_state = None
        self._set_dropped_obj_kinematic_state = False

    def reset(self) -> None:
        r"""Called before starting a new episode in environment."""
        self.assert_counter = 0
        self.follower.reset()
        self.place_policy.reset()
        self.pick_policy.reset()
    
    def get_navigation_mode_action(self):
        return {
            "action": ("navigation_mode"),
            "action_args": {"navigation_mode": [1.0], "is_last_action": True},
        }

    def turn_towards_goal(self, snapped_goal_orientation):
        current_rotation_yaw = get_agent_yaw(self.sim.get_agent_state())
        if type(snapped_goal_orientation) == float or len(snapped_goal_orientation) == 1:
            goal_rotation_yaw = snapped_goal_orientation
        else:
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

    def nav_to_goal(self, snapped_goal, snapped_goal_orientation):
        try:
            action = self.follower.next_action_along(snapped_goal)

            if action is HabitatSimActions.stop:
                action = self.turn_towards_goal(snapped_goal_orientation)

        except habitat_sim.errors.GreedyFollowerError:
            self.assert_counter += 1
            action = HabitatSimActions.stop

        if action is None:
            action = HabitatSimActions.stop

        action_dict = {
            "action": action
        }
        return action_dict, action

    def pick(self, observations, oracle_goal_idx):
        query_label = self.sim._candidate_obj_idx_to_rigid_obj_handle[oracle_goal_idx]
        obj_id = self.sim.handle_to_object_id[query_label]
        scene_obj_id = self.sim.scene_obj_ids[obj_id]
        pick_goal_sem_id = self.task._object_semantic_ids[scene_obj_id]
        
        pick_action, _, action_desc = self.pick_policy(observations, pick_goal_sem_id)

        if type(pick_action) == ContinuousFullBodyAction:
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": pick_action.joints, "grip_action": [0.0, 0.0, 0.0]},
            }
            action_idx = 8

        elif pick_action.name == "MOVE_FORWARD":
            action_dict = {
                "action": (HabitatSimActions.move_forward),
            }
            action_idx = 1
        elif pick_action.name == "TURN_LEFT":
            action_dict = {
                "action": (HabitatSimActions.turn_left),
            }
            action_idx = 2
        elif pick_action.name == "TURN_RIGHT":
            action_dict = {
                "action": (HabitatSimActions.turn_right),
            }
            action_idx = 3
        elif pick_action.name == "MANIPULATION_MODE":
            action_dict = {
                "action": ("manipulation_mode"),
                "action_args": {"manipulation_mode": [1.0], "is_last_action": True},
            }
            action_idx = 4
        elif pick_action.name == "NAVIGATION_MODE":
            action_dict = {
                "action": ("navigation_mode"),
                "action_args": {"navigation_mode": [1.0], "is_last_action": True},
            }
            action_idx = 10
        elif pick_action.name == "DESNAP_OBJECT":
            self.grasp_manager.desnap()
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
            }
            action_idx = 5
        elif pick_action.name == "STOP":
            action_dict = {
                "action": (HabitatSimActions.stop),
            }
            action_idx = 0

        elif pick_action.name == "EMPTY_ACTION":
            # setup a dummy action based on zero relative joint movement
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
            }
            action_idx = 6
        elif pick_action.name == "EXTEND_ARM":
            action_dict = {
                "action": ("arm_action"),
                "action_args": {
                    "arm_action": np.array(
                        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ),
                    "grip_action": [0.0, 0.0, 0.0]
                },
            }
            action_idx = 7
        elif pick_action.name == "SNAP_OBJECT":
            rom = self.sim.get_rigid_object_manager()
            self.grasp_manager.snap_to_obj(scene_obj_id)
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
            }
            action_idx = 9
        else:
            raise AssertionError("Unknown action returned by heuristic place policy")

        self.pick_task_step += 1
        return (action_dict, action_idx)

    def place(self, observations, oracle_agent_goal_idx, place_snapped_goal, place_snapped_goal_orientation, measures):
        
        oracle_place_action = None
        oracle_place_action_name = None
        
        # Before teleportation to the placing viewpoint, set the agent arm joints as setting joints after teleportation causes issues in exactly reproducing the oracle placing routine with cached actions
        if self.place_setup_arm_orientation:
            action_dict = {
                "action": ("arm_action"),
                "action_args": {
                    "arm_action": np.array(
                        [0., 0., 0., 0., 0.8611111, 0., -1.57, 0., 0., -0.58177644]     # This is the exact initial arm joint configuration that was used while selecting the placing viewpoint
                    ),
                    "grip_action": [0.0, 0.0, 0.0]
                },
            }
            action_idx = 6
            action_desc = "set_arm_joints_for_place"
            self.place_setup_arm_orientation = False

            return (action_dict, action_idx)

        # Teleport the agent to the specific place viewpoint that gives us a successful object placement (based on cached actions)
        # This is because the shortest path greedy follower may not bring the agent to the exact place viewpoint but only very close to it. 
        if self.place_first_step:
            # Teleport the agent to the exact viewpoint orientation
            teleport_agent_to_state_quat_assert(self.sim, place_snapped_goal_orientation, place_snapped_goal)

            # Empty action to verify agent is infront of the target recep
            action_dict = {
                "action": ("arm_action"),
                "action_args": {
                    "arm_action": np.zeros(10),
                    "grip_action": [0.0, 0.0, 0.0]
                },
            }
            action_idx = 6
            action_desc = "teleport_for_place"
            self.place_first_step = False

            return (action_dict, action_idx)
        
        # Re check if the agent has been exactly teleported to the required placing viewpoint
        if self._place_orientation_check:
            cur_pos = np.array(
                [
                    self.sim.get_agent_state().position.x,
                    self.sim.get_agent_state().position.y,
                    self.sim.get_agent_state().position.z
                ]
            )
            cur_orient = np.array(
                [
                    self.sim.get_agent_state().rotation.x,
                    self.sim.get_agent_state().rotation.y,
                    self.sim.get_agent_state().rotation.z,
                    self.sim.get_agent_state().rotation.w
                ]
            )
            assert np.allclose(cur_orient, place_snapped_goal_orientation, atol=1e-5) or np.allclose(-cur_orient, place_snapped_goal_orientation, atol=1e-5), \
                "Agent orientation doesnt match with viewpoint orientation after teleport operation!"
            assert np.allclose(cur_pos, place_snapped_goal, atol=1e-5), \
                "Agent position doesnt match with viewpoint position after teleport operation!"
            
            self._place_orientation_check = False
        
        # We are ready to desnap the object as the agent/object have been teleported to the pre-desnap state 
        if self.ready_to_desnap:
            oracle_place_action_name = "DESNAP_OBJECT"
        else:    
            oracle_place_action = self.sim.ep_info.place_oracle_action_seq[str(oracle_agent_goal_idx)].pop(0)
            if isinstance(oracle_place_action, str):        # If the action is not an arm_extension/desnap, we dont have any action_args so directly parse the action_name string
                oracle_place_action_name = oracle_place_action
            else:
                oracle_place_action_name = oracle_place_action[0]
        
        # If the object has been desnapped, we set the exact object kinematic state to ensure reproducibility fo the oracle place sequence (instead of relying on bullet dynamics which is non-deterministic)
        if self._set_dropped_obj_kinematic_state and oracle_place_action_name != "STOP":
            rom = self.sim.get_rigid_object_manager()
            query_label = self.sim._candidate_obj_idx_to_rigid_obj_handle[oracle_agent_goal_idx]
            dropped_obj = rom.get_object_by_handle(query_label)
            dropped_obj.motion_type = MotionType.KINEMATIC
            
            dropped_state = None
            if oracle_place_action_name == "MOVE_ARM":
                dropped_state = mn.Matrix4(np.array(oracle_place_action[2]))
            else:
                dropped_state = mn.Matrix4(np.array(oracle_place_action[1]))
            dropped_obj.transformation = dropped_state

        if oracle_place_action_name == "MOVE_ARM":
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": np.array(oracle_place_action[1]["action_args"]["arm_action"]), "grip_action": [0.0, 0.0, 0.0]},
            }
            action_idx = 8

        elif oracle_place_action_name == "MOVE_FORWARD":
            action_dict = {
                "action": (HabitatSimActions.move_forward),
            }
            action_idx = 1
        elif oracle_place_action_name == "TURN_LEFT":
            action_dict = {
                "action": (HabitatSimActions.turn_left),
            }
            action_idx = 2
        elif oracle_place_action_name == "TURN_RIGHT":
            action_dict = {
                "action": (HabitatSimActions.turn_right),
            }
            action_idx = 3
        elif oracle_place_action_name == "MANIPULATION_MODE":
            action_dict = {
                "action": ("manipulation_mode"),
                "action_args": {"manipulation_mode": [1.0], "is_last_action": True},
            }
            action_idx = 4
        elif oracle_place_action_name == "DESNAP_OBJECT":
            # Before actually desnapping, make sure the agent and picked object are in the exact state which leads to successful placement on receptacle after dropping
            # DESNAP oracle place action list contains: [action_name, densap_agent_pos, desnap_agent_orientation, desnap_picked_obj_transformation]
            if not self.ready_to_desnap:
                teleport_agent_to_state_quat_assert(self.sim, np.array(oracle_place_action[2]), np.array(oracle_place_action[1]))                
                self.pre_drop_state = mn.Matrix4(np.array(oracle_place_action[3]))

                action_dict = {
                    "action": ("arm_action"),
                    "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
                }
                action_idx = 6
                self.ready_to_desnap = True
            else:
                # Right when we are ready to desnap, we set the picked object state
                query_label = self.sim._candidate_obj_idx_to_rigid_obj_handle[self.task._current_oracle_goal_idx]
                rom = self.sim.get_rigid_object_manager()
                picked_obj = rom.get_object_by_handle(query_label)
                picked_obj.transformation = self.pre_drop_state
                
                self.grasp_manager.desnap()
                action_dict = {
                    "action": ("arm_action"),
                    "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
                }
                action_idx = 5
                self.ready_to_desnap = False
                self._set_dropped_obj_kinematic_state = True
                
        elif oracle_place_action_name == "STOP":
            # Explicitly set the motion_type of the placed object to KINEMATIC to prevent any further rolling as motion_type changes to DYNAMIC after desnap action
            rom = self.sim.get_rigid_object_manager()
            query_label = self.sim._candidate_obj_idx_to_rigid_obj_handle[oracle_agent_goal_idx]
            placed_object = rom.get_object_by_handle(query_label)
            placed_object.motion_type = MotionType.KINEMATIC

            action_dict = {
                "action": (HabitatSimActions.stop),
            }
            action_idx = 0
            
            # NOTE: Hard assertion that checks if object was placed on receptacle or not
            # assert measures['picked_obj_anywhere_on_goal']._metric[str(placed_object.object_id)] == True, "Oracle agent failed to place object on receptacle !"
            
        elif oracle_place_action_name == "EMPTY_ACTION":
            # setup a dummy action based on zero relative joint movement
            action_dict = {
                "action": ("arm_action"),
                "action_args": {"arm_action": np.zeros(10), "grip_action": [0.0, 0.0, 0.0]},
            }
            action_idx = 6
        elif oracle_place_action_name == "EXTEND_ARM":
            action_dict = {
                "action": ("arm_action"),
                "action_args": {
                    "arm_action": np.array(
                        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ),
                    "grip_action": [0.0, 0.0, 0.0]
                },
            }
            action_idx = 7
        else:
            raise ValueError(f"Unknown action {oracle_place_action_name} returned by heuristic place policy")

        self.place_task_step += 1
        return (action_dict, action_idx)
    
    def act(self, observations, goal_info, oracle_agent_goal_idx, measures):
        # stretch robot joints:
        # [0,1,2,3] -> arm extension
        # [4] - arm lift joint
        # [5] - gripper yaw
        # [6] - wrist pitch
        # [7] - wrist roll
        # [8,9] - head camera pan/tilt

        if 'goal_position' in goal_info:
            action, self.action_idx = self.nav_to_goal(
                goal_info['goal_position'],
                goal_info['goal_orientation']
            )
            current_task_done = action['action'] == HabitatSimActions.stop
            if current_task_done:
                action = {
                    "action": ("navigation_mode"),
                    "action_args": {"navigation_mode": [1.0], "is_last_action": True},
                }
                self.action_idx = 10
            return action, current_task_done, self.action_idx
        else:
            pick_goal = goal_info['pick_goal']
            pick_goal_idx = goal_info['pick_goal_idx']
            place_goal_idx = goal_info['place_goal_idx']
            place_goal = goal_info['place_goal']

            current_task_done = False
            if self.current_policy == "nav_pick":
                # Navigate towards the chosen candidate object
                action, self.action_idx = self.nav_to_goal(
                    # snapped viewpoint position and orientation
                    pick_goal.view_points[pick_goal_idx].agent_state.position,
                    pick_goal.view_points[pick_goal_idx].agent_state.rotation
                )

                if action['action'] == HabitatSimActions.stop:
                    self.current_policy = "pick"
                    self.pick_task_step = 0

            if self.current_policy == "pick":
                # Execute a heuristic pick action
                action, self.action_idx = self.pick(observations, oracle_agent_goal_idx)

                if action['action'] == HabitatSimActions.stop:
                    self.pick_policy.reset()
                    self.current_policy = "navigation_mode"
                    self.switch_to_nav_place = True
                
                # This happens when the object to be picked is not visible in the agent frame so we will try to navigate to an alternate viewpoint of the pick object goal
                if self.get_current_action_name == "empty_action":
                    self.pick_policy.reset()
                    self.pick_task_step = 0
                    self.current_policy = "retry_nav_pick"
                    action = {
                        "action": ("navigation_mode"),
                        "action_args": {"navigation_mode": [1.0], "is_last_action": True},
                    }       # First we will switch to the navigation_mode action before so that we can re-execute the nav_pick routine
                    self.action_idx = 10
                
            if self.current_policy == "nav_place":
                # Move towards the target goal receptacle
                action, self.action_idx = self.nav_to_goal(
                    # snapped viewpoint position and orientation
                    place_goal.view_points[place_goal_idx].agent_state.position,
                    place_goal.view_points[place_goal_idx].agent_state.rotation
                )

                if action['action'] == HabitatSimActions.stop:
                    self.current_policy = "place"
                    self.place_task_step = 0

            if self.current_policy == "place":
                # Place object on goal receptacle by replaying the oracle place policy actions which gave a successful placement                        
                action, self.action_idx = self.place(observations,
                                                    oracle_agent_goal_idx,
                                                    place_goal.view_points[place_goal_idx].agent_state.position,
                                                    place_goal.view_points[place_goal_idx].agent_state.rotation,
                                                    measures)

                if self.action_idx == HabitatSimActions.stop or self.place_task_step >= MAX_PLACE_BUDGET:
                    self.place_policy.reset()
                    self.current_policy = "navigation_mode"
                    self.place_first_step = True
                    self.place_setup_arm_orientation = True
                    self._place_orientation_check = True
                    self._set_dropped_obj_kinematic_state = False
                    self.switch_to_nav_place = False

            if self.current_policy == "navigation_mode":
                action = {
                    "action": ("navigation_mode"),
                    "action_args": {"navigation_mode": [1.0], "is_last_action": True},
                }
                self.action_idx = 10
                
                # Current task should not end when we are switching to nav mode when the pick routine ends. Current task ends only when we switch to nav mode after place routine ends
                if not self.switch_to_nav_place:    
                    current_task_done = True
                    self.current_policy = "nav_pick"
                else:
                    self.current_policy = "nav_place"
                    
            return action, current_task_done, self.action_idx

    @property
    def get_current_action_name(self):
        return ACTION_MAPPING[self.action_idx]