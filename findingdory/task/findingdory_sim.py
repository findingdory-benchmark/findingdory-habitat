#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Dict
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import magnum as mn

import habitat_sim
from habitat.core.simulator import Observations
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import set_agent_base_via_obj_trans
from findingdory.dataset.findingdory_dataset import FindingDoryEpisode
from habitat_sim.logging import logger
from findingdory.dataset.utils import teleport_agent_to_state

import habitat_sim
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from typing import (
    Dict,
    List,
    Optional,
)
from collections import defaultdict
import time
import magnum as mn
import os.path as osp
import numpy as np

@registry.register_simulator(name="FindingDorySim-v0")
class FindingDorySim(RearrangeSim):
    def _setup_targets(self, ep_info):
        super()._setup_targets(ep_info)
        
        if ep_info.candidate_start_receps is not None:
            self.valid_goal_rec_obj_ids = {
                int(g.object_id) for g in ep_info.candidate_start_receps
            }

        if ep_info.candidate_start_receps is not None:
            self.valid_goal_rec_names = [
                g.object_name for g in ep_info.candidate_start_receps
            ]

    def _create_recep_info(
        self, scene_id: str, ignore_handles: List[str]
    ):
        super()._create_recep_info(scene_id, ignore_handles)
        scene_obj_name = set([r.parent_object_handle for r in self._receptacles_cache[scene_id].values()])

        rom = self.get_rigid_object_manager()
        self._scene_recep_ids = [
            int(rom.get_object_id_by_handle(obj_name))
            for obj_name in scene_obj_name
        ]
        self._obj_handle_to_receps = {}
        for obj_name in scene_obj_name:
            self._obj_handle_to_receps[obj_name] = [
                r
                for r in self._receptacles_cache[scene_id].values()
                if r.parent_object_handle == obj_name
            ]

    @property
    def scene_recep_ids(self) -> List[int]:
        """
        The simulator rigid body IDs of all objects in the scene.
        """
        return self._scene_recep_ids

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()

        agent = self.get_agent_data(0).articulated_agent
        if position is None or rotation is None:
            return None

        teleport_agent_to_state(self, rotation, position)
        agent.update()

        sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs

        observations = self._sensor_suite.get_observations(sim_obs)
        if not keep_agent_at_new_pose:
            teleport_agent_to_state(
                self,
                current_state.rotation.angle(),
                current_state.position,
            )

            agent.update()

        return observations

    def _add_objs(
        self,
        ep_info: FindingDoryEpisode,
        should_add_objects: bool,
        new_scene: bool,
    ) -> None:
        # Load clutter objects:
        rom = self.get_rigid_object_manager()
        obj_counts: Dict[str, int] = defaultdict(int)

        self._handle_to_object_id = {}
        if should_add_objects:
            self._scene_obj_ids = []
        
        if ep_info.candidate_objects is not None:
            candidate_obj_names = [object.object_name for object in ep_info.candidate_objects]
        else:
            candidate_obj_names = []
        self._candidate_obj_idx_to_rigid_obj_handle = {}

        if ep_info.candidate_objects_noninteracted is not None:
            candidate_obj_noninteracted_names = [object.object_name for object in ep_info.candidate_objects_noninteracted]
        else:
            candidate_obj_noninteracted_names = []
        self._candidate_obj_noninteracted_idx_to_rigid_obj_handle = {}

        # Get Object template manager
        otm = self.get_object_template_manager()
        
        # Separate the candidate objects from the remaining ones
        candidate_objs = [obj for obj in ep_info.rigid_objs if obj[0] in candidate_obj_names]
        remaining_objs = [obj for obj in ep_info.rigid_objs if obj[0] not in candidate_obj_names]

        # Combine them: first candidate_objs, then remaining_objs
        sorted_objs = candidate_objs + remaining_objs
        assert len(sorted_objs) == len(ep_info.rigid_objs)

        for i, (obj_handle, transform) in enumerate(sorted_objs):
            t_start = time.time()
            if should_add_objects:
                # Get object path
                # object_template = otm.get_templates_by_handle_substring(
                #     obj_handle
                # )
                object_template = None
                for obj_path in self._additional_object_paths:
                    object_template = osp.join(obj_path, obj_handle)
                    if osp.isfile(object_template):
                        break

                # Exit if template is invalid
                if not object_template:
                    raise ValueError(
                        f"Template not found for object with handle {obj_handle}"
                    )

                # Get object path
                object_path = object_template #list(object_template.keys())[0]

                # Get rigid object from the path
                ro = rom.add_object_by_template_handle(object_path)
            else:
                ro = rom.get_object_by_id(self._scene_obj_ids[i])

            self.add_perf_timing("create_asset", t_start)

            other_obj_handle = (
                obj_handle.split(".")[0] + f"_:{obj_counts[obj_handle]:04d}"
            )
            # If other_obj_handle does not match the object handle assigned by the cpp-level rigid object manager, then explicitly set the other_obj_handle to the ro.handle 
            if ro.handle != other_obj_handle:
                logger.info(
                    f"Object added by ROM has different handle name than what is expected potentially because a similar object name was previously added. Setting the expected handle to the actual RO handle.... \nExpected handle: {other_obj_handle}, but actual handle set by ROM: {ro.handle}"
                )
                other_obj_handle = ro.handle
                
            # The saved matrices need to be flipped when reloading.
            ro.transformation = mn.Matrix4(
                [[transform[j][i] for j in range(4)] for i in range(4)]
            )
            ro.angular_velocity = mn.Vector3.zero_init()
            ro.linear_velocity = mn.Vector3.zero_init()

            if self._kinematic_mode:
                ro.motion_type = habitat_sim.physics.MotionType.KINEMATIC

            if should_add_objects:
                self._scene_obj_ids.append(ro.object_id)
            rel_idx = self._scene_obj_ids.index(ro.object_id)
            self._handle_to_object_id[other_obj_handle] = rel_idx

            if other_obj_handle in self._handle_to_goal_name:
                ref_handle = self._handle_to_goal_name[other_obj_handle]
                self._handle_to_object_id[ref_handle] = rel_idx

            obj_counts[obj_handle] += 1

            # Create a mapping between candidate object indices and the rigid object handle name as there maybe multiple candidate objects with same object_name
            if obj_handle in candidate_obj_names:
                indices = [i for i, x in enumerate(candidate_obj_names) if x == obj_handle]

                if len(indices) > 0:
                    cur_obj_trans = rom.get_object_by_handle(other_obj_handle).transformation.translation
                    cur_obj_trans_np = np.array([cur_obj_trans.x, cur_obj_trans.y, cur_obj_trans.z])

                    for idx in indices:
                        if np.allclose(cur_obj_trans_np, ep_info.candidate_objects[idx].position, atol=1e-5):
                            assert idx not in self._candidate_obj_idx_to_rigid_obj_handle.keys(), "While adding rigid objects to sim, same rigid object found multiple times !"
                            self._candidate_obj_idx_to_rigid_obj_handle[idx] = other_obj_handle
                            break
                        
            # Create a mapping between non-interacted candidate object indices and the rigid object handle name as there maybe multiple non-interacted candidate objects with same object_name
            if obj_handle in candidate_obj_noninteracted_names:
                indices = [i for i, x in enumerate(candidate_obj_noninteracted_names) if x == obj_handle]

                if len(indices) > 0:
                    cur_obj_trans = rom.get_object_by_handle(other_obj_handle).transformation.translation
                    cur_obj_trans_np = np.array([cur_obj_trans.x, cur_obj_trans.y, cur_obj_trans.z])

                    for idx in indices:
                        if np.allclose(cur_obj_trans_np, ep_info.candidate_objects_noninteracted[idx].position, atol=1e-5):
                            assert idx not in self._candidate_obj_noninteracted_idx_to_rigid_obj_handle.keys(), "While adding rigid objects to sim, same rigid object found multiple times !"
                            self._candidate_obj_noninteracted_idx_to_rigid_obj_handle[idx] = other_obj_handle
                            break

        if new_scene:
            self._receptacles = self._create_recep_info(
                ep_info.scene_id, list(self._handle_to_object_id.keys())
            )

            ao_mgr = self.get_articulated_object_manager()
            # Make all articulated objects (including the robots) kinematic
            for aoi_handle in ao_mgr.get_object_handles():
                ao = ao_mgr.get_object_by_handle(aoi_handle)
                if self._kinematic_mode:
                    ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                    # remove any existing motors when converting to kinematic AO
                    for motor_id in ao.existing_joint_motor_ids:
                        ao.remove_joint_motor(motor_id)
                self.art_objs.append(ao)