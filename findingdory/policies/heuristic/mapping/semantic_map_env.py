# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional, Tuple, Union, cast

import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from .constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
    FindingDoryReceptacleCategories,
)

from .visualizer import Visualizer

import findingdory.policies.heuristic.mapping.interfaces as interfaces


class SemanticMappingEnv:
    '''
    Taken from home-robot/src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/habitat_objectnav_env.HabitatObjectNavEnv
    This class handles preprocessing of raw observations from habitat_sim before sending to the semantic mapper agent/module
        - We need preprocessing because we convert the raw habitat observations to the format that is used by the semantic mapper ported from the home-robot library
    It also wraps the semantic mapping visualizer class
    '''

    def __init__(self, config):
        
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)

        if config.AGENT.SEMANTIC_MAP.semantic_categories == "findingdory_receptacles":
            self.semantic_category_mapping = FindingDoryReceptacleCategories()
        else:
            raise AssertionError("Unsupported category list supplied to semantic mapper !")

        if not self.ground_truth_semantics:
            raise AssertionError("DETIC perception not supported; please use Ground Truth Semantics in config")
 
        self.config = config

    def reset(self):
        
        self.visualizer.reset()
        # scene_id = self.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[
        #     0
        # ]
        # self.visualizer.set_vis_dir(
        #     scene_id, self.habitat_env.current_episode.episode_id
        # )
        # TODO: Right now, we hardcode the mapping visualization directory but set this based on the goal for pick/place during data collection
        self.visualizer.set_vis_dir(
            "random_scene", "random_ep_id"
        )

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> interfaces.Observations:
        
        depth_key = "head_depth"
        rgb_key = "head_rgb"
        
        if self.config.AGENT.SEMANTIC_MAP.semantic_categories == "findingdory_receptacles":
            semantic_key = "receptacle_segmentation"
        else:
            raise AssertionError("Please specify which semantic key is to be used based on the supplied category list for semantic mapping !")
    
        depth = self._preprocess_depth(habitat_obs[depth_key])

        # goal_id, goal_name = None, None         # This was used for objectnav but we dont use in OVMM/memory bench tasks
        goal_id, goal_name = [17], "table"
        
        obs = interfaces.Observations(
            rgb=habitat_obs[rgb_key],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=self._preprocess_xy(habitat_obs["gps"]),
            task_observations={
                "object_goal": goal_id,
                "goal_name": goal_name,
                "recep_goal": None,
            },
            camera_pose=habitat_obs["camera_pose"],
            third_person_image=None,
        )
        obs = self._preprocess_semantic_and_instance(obs, habitat_obs[semantic_key])
        return obs

    def _preprocess_semantic_and_instance(
        self, obs: interfaces.Observations, habitat_semantic: np.ndarray
    ) -> interfaces.Observations:
        
        obs.task_observations["gt_instance_ids"] = habitat_semantic[:, :, -1] + 1
        
        if self.ground_truth_semantics:    
            if type(self.semantic_category_mapping) == FindingDoryReceptacleCategories:
                obs.semantic = habitat_semantic[:, :, -1]
                obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1
                obs.task_observations["semantic_frame"] = obs.rgb

            else:
                raise AssertionError("Unsupported category list supplied to semantic mapper !")
        else:
            raise AssertionError("DETIC perception not supported; please use Ground Truth Semantics in config")
        
        obs.task_observations["instance_map"] = obs.task_observations[
            "instance_map"
        ].astype(int)
        obs.task_observations["gt_instance_ids"] = obs.task_observations[
            "gt_instance_ids"
        ].astype(int)
        obs.task_observations["semantic_frame"] = np.concatenate(
            [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
        ).astype(np.uint8)
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info:
            info["semantic_category_mapping"] = self.semantic_category_mapping
            return self.visualizer.visualize(**info)

    def _preprocess_xy(self, xy: np.array) -> np.array:
        """Translate Habitat navigation (x, y) (i.e., GPS sensor) into robot (x, y)."""
        return np.array([xy[0], -1 * xy[1]])