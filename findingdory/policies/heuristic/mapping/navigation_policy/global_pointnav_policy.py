
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from findingdory.policies.heuristic.mapping.modules.semantic.constants import MapConstants as MC
from findingdory.policies.heuristic.mapping.utils.morphology import binary_dilation, binary_erosion


class GlobalPointNavigationPolicy(nn.Module):
    """
    Policy to select high-level goals for moving to a point goal corresponding to image goal selected by VLM agent
    """

    def __init__(self):
        super().__init__()

    def reach_point_goal(self, global_map, pointnav_goal):
        goal_map = self.reach_goal_in_global_map(global_map, pointnav_goal)
        return goal_map

    @property
    def goal_update_steps(self):
        return 1

    def forward(
        self,
        global_map,
        pointnav_goal,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
        """
        return self.reach_point_goal(global_map, pointnav_goal)

    def reach_goal_in_global_map(
        self,
        global_map,
        pointnav_goal,
        ):
        
        # TODO: assert pointnav goal cooridnates are within global map shape bounds
        
        goal_map = torch.zeros_like(global_map[0,0]).unsqueeze(0).unsqueeze(0)
        x, y = pointnav_goal[0][0].item(), pointnav_goal[0][1].item()
        goal_map[0, 0, y - 2 : y + 3, x - 2 : x + 3] = 1.
        
        return goal_map