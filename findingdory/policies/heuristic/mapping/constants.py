# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


# Color constants we use.
# Note: originally from Habitat
# from habitat_sim.utils.common import d3_40_colors_rgb
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


class SemanticCategoryMapping(ABC):
    """
    This class contains a mapping from semantic and goal category IDs provided by
    a Habitat environment to category IDs stored in the semantic map, as well as
    the color palettes and legends to visualize these categories.
    """

    def __init__(self, goal_id_to_goal_name: Dict[int, str]):
        self.goal_id_to_goal_name = goal_id_to_goal_name
        for gid, gname in self.goal_id_to_goal_name.items():
            self.goal_name_to_goal_id[gname] = gid

    @abstractmethod
    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        pass

    @abstractmethod
    def reset_instance_id_to_category_id(self, env):
        """Reset instance id. Env should be a simulation environment."""
        pass

    @property
    @abstractmethod
    def instance_id_to_category_id(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def map_color_palette(self):
        pass

    @property
    @abstractmethod
    def frame_color_palette(self):
        pass

    @property
    @abstractmethod
    def categories_legend_path(self):
        pass

    @property
    @abstractmethod
    def num_sem_categories(self):
        pass

    @property
    def num_sem_obj_categories(self):
        return self.num_sem_categories()


class PaletteIndices:
    """
    Indices of different types of maps maintained in the agent's map state.
    """

    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    BEEN_CLOSE = 6
    SHORT_TERM_GOAL = 7
    BLACKLISTED_TARGETS_MAP = 8
    INSTANCE_BORDER = 9
    SEM_START = 10


########################################################################################
# HSSD FindingDory semantic category mapping

findingdory_receptacle_cats = {
    'bathtub': 0+1,
    'bed': 1+1,
    'bench': 2+1,
    'cabinet': 3+1,
    'chair': 4+1,
    'chest_of_drawers': 5+1,
    'couch': 6+1,
    'counter': 7+1,
    'filing_cabinet': 8+1,
    'hamper': 9+1,
    'serving_cart': 10+1,
    'shelves': 11+1,
    'shoe_rack': 12+1,
    'sink': 13+1,
    'stand': 14+1,
    'stool': 15+1,
    'table': 16+1,
    'toilet': 17+1,
    'trunk': 18+1,
    'wardrobe': 19+1,
    'washer_dryer': 20+1,
    'unknown': 21+1,
}

findingdory_recep_cat_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
    ]
] + list(d3_40_colors_rgb[1:23].flatten())

findingdory_recep_cat_frame_color_palette = findingdory_recep_cat_color_palette + [
    255,
    255,
    255,
]

class FindingDoryReceptacleCategories(SemanticCategoryMapping):
    """
    Mapping from category IDs in HM3D ObjectNav scenes/episodes to COCO indoor
    category IDs.
    """

    def __init__(self):
        self.goal_name_to_goal_id: Dict[str, int] = {}
        self.goal_id_to_goal_name = {idx: name for name, idx in findingdory_receptacle_cats.items()}
        self._instance_id_to_category_id = None
        super().__init__(self.goal_id_to_goal_name)

    def map_goal_id(self, goal_id: int) -> Tuple[int, str]:
        raise NotImplementedError

    def reset_instance_id_to_category_id(self, env):

        # NOTE: For HSSD findingdory dataset, this function does not do any meaningful operation
        # We directly use receptacle/object segmentation sensors to get the semantic ID information (covnerted from instance ID)
        # This function was required for HM3D datasets to convert instance IDs to category IDs on the fly
        pass

    @property
    def instance_id_to_category_id(self) -> np.ndarray:
        return self._instance_id_to_category_id

    @property
    def map_color_palette(self):
        return findingdory_recep_cat_color_palette

    @property
    def frame_color_palette(self):
        return findingdory_recep_cat_frame_color_palette

    @property
    def categories_legend_path(self):
        return 'findingdory/policies/heuristic/mapping/hssd_recep_legend.png'

    @property
    def num_sem_categories(self):
        return 21