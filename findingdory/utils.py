import numpy as np
import torch

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector


ACTION_MAPPING = {
    "": "No action",
    HabitatSimActions.stop: "stop", # 0
    HabitatSimActions.move_forward: "move_forward", # 1
    HabitatSimActions.turn_left: "turn_left", # 2
    HabitatSimActions.turn_right: "turn_right", # 3
    4: "manipulation_mode",
    5: "desnap_object",
    6: "empty_action",
    7: "extend_arm",
    8: "full_body_action",
    9: "snap_object",
    10: "navigation_mode",
}


def quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])

    heading_vector = quaternion_rotate_vector(quat, direction_vector)

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array([phi], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")