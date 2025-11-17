import cv2
import numpy as np
import warnings
import torch
from argparse import Namespace
from enum import Enum


ANGLE_EPS = 0.001
MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


def convert_pose_to_real_world_axis(hab_pose):
    """Update axis convention of habitat pose to match the real-world axis convention"""
    hab_pose[[0, 1, 2]] = hab_pose[[2, 0, 1]]
    hab_pose[:, [0, 1, 2]] = hab_pose[:, [2, 0, 1]]
    return hab_pose


# OVMM home-robot code requries depth preprocessing
def preprocess_depth(depth):
    rescaled_depth = 0.0 + depth * (10.0 - 0.0)
    rescaled_depth[depth == 0.0] = 10000
    rescaled_depth[depth == 1.0] = 10001
    return rescaled_depth  # [:, :, -1]


def smooth_mask(mask, kernel=None, num_iterations=3):
    """Dilate and then erode.

    Arguments:
        mask: the mask to clean up

    Returns:
        mask: the dilated mask
        mask2: dilated, then eroded mask
    """
    if kernel is None:
        kernel = np.ones((5, 5))
    mask = mask.astype(np.uint8)
    mask1 = cv2.dilate(mask, kernel, iterations=num_iterations)
    # second step
    mask2 = mask
    mask2 = cv2.erode(mask2, kernel, iterations=num_iterations)
    mask2 = np.bitwise_and(mask, mask2)
    return mask1, mask2


def normalize(v):
    return v / np.linalg.norm(v)


def get_angle(x, y):
    """
    Gets the angle between two vectors in radians.
    """
    if np.linalg.norm(x) != 0:
        x_norm = normalize(x)
    else:
        x_norm = x

    if np.linalg.norm(y) != 0:
        y_norm = normalize(y)
    else:
        y_norm = y
    return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))


def get_angle_to_pos(rel_pos: np.ndarray) -> float:
    """
    :param rel_pos: Relative 3D positive from the robot to the target like: `target_pos - robot_pos`.
    :returns: Angle in radians.
    """

    forward = np.array([1.0, 0, 0])
    rel_pos = np.array(rel_pos)
    forward = forward[[0, 2]]
    rel_pos = rel_pos[[0, 2]]

    heading_angle = get_angle(forward, rel_pos)
    c = np.cross(forward, rel_pos) < 0
    if not c:
        heading_angle = -1.0 * heading_angle
    return heading_angle


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    warnings.warn(
        "This function is deprecated and will be removed in future versions. Use the Camera class from src/home_robot/utils/image.py instead.",
        DeprecationWarning,
    )
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    camera_matrix = {"xc": xc, "zc": zc, "f": f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(
        torch.arange(Y_t.shape[-1], device=device),
        torch.arange(Y_t.shape[-2] - 1, -1, -1, device=device),
    )
    grid_x = grid_x.transpose(1, 0)
    grid_z = grid_z.transpose(1, 0)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (
        (grid_x[:, ::scale, ::scale] - camera_matrix.xc)
        * Y_t[:, ::scale, ::scale]
        / camera_matrix.f
    )
    Z_t = (
        (grid_z[:, ::scale, ::scale] - camera_matrix.zc)
        * Y_t[:, ::scale, ::scale]
        / camera_matrix.f
    )

    XYZ = torch.stack((X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))

    return XYZ


def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32,
        )
        R = (
            np.eye(3)
            + np.sin(angle) * S_hat
            + (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
        )
    else:
        R = np.eye(3)
    return R


def transform_camera_view_t(XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = get_r_matrix([1.0, 0.0, 0.0], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(
        XYZ.reshape(-1, 3), torch.from_numpy(R).float().transpose(1, 0).to(device)
    ).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    """Return a mask of all valid depth pixels."""
    return np.bitwise_and(
        depth != MIN_DEPTH_REPLACEMENT_VALUE, depth != MAX_DEPTH_REPLACEMENT_VALUE
    )


class Action:
    """Controls."""

    pass


class ContinuousNavigationAction(Action):
    xyt: np.ndarray

    def __init__(self, xyt: np.ndarray):
        if not len(xyt) == 3:
            raise RuntimeError(
                "continuous navigation action space has 3 dimentions, x y and theta"
            )
        self.xyt = xyt


class ContinuousFullBodyAction:
    xyt: np.ndarray
    joints: np.ndarray

    def __init__(self, joints: np.ndarray, xyt: np.ndarray = None):
        """Create full-body continuous action"""
        if xyt is not None and not len(xyt) == 3:
            raise RuntimeError(
                "continuous navigation action space has 3 dimentions, x y and theta"
            )
        self.xyt = xyt
        # Joint states in robot action format
        self.joints = joints


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_OBJECT = 4
    PLACE_OBJECT = 5
    NAVIGATION_MODE = 6
    MANIPULATION_MODE = 7
    POST_NAV_MODE = 8
    # Arm extension to a fixed position and height
    EXTEND_ARM = 9
    EMPTY_ACTION = 10
    # Simulation only actions
    SNAP_OBJECT = 11
    DESNAP_OBJECT = 12
    # Discrete gripper commands
    OPEN_GRIPPER = 13
    CLOSE_GRIPPER = 14
