import numpy as np
import cv2
import json
import random

from gym import spaces

import magnum as mn
import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def iterate_action_space_recursively_with_keys(action_space, action_space_key=""):
    if isinstance(action_space, spaces.Dict):
        for k, v in action_space.items():
            yield from iterate_action_space_recursively_with_keys(v, k)
    else:
        yield action_space, action_space_key


def convert_discrete_actions_to_continuous(action_dict, forward_step_size, turn_angle):
    action = action_dict['action']

    # We only need to do the conversion for discrete actions (which are stored as integers)
    if isinstance(action, int):
        if action == HabitatSimActions.move_forward:
            action_dict = {
                "action": ("base_velocity"),
                "action_args": {
                    "base_vel": np.array(
                        [forward_step_size, 0.0, 0.0]
                    ),
                    "is_last_action": True,
                },
            }

        if action == HabitatSimActions.turn_left:
            action_dict = {
                "action": ("base_velocity"),
                "action_args": {
                    "base_vel": np.array(
                        [0.0, 0.0, turn_angle * np.pi / 180]
                    ),
                    "is_last_action": True,
                },
            }

        if action == HabitatSimActions.turn_right:
            action_dict = {
                "action": ("base_velocity"),
                "action_args": {
                    "base_vel": np.array(
                        [0.0, 0.0, -1 * turn_angle * np.pi / 180]
                    ),
                    "is_last_action": True,
                },
            }

    return action_dict


def save_imagenav_rollouts(rollouts, file_path):
    '''
    Helper function to save the VLM+Imagenav policy rollouts for quick analysis
    '''
    
    frames = rollouts  # Replace with your list of NumPy arrays

    # Get the dimensions of the frames
    height, width, _ = frames[0].shape

    # Define the output video writer
    output_file = file_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    fps = 10
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in frames:
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()


def load_json_file(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def get_closest_dist_to_region(
    sim: habitat_sim.Simulator,
    region_id: str,
    source_point: mn.Vector3,
    max_samples: int = 100,
) -> float:
    """
    Calculate the closest distance from a source point to a given region.

    :param sim: The Simulator instance.
    :param region_id: The ID of the Region to which we want to find the distance.
    :param source_point: The source point from which to calculate the distance.
    :param island_index: The index of the navmesh island representing the active floor area. Default -1 is all islands.
    :param max_samples: The maximum number of points to sample in the region to find the closest one.
    :return: The closest geodesic distance from the source point to the region, or float('inf') if no path exists.
    """
    # Find the region with the matching ID
    region = None
    for potential_region in sim.semantic_scene.regions:
        if potential_region.id == region_id:
            region = potential_region
            break
    
    if region is None:
        return float('inf')  # Return infinity if region not found
    
    # Initialize with infinity
    min_distance = float('inf')
    
    island_index = sim.pathfinder.get_island(source_point)

    # First try the center of the region
    region_center = region.aabb.center
    region_center_snap = sim.pathfinder.snap_point(
        region_center, island_index=island_index
    )
    
    if not np.isnan(region_center_snap[0]):
        # Calculate geodesic distance to the center
        center_distance = sim.geodesic_distance(
            source_point, region_center_snap
        )
        if center_distance != np.inf:
            min_distance = center_distance
    
    # Sample multiple points in the region to find the closest one
    attempts = 0
    closest_point = None
    
    while attempts < max_samples:
        # Try to get a random navigable point
        sample = sim.pathfinder.get_random_navigable_point(
            island_index=island_index
        )
        
        # Check if the point is in the region
        if region.contains(sample):
            # Calculate geodesic distance to this point
            distance = sim.geodesic_distance(
                source_point, sample
            )
            
            # Update minimum distance if this point is closer
            if distance != np.inf and distance < min_distance:
                min_distance = distance
                closest_point = sample
        
        attempts += 1
    
    # After finding the closest point, calculate the path-based distance
    if closest_point is not None:
        # Generate a path from source_point to the closest point
        path = habitat_sim.ShortestPath()
        path.requested_start = source_point
        path.requested_end = closest_point
        path_found = sim.pathfinder.find_path(path)
        
        # Check all points on the path to find the closest point that's in the region
        for point in path.points:
            if region.contains(point):
                # This is the first point on the path that's in the region
                # Interpolate between source_point and point to find precise boundary
                # Calculate total distance between points
                total_distance = np.linalg.norm(point - source_point)
                # Use 10 cm steps for interpolation
                step_size = 0.25
                num_steps = int(total_distance / step_size)
                
                for i in range(num_steps + 1):
                    # Linear interpolation between source_point and point based on distance
                    alpha = i * step_size / total_distance if total_distance > 0 else 0
                    interpolated_point = source_point * (1 - alpha) + point * alpha
                    
                    # Check if interpolated point is in the region
                    if region.contains(interpolated_point):
                        # Calculate the geodesic distance to this boundary point
                        boundary_distance = sim.geodesic_distance(
                            source_point, interpolated_point
                        )
                        if boundary_distance < min_distance:
                            min_distance = boundary_distance
                        break
                break

    return min_distance
