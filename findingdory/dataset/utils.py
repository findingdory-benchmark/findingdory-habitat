import cv2
import imageio
import json
import os

import numpy as np
from PIL import Image
import quaternion

from habitat.utils.visualizations import maps
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.tasks.rearrange.utils import set_agent_base_via_obj_trans
from habitat.tasks.rearrange.utils import get_aabb

from findingdory.utils import quat_to_xy_heading


def coeff_to_yaw(coeff):
    return quat_to_xy_heading(quaternion_from_coeff(coeff).inverse())[0]

def get_agent_yaw(agent_state):
    agent_rot = agent_state.rotation
    agent_rot = quat_to_xy_heading(agent_rot.inverse())[0]
    
    # Make sure angle returned is in the range 0 to 2 * np.pi
    agent_rot = agent_rot % (2 * np.pi)
    
    return agent_rot

def teleport_agent_to_state(sim, target_orientation, target_position):
    
    if not isinstance(target_orientation, float) or isinstance(target_orientation, np.float32):
        target_orientation = coeff_to_yaw(target_orientation)

    if target_orientation < 0:
        target_orientation += 2 * np.pi
        
    # NOTE: We need to add PI/2 for the Stretch embodiment as the robot front is offset by PI/2 in URDF
    urdf_target_orientation = (target_orientation + np.pi / 2) % (2 * np.pi)

    set_agent_base_via_obj_trans(target_position, urdf_target_orientation, sim.articulated_agent)
    
    cur_agent_state = sim.get_agent_state()
    cur_position = np.array(
        [
            cur_agent_state.position.x,
            cur_agent_state.position.y,
            cur_agent_state.position.z
        ]
    )

    cur_orientation = get_agent_yaw(cur_agent_state)
    
    try:
        assert np.allclose(cur_orientation, target_orientation, atol=1e-5), \
            "Agent orientation doesn't match with viewpoint orientation after teleport operation!"
            
        assert np.allclose(cur_position, target_position, atol=1e-5), \
            "Agent position doesn't match with viewpoint position after teleport operation!"
    except:
        breakpoint()
        

def teleport_agent_to_state_quat_assert(sim, target_orientation, target_position):
    '''
    Teleportation to a target position/orientation based on an explicit quaternion comparison instead of yaw comparison
    '''
    
    viewpoint_rot = coeff_to_yaw(target_orientation)
    if viewpoint_rot < 0:
        viewpoint_rot += 2 * np.pi
        
    # NOTE: We need to add PI/2 for the Stretch embodiment as the robot front is offset by PI/2 in URDF
    target_rot = (viewpoint_rot + np.pi / 2) % (2 * np.pi)

    set_agent_base_via_obj_trans(target_position, target_rot, sim.articulated_agent)
    
    cur_pos = np.array(
        [
            sim.get_agent_state().position.x,
            sim.get_agent_state().position.y,
            sim.get_agent_state().position.z
        ]
    )

    cur_orient = np.array(
        [
            sim.get_agent_state().rotation.x,
            sim.get_agent_state().rotation.y,
            sim.get_agent_state().rotation.z,
            sim.get_agent_state().rotation.w
        ]
    )
    
    assert np.allclose(cur_orient, target_orientation, atol=1e-5) or np.allclose(-cur_orient, target_orientation, atol=1e-5), \
        "Agent orientation doesn't match with viewpoint orientation after teleport operation!"
        
    assert np.allclose(cur_pos, target_position, atol=1e-5), \
        "Agent position doesn't match with viewpoint position after teleport operation!"

            
def create_top_down_image(
    top_down_map, desired_height, desired_width, border_color=(255, 255, 255)
):
    """
    Resize an image while maintaining aspect ratio and adding borders if necessary.

    Args:
        top_down_map: The image to resize.
        desired_height: The desired height of the final image.
        desired_width: The desired width of the final image.
        border_color: The color of the border to fill the extra space (default is black).

    Returns:
        The resized image with borders to match the desired dimensions.
    """
    # Get the original dimensions
    image = maps.colorize_draw_agent_and_fit_to_height(top_down_map, desired_height)
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratios
    original_aspect = original_width / original_height
    desired_aspect = desired_width / desired_height

    # Determine the scaling factor and resize the image
    if original_aspect > desired_aspect:
        # Image is wider than the desired aspect ratio
        scale_factor = desired_width / original_width
    else:
        # Image is taller than the desired aspect ratio
        scale_factor = desired_height / original_height

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the desired dimensions and fill it with a border color
    new_image = np.full(
        (desired_height, desired_width, 3), border_color, dtype=np.uint8
    )

    # Calculate the position to place the resized image
    x_offset = (desired_width - new_width) // 2
    y_offset = (desired_height - new_height) // 2

    # Place the resized image in the center of the new image
    new_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return new_image


def create_final_image(
    observations,
    goal_name,
    top_down_map,
    classes_in_frame,
    recep_classes_in_frame,
    other_classes_in_frame,
    robot_region_ids,
    sem_map_frame=None,
    sem_map_vis=None,
    obstacle_map_vis=None,
    time_of_day="",
    action="",
    robot_pose=[],
    map_height=(320),
):
    rgb = observations["head_rgb"]
    third_rgb = observations["third_rgb"]
    recep_segment = observations["receptacle_segmentation"]
    object_segment = observations["all_object_segmentation"]
    other_object_segment = observations["other_object_segmentation"]

    # combine segments along channels
    segments = (
        np.concatenate([recep_segment, other_object_segment, object_segment], axis=2)
        * 255
    ).astype(np.uint8)

    # write this information on a new image
    # create a new image with the same size as the rgb image
    new_image = np.ones((map_height, 2*rgb.shape[1], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    y_delta = 15
    x_delta = 125
    object_rows = 6

    y_position = y_delta
    x_position = 10
    new_image = cv2.putText(
        new_image,
        f"Goal: {goal_name}",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    y_position += y_delta
    new_image = cv2.putText(
        new_image,
        f"Action: {action}",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    y_position += y_delta
    new_image = cv2.putText(
        new_image,
        f"Time of Day: {time_of_day}",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    y_position = y_delta
    x_position = x_delta * 2
    new_image = cv2.putText(
        new_image,
        f"Robot Region IDs: {str(robot_region_ids)}",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    y_position += y_delta
    new_image = cv2.putText(
        new_image,
        f"Robot Position: {str(np.round(robot_pose[:2], 2))} m",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    y_position += y_delta
    new_image = cv2.putText(
        new_image,
        f"Robot Orientation: {str(np.round(robot_pose[2], 2))} deg",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    y_position += y_delta * 2
    x_position = 10
    new_image = cv2.putText(
        new_image,
        f"Objects:",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for i, class_name in enumerate(classes_in_frame):
        y_position += y_delta
        if i != 0 and i % object_rows == 0:
            y_position -= y_delta * object_rows
            x_position += x_delta
        new_image = cv2.putText(
            new_image,
            class_name,
            (x_position, y_position),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    y_position = y_delta * (5 + object_rows + 2)
    x_position = 10
    new_image = cv2.putText(
        new_image,
        f"Receptacles:",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for i, class_name in enumerate(recep_classes_in_frame):
        y_position += y_delta
        if i != 0 and i % object_rows == 0:
            y_position -= y_delta * object_rows
            x_position += x_delta
        new_image = cv2.putText(
            new_image,
            class_name,
            (x_position, y_position),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    y_position = y_delta * (5 + object_rows + 2)
    x_position = x_delta * 2
    new_image = cv2.putText(
        new_image,
        f"Other Objects:",
        (x_position, y_position),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for i, class_name in enumerate(other_classes_in_frame):
        y_position += y_delta
        new_image = cv2.putText(
            new_image,
            class_name,
            (x_position, y_position),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if top_down_map is not None:
        top_down_map = create_top_down_image(top_down_map, map_height, rgb.shape[1])
        has_top_down_map = True
    else:
        has_top_down_map = False
    
    top_image = np.concatenate([rgb, segments, third_rgb], axis=1)    

    legend = cv2.imread(os.path.join(os.path.dirname(__file__), "assets", "hssd_recep_legend.png"))

    if sem_map_vis is None and sem_map_frame is None:        
        sem_map_vis = np.ones((480, 480, 3), dtype=np.uint8) * 255
        sem_map_frame = np.ones((480, 360, 3), dtype=np.uint8) * 255
    
    if obstacle_map_vis is None:
        obstacle_map_vis = np.ones((480, 480, 3), dtype=np.uint8) * 255

    if map_height > sem_map_vis.shape[0] or map_height > obstacle_map_vis.shape[0]:
        bottom_height = map_height
    else:
        bottom_height = max(sem_map_vis.shape[0], obstacle_map_vis.shape[0])
    
    text_details_width = 640
    total_bottom_width = text_details_width
    if has_top_down_map:
        total_bottom_width += top_down_map.shape[1]
    total_bottom_width += sem_map_frame.shape[1] + sem_map_vis.shape[1] + obstacle_map_vis.shape[1]
    
    bottom_image = np.ones((bottom_height, total_bottom_width, 3), dtype=np.uint8) * 255

    bottom_image[:new_image.shape[0], :new_image.shape[1], :] = new_image
    
    current_width = text_details_width
    if has_top_down_map:
        bottom_image[:top_down_map.shape[0], current_width:current_width + top_down_map.shape[1], :] = top_down_map
        current_width += top_down_map.shape[1]
        
    bottom_image[:sem_map_frame.shape[0], current_width:current_width + sem_map_frame.shape[1], :] = sem_map_frame
    current_width += sem_map_frame.shape[1]
    
    bottom_image[:sem_map_vis.shape[0], current_width:current_width + sem_map_vis.shape[1], :] = sem_map_vis
    current_width += sem_map_vis.shape[1]
    
    bottom_image[:obstacle_map_vis.shape[0], current_width:current_width + obstacle_map_vis.shape[1], :] = obstacle_map_vis
    
    bottom_image[new_image.shape[0] + 10: new_image.shape[0] + 10 + legend.shape[0], : legend.shape[1], :] = legend
    
    target_width = bottom_image.shape[1]
    current_width = top_image.shape[1]
    total_padding = target_width - current_width

    # Ensure that the padding is split equally on both sides
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    # Pad the image A to the desired width
    top_image = np.pad(top_image, ((0, 0), (left_padding, right_padding), (0, 0)), mode='constant', constant_values=255)

    combined_image = np.concatenate(
        [
            top_image,
            bottom_image,
        ],
        axis=0,
    )

    return combined_image


def save_images(images, folder):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(folder, f"{i:04d}.jpg")
        Image.fromarray(img).convert("RGB").save(path)


def save_gif(images, folder):
    gif_path = os.path.join(folder, "trajectory.gif")
    with imageio.get_writer(gif_path, mode="I") as writer:
        for image in images:
            writer.append_data(image)

def save_mp4(images, folder, fps=30):
    if not os.path.exists(folder):    
        os.makedirs(folder, exist_ok=True)
    mp4_path = os.path.join(folder, "trajectory.mp4")
    with imageio.get_writer(mp4_path, fps=fps, codec='libx264') as writer:
        for image in images:
            writer.append_data(image)


class MetaDataSaver:
    def __init__(self):
        self.meta_data = []

    def add(
        self,
        idx,
        goal_name,
        classes_in_frame,
        recep_classes_in_frame,
        other_classes_in_frame,
        robot_region_ids,
        time_of_day="",
        action="",
        robot_pos=[],
    ):
        self.meta_data.append(
            {
                "Index": str(idx),
                "Goal": goal_name,
                "Objects": str(classes_in_frame),
                "Receptacles": str(recep_classes_in_frame),
                "Other Objects": str(other_classes_in_frame),
                "Robot in Regions": str(robot_region_ids),
                "Time of Day": time_of_day,
                "Action": action,
                "Robot Pose": str(np.round(robot_pos, 2)),
            }
        )

    def add_list(self, data_list):
        self.meta_data.extend(data_list)

    def clear(self):
        self.meta_data = []

    def merge_metadata(self):
        merged_metadata = []
        temp_entry = None
        for entry in self.meta_data:
            if temp_entry is None:
                temp_entry = entry.copy()
                temp_entry["Action"] = [temp_entry["Action"]]
                temp_entry["Index"] = [temp_entry["Index"]]
                end_time = entry["Time of Day"]
                end_pos = entry["Robot Pose"]
            else:
                if (
                    entry["Goal"] == temp_entry["Goal"]
                    and entry["Objects"] == temp_entry["Objects"]
                    and entry["Receptacles"] == temp_entry["Receptacles"]
                    and entry["Other Objects"] == temp_entry["Other Objects"]
                    and entry["Robot in Regions"] == temp_entry["Robot in Regions"]
                ):

                    temp_entry["Action"].append(entry["Action"])
                    temp_entry["Index"].append(entry["Index"])
                    end_time = entry["Time of Day"]
                    end_pos = entry["Robot Pose"]
                else:
                    temp_entry["Action"] = " | ".join(temp_entry["Action"])
                    temp_entry["Index"] = " | ".join(map(str, temp_entry["Index"]))
                    temp_entry["Time of Day"] = (
                        f"{temp_entry['Time of Day']} - {end_time}"
                        if end_time != temp_entry["Time of Day"]
                        else temp_entry["Time of Day"]
                    )
                    temp_entry["Robot Pose"] = (
                        f"{temp_entry['Robot Pose']} - {end_pos}"
                        if end_pos != temp_entry["Robot Pose"]
                        else temp_entry["Robot Pose"]
                    )
                    merged_metadata.append(temp_entry)
                    temp_entry = entry.copy()
                    temp_entry["Action"] = [temp_entry["Action"]]
                    temp_entry["Index"] = [temp_entry["Index"]]
                    end_time = entry["Time of Day"]
                    end_pos = entry["Robot Pose"]

        if temp_entry is not None:
            temp_entry["Action"] = " | ".join(temp_entry["Action"])
            temp_entry["Index"] = " | ".join(map(str, temp_entry["Index"]))
            temp_entry["Time of Day"] = (
                f"{temp_entry['Time of Day']} - {end_time}"
                if end_time != temp_entry["Time of Day"]
                else temp_entry["Time of Day"]
            )
            temp_entry["Robot Pose"] = (
                f"{temp_entry['Robot Pose']} - {end_pos}"
                if end_pos != temp_entry["Robot Pose"]
                else temp_entry["Robot Pose"]
            )
            merged_metadata.append(temp_entry)

        return merged_metadata

    def save(self, folder):
        merged_metadata = self.merge_metadata()
        # save meta_data.json
        meta_data_path = os.path.join(folder, "meta_data.json")
        with open(meta_data_path, "w") as f:
            json.dump(merged_metadata, f, indent=4)
        self.clear()

def convert_to_ordinal(num):
    ordinals = {
        0: "first",
        1: "second",
        2: "third",
        3: "fourth",
        4: "fifth",
        5: "sixth",
        6: "seventh",
        7: "eighth",
        8: "ninth",
        9: "tenth",
    }
    return ordinals.get(num, str(num) + "th")

def generate_order_string(order_list):
    ordinal_list = [convert_to_ordinal(num) for num in order_list]
    return f"The order to revisit them is: {', '.join(ordinal_list)}"

# Calculate the areas of all faces of the bounding box
def calculate_face_areas(self, points):
    
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    # XY planes: Use points p0, p1, p2, p3 for one face, and p4, p5, p6, p7 for the opposite face
    area_xy_1 = distance(points[0], points[1]) * distance(points[1], points[2])
    area_xy_2 = distance(points[4], points[5]) * distance(points[5], points[6])
    
    # XZ planes: Use points p0, p1, p5, p4 for one face, and p3, p2, p6, p7 for the opposite face
    area_xz_1 = distance(points[0], points[1]) * distance(points[1], points[5])
    area_xz_2 = distance(points[3], points[2]) * distance(points[2], points[6])
    
    # YZ planes: Use points p0, p3, p7, p4 for one face, and p1, p2, p6, p5 for the opposite face
    area_yz_1 = distance(points[0], points[3]) * distance(points[3], points[7])
    area_yz_2 = distance(points[1], points[2]) * distance(points[2], points[6])
    
    return [area_xy_1, area_xy_2, area_xz_1, area_xz_2, area_yz_1, area_yz_2]
    
def check_if_obj_fits_on_recep(self, target_recep, picked_obj):
    
    length = np.abs(target_recep.bounds.back_top_left.x - target_recep.bounds.back_top_right.x)
    width = np.abs(target_recep.bounds.back_top_left.z - target_recep.bounds.front_top_left.z)
    receptacle_top_area = length * width
    
    object_aabb = get_aabb(picked_obj.object_id, self._env.sim, transformed=True)
    
    object_corners = [
        object_aabb.back_bottom_left,
        object_aabb.back_bottom_right,
        object_aabb.front_bottom_left,
        object_aabb.front_bottom_right,
        object_aabb.back_top_left,
        object_aabb.back_top_right,
        object_aabb.front_top_left,
        object_aabb.front_top_right,
    ]
    
    face_areas = self.calculate_face_areas(object_corners)
    largest_area = max(face_areas)

    if receptacle_top_area / 3 > largest_area:
        return True
    else:
        return False
    
def magnum_to_python_quaternion(magnum_quat):
    # Extract components from magnum.Quaternion
    x, y, z = magnum_quat.vector.x, magnum_quat.vector.y, magnum_quat.vector.z
    w = magnum_quat.scalar

    # Create a quaternion object using quaternion package
    python_quat = quaternion.quaternion(w, x, y, z)
    return python_quat