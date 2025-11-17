import ast
import cv2
import glob
import json
import json5
import os
import re
import random
import warnings
from PIL import Image
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np
import magnum as mn
from habitat.core.logging import logger

from findingdory.policies.agent import Agent
from findingdory.policies.llm.utils import load_text
from findingdory.dataset.utils import magnum_to_python_quaternion


class VLMAgent(Agent):
    """VLM based agent in Habitat environments."""

    def __init__(self, config) -> None:
        """
        :param config: VLM config
        """
        self.model = None
        self.load_prompt(config.prompt_file)
        self.output_folder = config.output_folder
        self.vlm_query_freq = 5000

        self.vlm_temp_folder = config.vlm_temp_folder

        self._previous_llm_response = {
            "long_term_plan": "",
            "scratchpad": "",
            "timestamps": None,
            "chosen_object": None
        }
        self.chunk_size = config.chunk_size
        
        self.episode_index = 0
        self.task_id = None
        self.action_index = 0
        self._observations = []
        self._agent_states = []
        self._agent_in_nav_mode = []
        self._cur_ep_id = 0

    def load_prompt(self, prompt_path):
        self.prompt = load_text(prompt_path)

    def reset(self):
        self._observations = []
        self._agent_states = []
        self._agent_in_nav_mode = []
        self.action_index = 0
        self.output_folder_with_episode_index = os.path.join(
            self.output_folder, f"ep_{self.episode_index:04d}"
        )
        self._frame_num_to_original_frame_num = None

    def reset_new_task(self, task_id):
        self.action_index = 0
        self.task_id = task_id
        self.output_folder_with_episode_index = os.path.join(
            self.output_folder, f"ep_{self.episode_index:04d}"
        )
        self._frame_num_to_original_frame_num = None

    def act(self, obs):
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        # If the agent is in manipulation mode, the base is turned by +90 in the manipulation mode action step() so we need to correct for that
        if obs["manipulation_mode"]:
            agent_state = obs["agent_state"]
            quat_list = [
                agent_state.rotation.x,
                agent_state.rotation.y,
                agent_state.rotation.z,
                agent_state.rotation.w
            ]
            goal_quat = mn.Quaternion(mn.Vector3(quat_list[:3]), quat_list[3])
            turn_angle = -np.pi / 2  # Turn left by -90 degrees
            rot_quat = mn.Quaternion(
                mn.Vector3(0, np.sin(turn_angle / 2), 0),
                np.cos(turn_angle / 2),
            )
            final_goal_quat = rot_quat * goal_quat
            agent_state.rotation = magnum_to_python_quaternion(final_goal_quat)
        else:
            agent_state = obs["agent_state"]

        if not obs["can_take_action"]:
            self._observations.append(obs)
            self._agent_states.append(agent_state)
            self._agent_in_nav_mode.append(not obs["manipulation_mode"])
            action = self.get_random_action()
        else:
            frames = self.process_frames_for_vlm(save_images=False)
            nav_indices = self.run_vlm(frames, obs['lang_memory_goal'])

            # The predicted VLM subgoals need to be verified for task success -> so we return the predicted subgoals as an "action_dict" and pass it to the task for PDDL verification
            # Clip nav indices to valid range
            clipped_indices = [max(0, min(idx, len(self._frame_num_to_original_frame_num) - 1)) for idx in nav_indices]
            output_nav_indices = [self._frame_num_to_original_frame_num[idx] for idx in clipped_indices]

            logger.info(f"--------------------------> Frame index returned by VLM: {output_nav_indices}")

            action = {
                "action": "high_level_policy_action",
                "action_args": {
                    "nav_indices": output_nav_indices,            # Pass navigation goal information to task
                    "nav_goal_states": [self._agent_states[idx] for idx in clipped_indices] if clipped_indices else [],    # Store corresponding agent states for the selected navigation goals
                    "nav_mode_flag": [self._agent_in_nav_mode[idx] for idx in clipped_indices] if clipped_indices else []
                },
            }
            logger.info(f"Action: {action}")

        self.action_index += 1

        return action

    def get_random_action(self):
        action = random.choice(
            [
                HabitatSimActions.move_forward,
                HabitatSimActions.turn_right,
                HabitatSimActions.turn_left,
            ]
        )

        action_dict = {
            "action": action,
        }
        return action_dict

    def response_json_loads(self, response):
        """Extract structured information from the model's response."""
        try:
            if 'json' in response or ('{' in response and '}' in response):
                # Clean the response string
                cleaned_response = response.strip().replace('json', '').replace('```', '')
 
                # Try to extract JSON by finding the first { and last } in the string
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    cleaned_response = cleaned_response[start_idx:end_idx]
                
                # Try different parsing methods in order of safety
                try:
                    return json5.loads(cleaned_response)
                except Exception as e1:
                    warnings.warn(f"Error parsing with json5: {e1}, trying ast.literal_eval", RuntimeWarning)
                    try:
                        return ast.literal_eval(cleaned_response)
                    except Exception as e2:
                        warnings.warn(f"Error parsing with ast.literal_eval: {e2}", RuntimeWarning)
                        return None
        except Exception as e:
            warnings.warn(f"Failed to extract info from response: {response}. Error: {e}", RuntimeWarning)
            return None

    def process_frames_for_vlm(self, save_images=True):
        images = []

        if save_images:
            # if this folder already has saved images hab_frame_x.jpg from before, delete them first
            os.makedirs(self.output_folder_with_episode_index, exist_ok=True)

            for file in glob.glob(os.path.join(self.output_folder_with_episode_index, "hab_frame_*.jpg")):
                os.remove(file)

        if self._frame_num_to_original_frame_num is None and len(self._observations) > self.chunk_size:
            print(f"Subsampling frames with chunk size {self.chunk_size}")
            
            num_frames = np.linspace(0, len(self._observations)-1, self.chunk_size, dtype=int)
            num_frames = num_frames.tolist()  # Convert NumPy array to a standard Python list
            self._frame_num_to_original_frame_num = num_frames
            self._observations = [self._observations[i] for i in num_frames]
            self._agent_states = [self._agent_states[i] for i in num_frames]
            self._agent_in_nav_mode = [self._agent_in_nav_mode[i] for i in num_frames]
            assert len(self._observations) <= self.chunk_size, \
                f"Subsampled frames length {len(self._observations)} exceeds chunk size {self.chunk_size}"
        elif self._frame_num_to_original_frame_num is None:
            self._frame_num_to_original_frame_num = list(range(len(self._observations)))

        for idx, obs in enumerate(self._observations):
            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(obs['head_rgb'], cv2.COLOR_RGB2BGR)

            # Define texts to display
            frame_text = f"Frame: {idx}"
            time_of_day_text = f"Time of Day: {obs['time_of_day']}"

            # Define font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)  # Red color for text
            thickness = 1

            # Define positions for each text
            frame_position = (10, 50)  # Top-left corner for frame index
            time_of_day_position = (10, 70)  # Below frame index for time of day

            # Add black outline to make text more visible
            outline_color = (0, 0, 0)  # Black color for outline
            outline_thickness = 3

            # Draw text outline first
            cv2.putText(img_bgr, frame_text, frame_position, font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)
            cv2.putText(img_bgr, time_of_day_text, time_of_day_position, font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)

            # Draw main text on top
            cv2.putText(img_bgr, frame_text, frame_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            cv2.putText(img_bgr, time_of_day_text, time_of_day_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

            if save_images:
                # save cv2 image
                cv2.imwrite(os.path.join(self.output_folder_with_episode_index, f"hab_frame_{idx}.jpg"), img_bgr)

            images.append(img_bgr)

        return images

    def save_video(self, frames, fps=1):
        height, width, layers = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        os.makedirs(self.vlm_temp_folder, exist_ok=True)

        output_video_path = os.path.join(self.vlm_temp_folder, 'ep_id_' + str(self.episode_index) + '.mp4')

        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for idx, frame in enumerate(frames):
            # Write frame to video
            video_writer.write(frame)
        # Release the video writer
        video_writer.release()
        
        return output_video_path

    def save_image(self, image):
        os.makedirs(self.vlm_temp_folder, exist_ok=True)
        image_path = os.path.join(self.vlm_temp_folder, 'ep_id_' + str(self.episode_index) + '.jpg')
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(image_path)
        return image_path

    def save_images(self, images, chunk_idx):
        """
        Save a list of images to the vlm_temp_folder
        """
        os.makedirs(self.vlm_temp_folder, exist_ok=True)
        image_paths = []
        for idx, image in enumerate(images):
            image_path = os.path.join(self.vlm_temp_folder, 'ep_id_' + str(self.episode_index) + '_' + str(chunk_idx) + '_' + str(idx) + '.jpg')
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image.save(image_path)
            image_paths.append(image_path)
        return image_paths

    def run_vlm_on_chunk(self, images, prompt):
        response = self.get_vlm_response(images, prompt)

        response = self.validate_response(response)
        if response is None:
            return None
    
        logger.info(f"Response: {response}")
        response = self.extract_info_from_response(response)
        return response

    def extract_info_from_response(self, response):
        # Extract the scratchpad from the response
        try:
            if 'json' in response or ('{' in response and '}' in response):
                return self.response_json_loads(response)
            elif "NUM_TARGETS_TO_REVISIT:" in response and "TIMESTAMP_INDEX:" in response:
                pattern = r"NUM_TARGETS_TO_REVISIT:\s*\|\|(\d+)\|\|.*TIMESTAMP_INDEX:\s*\|\|([\d{2}:\d{2},]+)\|\|"
                match = re.search(pattern, response)
                num_targets_to_revisit = int(match.group(1))

                timestamp_str = match.group(2)
                timestamps = timestamp_str.split(',')
                assert len(timestamps) == num_targets_to_revisit
                response = {
                    "timestamps": timestamps,
                }
                return response
        except Exception as e:
            warnings.warn(f"Error in extracting info from response {response}: {e}", RuntimeWarning)
            return None

    def extract_frame_indices_from_response(self, llm_response, frames):
        if llm_response is not None and "frame_indices" in llm_response:
            frame_indices = llm_response['frame_indices']
            try:
                if Ellipsis in frame_indices:
                    # Qwen outputs ellipsis for some reason, we consider those episodes as failed
                    return [-1]

                frame_indices = [int(t) if int(t) < len(frames) else -1 for t in frame_indices]

                return frame_indices
            except ValueError:
                warnings.warn(f"Invalid frame index found in response: {frame_indices}", RuntimeWarning)
                return [-1]
        else:
            warnings.warn(f"No frame index found in response: {llm_response}", RuntimeWarning)
            return [-1]

    # Helper Functions to run the agent offline
    def load_frames_and_run(self, folder_path, lang_goal):
        frames = []
        images = glob.glob(os.path.join(folder_path, "hab_frame_*.jpg"))
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file in images:
            frames.append(cv2.imread(file))

        timestamp_in_seconds = self.run_vlm(frames, lang_goal)

    def set_vlm_high_level_success_metrics(self, nav_indices, task):
        '''Function to set the various flags to track VLM agent failure modes'''

        # Track an invalid response from VLM
        if len(nav_indices) == 1 and nav_indices[0] == -1:
            task._high_level_goal_success = False
            task._high_level_goal_assigned = True
            task._vlm_response_error = True
            logger.info("Episode failed as VLM generated an invalid response !")
                        
        # Check if the VLM predicted the correct number of targets               
        elif task._num_actual_targets != len(nav_indices):
            task._high_level_goal_success = False
            task._high_level_goal_assigned = True
            task._vlm_num_targets_error = True
            logger.info("Episode failed as VLM predicted incorrect number of goals !")
        
        # Iterate over each index in self.nav_indices to check for out-of-bound index, otherwise add the goal frames for low level navigation policy
        else:
            for index in self.nav_indices:
                if index > len(self._observations) or index < 0:
                    # If any index is out-of-bounds, mark the task as unsuccessful and handle errors
                    task._high_level_goal_success = False
                    task._high_level_goal_assigned = True
                    task._out_of_bounds_pred = True
                    logger.info("Episode failed as VLM predicted out-of-bounds frame index!")

        if not task._high_level_goal_assigned:
            goal_states = []
            for index in nav_indices:
                goal_states.append(self._agent_states[index])
            task.assign_and_validate_high_level_goal(goal_states)
        
            logger.info(f"--------------------------> Did VLM locate the image frame successfully ?: {task._high_level_goal_success}")
