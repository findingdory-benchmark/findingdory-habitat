#!/usr/bin/env python3
import os
import random
import time
import warnings
import ast

import torch
from findingdory.policies.end_to_end.qwen_imagenav_agent import QwenImageNavAgent
import cv2

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from findingdory.policies.llm.utils import save_response


class QwenTrainedImageNavAgent(QwenImageNavAgent):
    """LLM based agent in Habitat environments."""
    def __init__(self, config) -> None:
        """
        :param config: QWEN config
        """
        super().__init__(config)

        self.processor = AutoProcessor.from_pretrained(
            config.model,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        
        self._output_offset_fix = config.output_offset_fix

    def get_vlm_response(self, images, prompt, path=None):
        torch.cuda.empty_cache()
        mm_prompt = None
        if path:
            mm_prompt = path
            mm_type = "video" if path.endswith(".mp4") else "image"
        else:
            mm_prompt = self.save_video(images, fps=1)
            mm_type = "video"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": mm_type, mm_type: mm_prompt, "max_pixels": 360 * 420, "fps": 1.0},
                    {"type": "text", "text": f"The robot's goal is: {prompt}"},
                ]
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False
        )
        inputs = self.processor(
            text=[text],
            videos=process_vision_info(messages)[1][0],
            return_tensors="pt",
        ).to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.decode(
            generated_ids_trimmed[0], skip_special_tokens=True
        )
        # Clear CUDA cache to free up GPU memory
        torch.cuda.empty_cache()

        return output_text

    def extract_info_from_response(self, response):
        # Extract the scratchpad from the response
        if "assistant\n" in response:
            response = response.split("assistant\n", 1)[1].strip()
        try:
            return ast.literal_eval(response)
        except Exception as e:
            warnings.warn(f"Error in extracting info from response {response}: {e}", RuntimeWarning)
            return None

    def extract_frame_indices_from_response(self, llm_response, frames):
        if llm_response is not None:
            try:
                # output should be a list of lists, each inner list contains the possible frame indices for a single goal
                # hardcoded to use the first frame index for each goal since it works the best
                if self._output_offset_fix:
                    frame_indices = [max(0, frames[0] - 1) if len(frames) > 0 else -1 for frames in llm_response]
                    print("Frame indices after subtracting by 1: ", frame_indices)
                else:
                    frame_indices = [frames[0] if len(frames) > 0 else -1 for frames in llm_response]
                return frame_indices, None, None
            except ValueError:
                warnings.warn(f"Invalid frame index found in response: {frame_indices}", RuntimeWarning)
                return [-1], None, None
        else:
            warnings.warn(f"No frame index found in response: {llm_response}", RuntimeWarning)
            return [-1], None, None

    def run_vlm(self, frames, lang_goal):
        assert len(frames) > 0, "No frames to process"

        llm_response = self.get_vlm_response(frames, lang_goal)
        print("LLM Response: ", llm_response.encode("utf-8"))
        llm_response = self.extract_info_from_response(llm_response)

        save_response(
            str(llm_response),
            self.output_folder_with_episode_index,
            model_name="qwen_agent",
            goal=lang_goal
        )

        return self.extract_frame_indices_from_response(llm_response, frames)

    def process_frames_for_vlm(self, save_images=True):
        # can be removed once the idx on the images during training are fixed to be 0-indexed
        images = []

        if self._frame_num_to_original_frame_num is None and len(self._observations) > self.chunk_size:
            num_frames = np.linspace(0, len(self._observations)-1, self.chunk_size, dtype=int)
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
            frame_text = f"Frame: {self._frame_num_to_original_frame_num[idx]}"
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