import copy
import cv2
import math
import os
import time
import ast

from PIL import Image
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from findingdory.policies.llm.vlm_agent import VLMAgent
from findingdory.policies.llm.utils import save_response

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenAgent(VLMAgent):
    """LLM based agent in Habitat environments."""

    def __init__(self, config) -> None:
        """
        :param config: QWEN config
        """
        super().__init__(config)

        self.processor = AutoProcessor.from_pretrained(
            config.model,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()

    def get_vlm_response(self, images, prompt, path=None):
        torch.cuda.empty_cache()
        mm_prompt = None
        if path:
            mm_prompt = path
            mm_type = "video" if path.endswith(".mp4") else "image"
        elif len(images) == 1:
            mm_prompt = self.save_image(images[0])
            mm_type = "image"
        else:
            mm_prompt = self.save_video(images, fps=1)
            mm_type = "video"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": mm_type, mm_type: mm_prompt},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Clear CUDA cache to free up GPU memory
        torch.cuda.empty_cache()

        return output_text

    def run_vlm(self, frames, lang_goal):
        assert len(frames) > 0, "No frames to process"

        prompt = copy.deepcopy(self.prompt)
        prompt = prompt.replace("{goal}", lang_goal)

        # Clear CUDA cache to free up GPU memory
        torch.cuda.empty_cache()

        llm_response = self.get_vlm_response(frames, prompt)
        print("LLM Response: ", llm_response.encode("utf-8"))
        llm_response = self.extract_info_from_response(llm_response)

        save_response(
            str(llm_response),
            self.output_folder_with_episode_index,
            model_name="qwen_agent",
            goal=lang_goal
        )

        return self.extract_frame_indices_from_response(llm_response, frames)

    def load_frames_and_run(self, video_path, lang_goal):
        prompt = copy.deepcopy(self.prompt)
        prompt = prompt.replace("{goal}", lang_goal)
        llm_response = self.get_vlm_response(images=None, prompt=prompt, path=video_path)
        print("LLM Response: ", llm_response.encode("utf-8"))
