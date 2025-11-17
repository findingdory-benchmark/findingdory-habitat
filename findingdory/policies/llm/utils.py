import os
import cv2
import json
from io import BytesIO
import numpy as np
from PIL import Image

from IPython.display import Image as VImage


def np_array_to_vimage(
    img: np.ndarray,
    idx: int,
    tmp_dir: str
):
    os.makedirs(tmp_dir, exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_path = os.path.join(tmp_dir, f"hab_frame_{idx}.jpg")
    cv2.imwrite(img_path, img)
    img_vertex = VImage(
        filename=img_path,
        height=img.shape[0],
        width=img.shape[1]
    )
    return img_vertex


def save_frames(image_folder, frames):
    """
    Save frames to a folder.
    """
    os.makedirs(image_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = os.path.join(image_folder, f"hab_frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)


def load_text(text_file):
    """
    Load the text from a file.
    """
    with open(text_file, "r") as f:
        return f.read().strip()


def save_response(response, folder, model_name="vlm", goal=None):
    """
    Save the response to a file.
    """
    os.makedirs(folder, exist_ok=True)
    output_file = os.path.join(folder, f"{model_name}_analysis_{goal}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)

    print(f"VLM analysis saved to {folder}")
