from Config import CONFIG, DEVICE
import MTCNN
import numpy as np
import cv2
from PIL import Image
from typing import Optional
import torch


_mtcnn = MTCNN(
    image_size=CONFIG["face_size"],
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, CONFIG["min_face_confidence"]],
    device=DEVICE,
    keep_all=False,          # keep only the largest / most-confident face
    post_process=False,      # return uint8 tensor, not normalised float
)


def detect_and_crop_face(
    frame_bgr: np.ndarray,
    output_size: int = CONFIG["face_size"]
) -> Optional[np.ndarray]:
    
    # Convert BGR → RGB PIL image (MTCNN expects RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)

    try:
        face_tensor = _mtcnn(pil_img)           # Tensor [3, H, W] or None
    except Exception as e:
        # print(f"[detect_and_crop_face] MTCNN exception: {e}")  # Siddhu
        return None

    if face_tensor is None:
        # print("[detect_and_crop_face] No face detected.")  # Siddhu
        return None

    # Convert tensor [3, H, W] → numpy [H, W, 3]
    face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    if face_np.shape[0] != output_size or face_np.shape[1] != output_size:
        face_np = cv2.resize(face_np, (output_size, output_size))

    return face_np


def fallback_center_crop(
    frame_bgr: np.ndarray,
    output_size: int = CONFIG["face_size"]
) -> np.ndarray:
    """
    When no face is detected, fall back to a center crop of the frame.
    Useful so the video is not silently discarded.
    """
    h, w = frame_bgr.shape[:2]
    side  = min(h, w)
    top   = (h - side) // 2
    left  = (w - side) // 2
    crop  = frame_bgr[top:top+side, left:left+side]
    crop  = cv2.resize(crop, (output_size, output_size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)