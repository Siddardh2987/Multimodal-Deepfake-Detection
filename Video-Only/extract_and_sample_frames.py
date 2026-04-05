from typing import List
from Config import CONFIG
import cv2
import numpy as np
import random
import numpy as np

def extract_frames(
    video_path: str,
    max_frames: int = CONFIG["max_frames_per_video"]
) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(1, total_frames)  # guard divide-by-zero

    # Sample `max_frames` indices spread across the video
    indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    return frames


def sample_frames(
    face_crops: List[np.ndarray],
    n: int = CONFIG["frame_sample_count"],
    strategy: str = "uniform"
) -> List[np.ndarray]:
    """
    Reduce a list of face crops to exactly `n` samples.

    Strategies:
      "uniform"  — evenly spaced indices (default)
      "random"   — random subset
      "first"    — first N frames

    Args:
        face_crops : List of RGB numpy arrays.
        n          : Target number of frames.
        strategy   : Sampling strategy.

    Returns:
        List of up to `n` face crops.
    """
    total = len(face_crops)
    if total == 0:
        return []
    if total <= n:
        return face_crops  # already few enough

    if strategy == "uniform":
        indices = np.linspace(0, total - 1, num=n, dtype=int)
    elif strategy == "random":
        indices = sorted(random.sample(range(total), n))
    elif strategy == "first":
        indices = list(range(n))
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    sampled = [face_crops[i] for i in indices]
    # print(f"[sample_frames] total={total}, sampled={len(sampled)}, strategy={strategy}")  # Siddhu
    return sampled