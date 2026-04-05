import numpy as np
from typing import List
from extract_and_sample_frames import extract_frames
from detect_faces import detect_and_crop_face
from detect_faces import fallback_center_crop

def extract_faces_from_video(
    video_path: str,
    use_fallback: bool = True
) -> List[np.ndarray]:
    """
    Full pipeline: extract frames → detect faces → return list of RGB face crops.

    Args:
        video_path   : Path to video file.
        use_fallback : If True, use center-crop when no face found in a frame.

    Returns:
        List of RGB numpy arrays, one per accepted frame.
    """
    raw_frames = extract_frames(video_path)

    if len(raw_frames) == 0:
        print(f"[WARNING] No frames extracted from: {video_path}")
        return []

    face_crops = []
    no_face_count = 0

    for i, frame in enumerate(raw_frames):
        face = detect_and_crop_face(frame)
        if face is not None:
            face_crops.append(face)
        else:
            no_face_count += 1
            if use_fallback:
                face_crops.append(fallback_center_crop(frame))
            # print(f"[extract_faces_from_video] Frame {i}: no face, fallback used={use_fallback}")  # Siddhu

    # print(f"[extract_faces_from_video] {video_path}: faces={len(face_crops)-no_face_count}, fallbacks={no_face_count}")  # Siddhu
    return face_crops