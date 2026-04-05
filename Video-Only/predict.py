from extract_and_sample_frames import sample_frames
from extract_faces_from_video import extract_faces_from_video
# from torchvision import transforms
from transform_augumentation import get_transforms
from Config import CONFIG,DEVICE
from typing import Dict
import torch
import torch.nn as nn

@torch.no_grad()
def predict_video(
    video_path      : str,
    model           : nn.Module,
    frame_sample_count: int = CONFIG["frame_sample_count"],
) -> Dict:
    """
    Run end-to-end inference on a single video.

    Args:
        video_path         : Path to input video file.
        model              : Loaded SwinDeepfakeDetector in eval mode.
        frame_sample_count : Number of frames to sample for prediction.

    Returns:
        dict with keys: label (str), confidence (float), probabilities (list)
    """
    transform = get_transforms(split="val")
    label_names = {0: "REAL", 1: "FAKE"}

    # Step 1: Extract faces
    print(f"[Inference] Processing: {video_path}")
    face_crops = extract_faces_from_video(video_path, use_fallback=True)
    # print(f"[predict_video] face_crops extracted: {len(face_crops)}")  # Siddhu

    # Step 2: Sample
    face_crops = sample_frames(face_crops, n=frame_sample_count)

    # Step 3: Handle edge case
    if len(face_crops) == 0:
        print("[Inference] WARNING: No frames could be processed. Returning UNKNOWN.")
        return {"label": "UNKNOWN", "confidence": 0.0, "probabilities": [0.5, 0.5]}

    # Pad if needed
    while len(face_crops) < frame_sample_count:
        face_crops.append(face_crops[-1])

    # Step 4: Transform & batch
    tensors = [transform(f) for f in face_crops]     # each (3, H, W)
    frames  = torch.stack(tensors).unsqueeze(0)       # (1, T, 3, H, W)
    frames  = frames.to(DEVICE)

    # Step 5: Forward pass
    logits = model(frames)                             # (1, 2)
    probs  = torch.softmax(logits, dim=1)[0]           # (2,)
    pred   = probs.argmax().item()
    conf   = probs[pred].item()

    result = {
        "label"        : label_names[pred],
        "confidence"   : round(conf * 100, 2),
        "probabilities": {
            "REAL": round(probs[0].item() * 100, 2),
            "FAKE": round(probs[1].item() * 100, 2),
        },
    }

    print(f"\n{'='*50}")
    print(f"  VIDEO   : {os.path.basename(video_path)}")
    print(f"  RESULT  : {result['label']}")
    print(f"  CONFIDENCE: {result['confidence']}%")
    print(f"  REAL    : {result['probabilities']['REAL']}%")
    print(f"  FAKE    : {result['probabilities']['FAKE']}%")
    print(f"{'='*50}\n")

    # print(f"[predict_video] logits={logits}  probs={probs}")  # Siddhu
    return result