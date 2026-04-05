from Config import CONFIG
from checkpoint import load_checkpoint
from build_model_and_class import build_model
import torch.nn as nn

def load_model_for_inference(checkpoint_path: str = CONFIG["best_model_path"]) -> nn.Module:
    """
    Load model weights for inference.

    Args:
        checkpoint_path : Path to .pth checkpoint.

    Returns:
        model in eval() mode.
    """
    model = build_model()
    model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
    model.eval()
    print(f"[Inference] Model loaded from '{checkpoint_path}'")
    return model
