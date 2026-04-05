import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from Config import CONFIG,DEVICE
import os

def save_checkpoint(
    model     : nn.Module,
    optimizer : optim.Optimizer,
    epoch     : int,
    val_acc   : float,
    path      : str,
) -> None:
    """Save training checkpoint."""
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc"             : val_acc,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(
    model     : nn.Module,
    optimizer : Optional[optim.Optimizer],
    path      : str,
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, float]:
    """
    Load a checkpoint.  Returns (model, optimizer, start_epoch, best_val_acc).
    Pass optimizer=None for inference-only loading.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch   = ckpt.get("epoch", 0) + 1
    best_val_acc  = ckpt.get("val_acc", 0.0)

    print(f"[Checkpoint] Loaded from '{path}'  epoch={start_epoch-1}  val_acc={best_val_acc:.4f}")
    return model, optimizer, start_epoch, best_val_acc
