"""
spectral_bridge_app.py  —  Standalone Deepfake Detector
=================================================================
A portable version of the Spectral Bridge inference script.
Place 'best_model.pt' in the same folder as this script, run it,
and select an image to test!
"""

import os
import sys
import torch
import clip
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image as PILImage
from pathlib import Path
import torch.nn as nn

# ── 1. Rebuild Architecture ──────────────────────────────────────────────────
class DeepfakeMLP(nn.Module):
    def __init__(self, input_dim=612):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.network(x)

# ── 2. FFT Helper ────────────────────────────────────────────────────────────
def radial_fft_profile(pil_img: PILImage.Image, n_bins=100) -> np.ndarray:
    """Computes the 1D Azimuthal Average of the 2D FFT."""
    gray = pil_img.convert("L").resize((224, 224), PILImage.LANCZOS)
    arr = np.array(gray, dtype=np.float32)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    log_mag = np.log1p(fft_mag)
    
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    radius = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    
    max_r = min(cx, cy)
    profile = np.array([
        log_mag[radius == r].mean() if (radius == r).any() else 0.0
        for r in range(max_r)
    ], dtype=np.float32)
    
    x_old = np.linspace(0, 1, len(profile))
    x_new = np.linspace(0, 1, n_bins)
    profile = np.interp(x_new, x_old, profile).astype(np.float32)
    
    rng = profile.max() - profile.min()
    if rng > 0:
        profile = (profile - profile.min()) / rng
    return profile

# ── 3. Main Inference Function ───────────────────────────────────────────────
def predict_image(model_path: Path, image_path: str):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading Model & Extractors (This might take a few seconds)...")
    
    # 1. Load CLIP
    import warnings
    warnings.filterwarnings("ignore")
    clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    clip_model.eval()
    
    # 2. Load trained MLP
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    mlp = DeepfakeMLP(input_dim=ckpt["feat_dim"]).to(DEVICE)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.eval()

    print(f"\nAnalyzing Image: {os.path.basename(image_path)}")
    
    # Load and format image
    try:
        img = PILImage.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    with torch.no_grad():
        # -- Extractor Phase --
        clip_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        clip_feats = clip_model.encode_image(clip_tensor)
        clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
        clip_feats = clip_feats.cpu().float()
        
        fft_feats = torch.from_numpy(radial_fft_profile(img)).unsqueeze(0)
        
        hybrid_vector = torch.cat([clip_feats, fft_feats], dim=1).to(DEVICE)
        
        # -- Classifier Phase --
        logits = mlp(hybrid_vector).squeeze()
        
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
            
        prob = torch.sigmoid(logits).item()

    # -- Output Formatting --
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    if prob >= 0.5:
        confidence = prob * 100
        print(f"  Verdict    : 🤖 AI GENERATED (FAKE)")
        print(f"  Confidence : {confidence:.2f}%")
    else:
        confidence = (1 - prob) * 100
        print(f"  Verdict    : 📸 REAL PHOTOGRAPH")
        print(f"  Confidence : {confidence:.2f}%")
        
    print("="*50 + "\n")

# ── Execute ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # 1. Locate the model
    # First, check the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    model_file = script_dir / "best_model.pt"

    if not model_file.exists():
        # If not there, check the current working directory
        model_file = Path.cwd() / "best_model.pt"

    # If STILL not found, ask the user to locate it manually
    if not model_file.exists():
        print("Could not find 'best_model.pt' automatically.")
        print("Please select the model file in the popup window...")
        found_path = filedialog.askopenfilename(
            title="Locate best_model.pt",
            filetypes=[("PyTorch Model", "*.pt *.pth"), ("All files", "*.*")]
        )
        if not found_path:
            print("❌ No model selected. Exiting.")
            sys.exit(0)
        model_file = Path(found_path)

    # 2. Select the image to test
    print("Waiting for image selection...")
    image_path = filedialog.askopenfilename(
        title="Select an Image to Analyze",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("❌ No image selected. Exiting.")
        sys.exit(0)
        
    # 3. Run Inference
    predict_image(model_file, image_path)
    
    # 4. Keep console open so the user can read the result
    input("Press Enter to exit...")