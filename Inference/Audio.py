import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION (Hardcoded for App)
# ==========================================
SAMPLE_RATE       = 16000
CHUNK_LENGTH      = 64000  # 4 seconds
CHUNK_STEP        = 32000  # 2 seconds
MAX_CHUNKS        = 20
MIN_CHUNK_SAMPLES = 16000  # 1 second

# ==========================================
# 2. AUDIO PREPROCESSING (From original utils.py)
# ==========================================
def preprocess_chunk(waveform_chunk, device):
    """
    Convert a raw waveform chunk to a mel spectrogram tensor.
    Matches training preprocessing exactly.
    """
    # Pad or trim to exact length
    if waveform_chunk.shape[1] < CHUNK_LENGTH:
        pad_length     = CHUNK_LENGTH - waveform_chunk.shape[1]
        waveform_chunk = F.pad(waveform_chunk, (0, pad_length))
    else:
        waveform_chunk = waveform_chunk[:, :CHUNK_LENGTH]

    # Normalize waveform
    waveform_chunk = waveform_chunk / (waveform_chunk.abs().max() + 1e-8)

    # Mel spectrogram setup
    mel_transform = T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft       = 1024,
        hop_length  = 512,
        n_mels      = 64,
        f_min       = 0.0,
        f_max       = 8000.0,
    ).to(device)

    # Apply transforms
    waveform_chunk = waveform_chunk.to(device)
    mel_spec       = mel_transform(waveform_chunk)
    mel_spec       = T.AmplitudeToDB()(mel_spec)
    mel_spec       = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    # Add batch dimension -> [1, 1, 64, T]
    mel_spec       = mel_spec.unsqueeze(0)   

    return mel_spec

# ==========================================
# 3. STUDENT CNN ARCHITECTURE
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se   = SEBlock(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.se(self.block(x)) + x)

class MobileStudentCNN(nn.Module):
    def __init__(self):
        super(MobileStudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.res_block     = ResBlock(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res_block(x)
        x = self.adaptive_pool(x)
        return self.fc(x)

# ==========================================
# 4. INFERENCE ENGINE
# ==========================================
def load_model(model_path, device):
    student = MobileStudentCNN().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()
    return student

def run_inference(filepath, student, device, save_plots=True):
    # waveform, sr = torchaudio.load(filepath, backend="soundfile")
    import soundfile as sf
    
    # 1. Read the audio directly using soundfile (bypassing torchaudio's broken backend)
    audio_array, sr = sf.read(filepath)
    
    # 2. Convert the numpy array to a PyTorch tensor
    # soundfile loads as [time, channels], but PyTorch expects [channels, time]
    if audio_array.ndim == 1:
        # Mono audio
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    else:
        # Stereo audio (transpose it)
        waveform = torch.tensor(audio_array, dtype=torch.float32).t()
    total_samples = waveform.shape[1]

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE

    chunks = []
    start = 0
    while start < total_samples:
        end = start + CHUNK_LENGTH
        chunk = waveform[:, start:min(end, total_samples)]
        if chunk.shape[1] >= MIN_CHUNK_SAMPLES:
            chunks.append(chunk)
        start += CHUNK_STEP
        if len(chunks) >= MAX_CHUNKS:
            break

    chunk_real_probs = []
    chunk_fake_probs = []
    start_time = time.time()

    with torch.no_grad():
        for chunk in chunks:
            mel_spec  = preprocess_chunk(chunk, device)
            logits    = student(mel_spec)
            probs     = F.softmax(logits, dim=1)
            chunk_real_probs.append(probs[0][0].item() * 100)
            chunk_fake_probs.append(probs[0][1].item() * 100)

    inference_ms = (time.time() - start_time) * 1000

    avg_real   = float(np.mean(chunk_real_probs))
    avg_fake   = float(np.mean(chunk_fake_probs))
    fake_votes = sum(1 for p in chunk_fake_probs if p > 50)
    total_v    = len(chunks)

    prediction = "FAKE" if avg_fake > avg_real else "REAL"
    confidence = max(avg_real, avg_fake)

    if save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Display the mel spectrogram of the first chunk
        first_chunk = waveform[:, :min(CHUNK_LENGTH, total_samples)]
        mel_display = preprocess_chunk(first_chunk, device).squeeze().cpu().numpy()
        im = axes[0].imshow(mel_display, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Mel Spectrogram")
        plt.colorbar(im, ax=axes[0])

        chunk_indices = list(range(1, len(chunks) + 1))
        colors = ["tomato" if f > 50 else "royalblue" for f in chunk_fake_probs]
        axes[1].bar(chunk_indices, chunk_fake_probs, color=colors, edgecolor="black")
        axes[1].axhline(y=50, color="gray", linestyle="--")
        axes[1].set_title("Fake Probability Per Chunk")
        axes[1].set_ylim(0, 110)

        bar_colors = ["royalblue" if prediction == "REAL" else "lightblue", "tomato" if prediction == "FAKE" else "lightsalmon"]
        bars = axes[2].bar(["Real Audio", "Fake Audio"], [avg_real, avg_fake], color=bar_colors, edgecolor="black")
        axes[2].set_ylim(0, 110)
        axes[2].set_title(f"Final: {prediction}")
        
        plt.tight_layout()
        plt.savefig("inference_result.png", dpi=150)
        plt.close() # Close plot to free memory

    return {
        "prediction": prediction,
        "confidence": confidence,
        "avg_real": avg_real,
        "avg_fake": avg_fake,
        "fake_votes": fake_votes,
        "total_chunks": total_v,
        "inference_ms": inference_ms,
    }