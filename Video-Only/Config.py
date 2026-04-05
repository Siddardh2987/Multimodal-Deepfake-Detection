# import os
# import cv2
# import json
# import math
# import random
# import shutil
# import time
import warnings
# from pathlib import Path
# from typing import List, Tuple, Optional, Dict

# import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm

# # timm provides pretrained Swin Transformer
# import timm

# # MTCNN for face detection (part of facenet-pytorch)
# from facenet_pytorch import MTCNN
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings("ignore")
CONFIG = {
    # Paths
    "dataset_root"    : "/content/drive/MyDrive/AI-Sem-Project/DFDC_Dataset",          # dataset/real  dataset/fake
    "checkpoint_dir"  : "/content/drive/MyDrive/AI-Sem-Project/DFDC_Dataset/checkpoints",
    "best_model_path" : "/content/drive/MyDrive/AI-Sem-Project/DFDC_Dataset/checkpoints/best_model.pth",
    "log_path"        : "/content/drive/MyDrive/AI-Sem-Project/training_log.json",

    # Frame extraction
    "max_frames_per_video"  : 30,    # hard cap before sampling
    "frame_sample_count"    : 16,    # frames to keep after sampling
    "face_size"             : 224,   # pixels (H=W fed to model)
    "min_face_confidence"   : 0.90,  # MTCNN threshold

    # Dataset
    "val_split"     : 0.15,   # 15 % for validation
    "test_split"    : 0.05,   # 5  % for test (held out)

    # Training
    "batch_size"          : 2, #Siddhu (Batch size and epochs changes).
    "accumulation_steps"  : 16,
    "num_epochs"          : 6,
    "lr"                  : 5e-5,
    "weight_decay"        : 0.05,
    "num_workers"         : 2,
    "save_every"          : 5,      # save checkpoint every N epochs

    # Model
    "model_name"    : "swin_base_patch4_window7_224",
    "num_classes"   : 2,      # REAL=0, FAKE=1
    "pretrained"    : True,
    "dropout"       : 0.3,

    # Augmentation
    "use_augment"   : True,
}