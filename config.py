import os
import torch

## Paths
IS_COLAB = os.path.exists("/content")

BASE_DIR = "/content/drive/MyDrive/medicinal_leaf" if IS_COLAB else os.getcwd()
DATA_DIR = "/content/dataset" if IS_COLAB else os.path.join(BASE_DIR, "data_small")  # full training ku "data_small" ah "data" nu maathu

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(CKPT_DIR, "best_model.pth")   # best model
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")    # resume ku

## Image and training settings
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_WORKERS = 2

## Model settings
EMBED_DIM = 512
UNFREEZE_LAST_N = 4      # how many DINOv2 blocks to fine-tune
ARC_SCALE = 30.0
ARC_MARGIN = 0.30

## Learning rates
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")