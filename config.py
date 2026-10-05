import torch

## Paths
DATA_DIR = "data"
CKPT_PATH = "best_model.pth"

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