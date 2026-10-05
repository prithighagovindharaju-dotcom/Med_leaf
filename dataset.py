from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import *

## ImageNet normalization (DINOv2 was trained with this)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_loaders():
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/valid", eval_tf)
    test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", eval_tf)

    mk = lambda ds, shuffle: DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True)

    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), train_ds.classes