import os, random, shutil

## Settings
RAW_DIR = "raw_data"
OUT_DIR = "data"
SPLIT = {"train": 0.7, "valid": 0.2, "test": 0.1}
EXTS = (".jpg", ".jpeg", ".png")
random.seed(42)

## Go through each class folder and split its images
for cls in sorted(os.listdir(RAW_DIR)):
    cls_path = os.path.join(RAW_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    files = [f for f in os.listdir(cls_path) if f.lower().endswith(EXTS)]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * SPLIT["train"])
    n_valid = int(n * SPLIT["valid"])
    parts = {
        "train": files[:n_train],
        "valid": files[n_train:n_train + n_valid],
        "test": files[n_train + n_valid:],
    }

    for split, names in parts.items():
        dest = os.path.join(OUT_DIR, split, cls)
        os.makedirs(dest, exist_ok=True)
        for name in names:
            shutil.copy(os.path.join(cls_path, name), os.path.join(dest, name))

    print(f"{cls}: {n} images -> {len(parts['train'])}/{len(parts['valid'])}/{len(parts['test'])}")

print("Done! Split saved in 'data/'")