import os
from PIL import Image

## Settings
SRC, DST, MAX_SIDE = "data", "data_small", 256
EXTS = (".jpg", ".jpeg", ".png")

## Shrink every image and keep the same folder structure
count = 0
for root, _, files in os.walk(SRC):
    for f in files:
        if not f.lower().endswith(EXTS):
            continue
        out_dir = root.replace(SRC, DST, 1)
        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(os.path.join(root, f)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(os.path.join(out_dir, os.path.splitext(f)[0] + ".jpg"), quality=90)
        count += 1

print(f"Done! {count} images saved in '{DST}/'")