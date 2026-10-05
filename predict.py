import torch
from PIL import Image
from config import *
from dataset import eval_tf
from model import LeafNet

## Load the trained model once
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = LeafNet(len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes

## Return top-k (name, confidence) for one image
@torch.no_grad()
def predict(model, classes, image, topk=3):
    if isinstance(image, str):
        image = Image.open(image)
    x = eval_tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.topk(topk)
    return [(classes[i], c.item()) for i, c in zip(idx.tolist(), conf)]

if __name__ == "__main__":
    import sys
    model, classes = load_model()
    for name, c in predict(model, classes, sys.argv[1]):
        print(f"{name}: {c * 100:.1f}%")