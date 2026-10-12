import torch
from config import *
from dataset import get_loaders
from predict import load_model

## Per-class precision, recall and F1 (no scikit-learn needed)
def report(y_true, y_pred, classes):
    print(f"{'Class':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    f1_list = []
    for c, name in enumerate(classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1_list.append(f1)
        print(f"{name:<18}{prec:>10.3f}{rec:>10.3f}{f1:>10.3f}{tp + fn:>10}")
    print(f"\nMacro F1: {sum(f1_list) / len(f1_list):.3f}")

def main():
    ## Load the trained model and the test data
    model, classes = load_model()
    _, _, test_dl, _ = get_loaders()

    ## Run predictions on the test set
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            out = model(imgs.to(DEVICE))
            y_pred += out.argmax(1).cpu().tolist()
            y_true += labels.tolist()

    ## Print the results
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"Test accuracy: {correct / len(y_true):.4f}  ({correct}/{len(y_true)})\n")
    report(y_true, y_pred, classes)

if __name__ == "__main__":
    main()