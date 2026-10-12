import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from config import *
from dataset import get_loaders
from model import LeafNet

## One pass over the data (train if optimizer given, else evaluate)
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(imgs, labels if training else None)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
        all_preds += preds.cpu().tolist()
        all_labels += labels.cpu().tolist()

    return total_loss / n, correct / n, all_labels, all_preds

def main():
    train_dl, val_dl, test_dl, classes = get_loaders()
    print(f"Classes: {len(classes)} | Device: {DEVICE}")

    model = LeafNet(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    ## Small LR for the backbone, bigger LR for new layers
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.embed.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_acc = 0.0
    start_epoch = 1

    ## Resume: last.pth irundha andha epoch irundhu continue pannu
    if os.path.exists(LAST_CKPT_PATH):
        last = torch.load(LAST_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        best_acc = last["best_acc"]
        start_epoch = last["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best acc so far {best_acc:.3f})")

    ## Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, criterion, optimizer, scaler)
        va_loss, va_acc, _, _ = run_epoch(model, val_dl, criterion)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{EPOCHS} | train acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} | val acc {va_acc:.3f}")

        ## Best model save
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes}, CKPT_PATH)
            print("  saved best model")

        ## Every epoch: resume checkpoint save
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "classes": classes}, LAST_CKPT_PATH)

    ## Final test evaluation with the best checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    _, test_acc, y_true, y_pred = run_epoch(model, test_dl, criterion)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

if __name__ == "__main__":
    main()