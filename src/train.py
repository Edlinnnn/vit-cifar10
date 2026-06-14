"""
train_torch.py
--------------
PyTorch training script — runs on your RTX 5050 GPU.

Usage:
    python src/train_torch.py
    python src/train_torch.py --epochs 30    # quick test
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vit_model import ViT

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(ROOT, "models")
DOCS_DIR   = os.path.join(ROOT, "docs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOCS_DIR,  exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "vit_cifar10_torch.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history_torch.json")

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    # 90/10 train/val split
    val_size   = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_set):,}  |  Val: {len(val_set):,}  |  Test: {len(test_set):,}")
    return train_loader, val_loader, test_loader


# ─── Train / Eval loops ───────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):   # mixed precision
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(history, save_path):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f0f0f")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    axes[0].plot(epochs, history["train_acc"], color="#7c3aed", lw=2, label="Train")
    axes[0].plot(epochs, history["val_acc"],   color="#a78bfa", lw=2, ls="--", label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[0].grid(alpha=0.2, color="white")

    axes[1].plot(epochs, history["train_loss"], color="#7c3aed", lw=2, label="Train")
    axes[1].plot(epochs, history["val_loss"],   color="#a78bfa", lw=2, ls="--", label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cross-Entropy")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[1].grid(alpha=0.2, color="white")

    plt.suptitle("ViT Training History (PyTorch · RTX 5050)", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Curves saved → {save_path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(epochs=60, base_lr=1e-3, batch_size=128, warmup_epochs=5, patience=15):

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*55}\n")

    # ── Data ──
    train_loader, val_loader, _ = get_dataloaders(batch_size)

    # ── Model ──
    model = ViT(image_size=32, patch_size=4, num_classes=10,
                embed_dim=64, num_heads=8, num_layers=6,
                mlp_ratio=4.0, dropout=0.1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01,
                                end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    # ── Mixed precision scaler ──
    scaler = torch.amp.GradScaler()

    # ── Training loop ──
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {vl_loss:.4f} val_acc {vl_acc:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.1f}s")

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":    vl_acc,
                "val_loss":   vl_loss,
            }, MODEL_PATH)
            print(f"  ✅ Saved best model (val_acc={vl_acc:.4f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── Save history ──
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history, save_path=os.path.join(DOCS_DIR, "training_curves_torch.png"))

    print(f"\n✅  Best val accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"✅  Model saved       → {MODEL_PATH}")
    return model, history


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--patience",       type=int,   default=15)
    args = p.parse_args()

    train(epochs=args.epochs, base_lr=args.lr, batch_size=args.batch_size,
          warmup_epochs=args.warmup_epochs, patience=args.patience)