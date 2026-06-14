"""
evaluate_torch.py
-----------------
Loads the PyTorch ViT checkpoint and runs full evaluation.

Usage:
    python src/evaluate_torch.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vit_model import ViT
from src.train import get_dataloaders

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "vit_cifar10_torch.pth")
DOCS_DIR   = os.path.join(ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]


def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ViT().to(device)
    ckpt   = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")
    return model, device


@torch.no_grad()
def get_predictions(model, loader, device):
    y_true, y_pred, y_probs_all = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        y_probs_all.append(probs)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred), np.concatenate(y_probs_all)


def plot_confusion_matrix(y_true, y_pred):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="magma",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.4, linecolor="#1a1a2e")
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=15)
    ax.set_xlabel("Predicted", color="#a78bfa"); ax.set_ylabel("True", color="#a78bfa")
    ax.tick_params(colors="white"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(DOCS_DIR, "confusion_matrix_torch.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved → {path}"); plt.show()


def plot_per_class_accuracy(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    colours = ["#4ade80" if a >= 0.75 else "#facc15" if a >= 0.60 else "#f87171" for a in acc]

    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(CLASS_NAMES, acc * 100, color=colours, edgecolor="#333")
    ax.axhline(acc.mean()*100, color="#a78bfa", ls="--", lw=1.5, label=f"Mean {acc.mean()*100:.1f}%")
    ax.set_ylim(0, 108); ax.set_title("Per-Class Accuracy", color="white", fontsize=13)
    ax.set_xlabel("Class", color="#a78bfa"); ax.set_ylabel("Accuracy (%)", color="#a78bfa")
    ax.tick_params(colors="white"); ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                f"{a*100:.1f}%", ha="center", va="bottom", color="white", fontsize=8)
    plt.xticks(rotation=30); plt.tight_layout()
    path = os.path.join(DOCS_DIR, "per_class_accuracy_torch.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved → {path}"); plt.show()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}\nRun  python src/train_torch.py  first.")
        return

    _, _, test_loader = get_dataloaders(batch_size=128)
    model, device     = load_model(MODEL_PATH)
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*50}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred)

    summary = {"test_accuracy": float(acc),
               "per_class": {c: float(v) for c, v in zip(
                   CLASS_NAMES,
                   confusion_matrix(y_true, y_pred).diagonal() /
                   confusion_matrix(y_true, y_pred).sum(axis=1))}}
    with open(os.path.join(ROOT, "models", "eval_summary_torch.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n✅  Evaluation complete.")


if __name__ == "__main__":
    main()