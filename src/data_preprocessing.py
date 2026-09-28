"""
data_preprocessing.py
---------------------
Loading, normalisation, augmentation and DataLoaders for CIFAR-10 (PyTorch).

CIFAR-10 classes:
    0: airplane  1: automobile  2: bird    3: cat   4: deer
    5: dog       6: frog        7: horse   8: ship  9: truck
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE  = 32        # CIFAR-10 native resolution
NUM_CLASSES = 10
BATCH_SIZE  = 128
VAL_FRACTION = 0.1
SEED = 42

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Per-channel mean / std of the CIFAR-10 training set.
# Used for training, evaluation AND inference so the model always sees the same scale.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


# ─── Transforms ───────────────────────────────────────────────────────────────

def train_transform():
    """Augmentation applied on the fly to training images only."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def eval_transform():
    """Deterministic transform for validation, test and inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 2, data_dir: str = DATA_DIR):
    """
    Downloads CIFAR-10 (if needed) and returns train / val / test DataLoaders.

    Split: 45,000 train / 5,000 validation (fixed seed) / 10,000 test.
    The validation subset uses the deterministic eval transform (no augmentation).
    """
    train_full = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform())
    val_full = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_transform())
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform())

    val_size   = int(VAL_FRACTION * len(train_full))
    train_size = len(train_full) - val_size
    train_idx, val_idx = random_split(
        range(len(train_full)), [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_set = torch.utils.data.Subset(train_full, list(train_idx))
    val_set   = torch.utils.data.Subset(val_full,   list(val_idx))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    print(f"Train: {len(train_set):,}  |  Val: {len(val_set):,}  |  Test: {len(test_set):,}")
    return train_loader, val_loader, test_loader


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo Normalize() so a (3, H, W) tensor can be displayed."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


# ─── Visualisation helpers ────────────────────────────────────────────────────

def plot_sample_images(dataset, n_cols: int = 10, n_rows: int = 3, save_path: str = None):
    """Random grid of samples from a torchvision CIFAR10 dataset (raw, un-normalised)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.6),
                             facecolor="#0f0f0f")
    idx = np.random.choice(len(dataset), n_rows * n_cols, replace=False)
    for ax, i in zip(axes.flat, idx):
        img, label = dataset.data[i], dataset.targets[i]
        ax.imshow(img)
        ax.set_title(CLASS_NAMES[label], fontsize=7, color="white", pad=2)
        ax.axis("off")
    plt.suptitle("CIFAR-10 Sample Images", color="white", fontsize=13)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


def plot_class_distribution(labels, title: str = "Class Distribution", save_path: str = None):
    """Bar chart of per-class sample counts."""
    import matplotlib.pyplot as plt

    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(CLASS_NAMES, counts, color="#7c3aed", edgecolor="#a78bfa", linewidth=0.7)
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel("Class", color="#a78bfa")
    ax.set_ylabel("Count", color="#a78bfa")
    ax.tick_params(colors="white", rotation=30)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", color="white", fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ─── Main (quick sanity check) ────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, _, _ = get_dataloaders()
    imgs, labels = next(iter(train_loader))
    print(f"Batch — images: {tuple(imgs.shape)}  labels: {tuple(labels.shape)}")

    raw_test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    plot_sample_images(raw_test, save_path=os.path.join(ROOT, "docs", "sample_images.png"))
    plot_class_distribution(raw_test.targets, title="Test Set — Class Distribution",
                            save_path=os.path.join(ROOT, "docs", "class_distribution.png"))
