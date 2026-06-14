"""
data_preprocessing.py
---------------------
Handles loading, preprocessing, and augmentation of the CIFAR-10 dataset.

CIFAR-10 classes:
    0: airplane  1: automobile  2: bird    3: cat   4: deer
    5: dog       6: frog        7: horse   8: ship  9: truck
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE  = 32        # CIFAR-10 native resolution
NUM_CLASSES = 10
BATCH_SIZE  = 64

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─── Load & Split ──────────────────────────────────────────────────────────────

def load_cifar10():
    """
    Downloads (if needed) and returns split datasets.

    Returns
    -------
    (x_train, y_train), (x_val, y_val), (x_test, y_test)
        Arrays with float32 images in [0, 1] and integer labels.
    """
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Squeeze label dim: (N,1) → (N,)
    y_train_full = y_train_full.squeeze()
    y_test       = y_test.squeeze()

    # Normalise to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test       = x_test.astype("float32")       / 255.0

    # Carve out 10 % of training data as validation
    val_size = int(len(x_train_full) * 0.1)
    x_val,   y_val   = x_train_full[:val_size],  y_train_full[:val_size]
    x_train, y_train = x_train_full[val_size:],  y_train_full[val_size:]

    print(f"Train : {x_train.shape}  |  Val : {x_val.shape}  |  Test : {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ─── Augmentation ─────────────────────────────────────────────────────────────

def build_augmentation_layer():
    """
    Keras Sequential layer for on-the-fly training augmentation.
    Applied only during training (training=True).
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomContrast(0.1),
    ], name="augmentation")


# ─── tf.data Pipeline ─────────────────────────────────────────────────────────

def make_dataset(x, y, *, augment: bool = False, batch_size: int = BATCH_SIZE):
    """
    Builds an optimised tf.data.Dataset pipeline.

    Parameters
    ----------
    x        : numpy array of images (float32, [0,1])
    y        : numpy array of integer labels
    augment  : if True, applies random augmentation per batch
    batch_size: mini-batch size
    """
    aug = build_augmentation_layer() if augment else None

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True) if augment else ds
    ds = ds.batch(batch_size, drop_remainder=False)

    if aug is not None:
        ds = ds.map(
            lambda imgs, lbls: (aug(imgs, training=True), lbls),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(batch_size: int = BATCH_SIZE):
    """
    One-call convenience wrapper.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset objects
    (x_test, y_test)          : raw numpy arrays (for evaluation plotting)
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10()

    train_ds = make_dataset(x_train, y_train, augment=True,  batch_size=batch_size)
    val_ds   = make_dataset(x_val,   y_val,   augment=False, batch_size=batch_size)
    test_ds  = make_dataset(x_test,  y_test,  augment=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds, (x_test, y_test)


# ─── Visualisation helpers ────────────────────────────────────────────────────

def plot_sample_images(x, y, n_cols: int = 10, n_rows: int = 3, save_path: str = None):
    """Display a random grid of CIFAR-10 samples with class labels."""
    fig = plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.05)

    indices = np.random.choice(len(x), n_rows * n_cols, replace=False)

    for idx, flat_i in enumerate(indices):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.imshow(x[flat_i])
        ax.set_title(CLASS_NAMES[y[flat_i]], fontsize=7, color="white", pad=2)
        ax.axis("off")

    plt.suptitle("CIFAR-10 Sample Images", color="white", fontsize=13, y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


def plot_class_distribution(y, title: str = "Class Distribution", save_path: str = None):
    """Bar chart of per-class sample counts."""
    counts = np.bincount(y, minlength=NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(CLASS_NAMES, counts, color="#7c3aed", edgecolor="#a78bfa", linewidth=0.7)
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel("Class", color="#a78bfa")
    ax.set_ylabel("Count",  color="#a78bfa")
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


# ─── Main (quick sanity-check) ────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds, test_ds, (x_test, y_test) = get_datasets()

    # Peek at one batch
    for imgs, labels in train_ds.take(1):
        print(f"Batch — images: {imgs.shape}  labels: {labels.shape}")
        print(f"Pixel range: [{imgs.numpy().min():.3f}, {imgs.numpy().max():.3f}]")

    plot_sample_images(x_test, y_test, save_path="docs/sample_images.png")
    plot_class_distribution(y_test, title="Test Set — Class Distribution",
                            save_path="docs/class_distribution.png")