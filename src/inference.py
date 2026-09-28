"""
inference.py
------------
Load a trained ViT checkpoint and classify images. Shared by evaluate.py and the Flask app.
"""

import io
import os

import torch
from PIL import Image

from src.data_preprocessing import CLASS_NAMES, eval_transform
from src.vit_model import build_vit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "vit_cifar10_torch.pth")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: str = DEFAULT_MODEL_PATH, device: torch.device = None):
    """
    Load a checkpoint written by train.py and return (model, device, checkpoint).
    Accepts either the full checkpoint dict or a bare state_dict.
    """
    device = device or get_device()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Train one with `python src/train.py` "
            "or copy your trained vit_cifar10_torch.pth into models/."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = build_vit(**config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, device, (ckpt if isinstance(ckpt, dict) else {})


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """Raw image bytes -> normalised (1, 3, 32, 32) float tensor."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return eval_transform()(img).unsqueeze(0)


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, device: torch.device) -> list:
    """Return the 10 class probabilities for a single preprocessed image."""
    logits = model(x.to(device))
    return torch.softmax(logits, dim=-1)[0].cpu().tolist()


def top_k(probs: list, k: int = 3):
    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda cp: cp[1], reverse=True)
    return ranked[:k]
