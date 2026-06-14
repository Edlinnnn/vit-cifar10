"""
app.py
------
Flask REST API for the ViT CIFAR-10 image classifier.

Endpoints:
    GET  /              → Web UI
    POST /predict       → JSON {class, confidence, probabilities}
    GET  /health        → {"status": "ok"}
    GET  /classes       → list of class names

Run locally:
    python app/app.py

Production (Render / Railway):
    gunicorn app.app:app --bind 0.0.0.0:$PORT
"""

import os
import sys
import io
import json
import base64
import logging
import numpy as np
from PIL import Image

# ─── TensorFlow (lazy import so startup is fast on serverless) ────────────────
import tensorflow as tf
from tensorflow import keras

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory

# ─── Config ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "vit_cifar10.keras")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_EMOJI = {
    "airplane":    "✈️",  "automobile": "🚗",  "bird":  "🐦",
    "cat":         "🐱",  "deer":       "🦌",  "dog":   "🐶",
    "frog":        "🐸",  "horse":      "🐴",  "ship":  "🚢",
    "truck":       "🚛",
}

IMAGE_SIZE = 32

# ─── App factory ──────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder = os.path.join(os.path.dirname(__file__), "templates"),
            static_folder   = os.path.join(os.path.dirname(__file__), "static"))

# Global model holder (loaded once on first request)
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Please run  python src/train.py  first."
            )
        logger.info("Loading ViT model …")
        _model = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded ✓")
    return _model


# ─── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Accepts raw image bytes, returns a (1, 32, 32, 3) float32 array
    normalised to [0, 1].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]   # add batch dimension


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/classes")
def classes():
    return jsonify({"classes": CLASS_NAMES, "emoji": CLASS_EMOJI})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts multipart/form-data with field name "image".
    Returns JSON:
        {
          "class":         "dog",
          "emoji":         "🐶",
          "confidence":    0.92,
          "probabilities": {"airplane": 0.01, "dog": 0.92, ...}
        }
    """
    # ── Validate request ──
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 415

    # ── Inference ──
    try:
        raw = file.read()
        img = preprocess_image(raw)

        model = get_model()
        probs = model(img, training=False).numpy()[0]   # (10,)

        pred_idx  = int(np.argmax(probs))
        pred_cls  = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {cls: round(float(p), 6)
                     for cls, p in zip(CLASS_NAMES, probs)}

        # Top-3 for UI
        top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            "class":         pred_cls,
            "emoji":         CLASS_EMOJI.get(pred_cls, ""),
            "confidence":    confidence,
            "probabilities": prob_dict,
            "top3":          [{"class": c, "emoji": CLASS_EMOJI.get(c, ""),
                               "confidence": round(p * 100, 1)}
                              for c, p in top3],
        })

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


# ─── Demo endpoint (no real model needed — returns mock data) ─────────────────

@app.route("/predict/demo", methods=["POST"])
def predict_demo():
    """Returns plausible fake predictions. Useful for UI testing without a trained model."""
    import random
    pred_idx  = random.randint(0, 9)
    pred_cls  = CLASS_NAMES[pred_idx]
    raw_probs = np.abs(np.random.randn(10))
    raw_probs[pred_idx] *= 5
    probs = (raw_probs / raw_probs.sum()).tolist()

    prob_dict = {cls: round(float(p), 6) for cls, p in zip(CLASS_NAMES, probs)}
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return jsonify({
        "class":         pred_cls,
        "emoji":         CLASS_EMOJI.get(pred_cls, ""),
        "confidence":    max(probs),
        "probabilities": prob_dict,
        "top3":          [{"class": c, "emoji": CLASS_EMOJI.get(c, ""),
                           "confidence": round(p * 100, 1)}
                          for c, p in top3],
        "demo":          True
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting ViT CIFAR-10 app on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)