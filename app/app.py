"""
app.py
------
Flask REST API for the ViT CIFAR-10 image classifier (PyTorch).

Endpoints:
    GET  /              → Web UI
    POST /predict       → JSON {class, confidence, probabilities}
    GET  /health        → {"status": "ok"}
    GET  /classes       → list of class names

Run locally:
    python app/app.py

Production (Render / Cloud Run / any Docker host):
    gunicorn app.app:app --bind 0.0.0.0:$PORT
"""

import os
import sys
import logging
import threading

import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template

from src.data_preprocessing import CLASS_NAMES
from src.inference import DEFAULT_MODEL_PATH, load_model, preprocess_image, predict_probs, top_k

# ─── Config ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
MAX_UPLOAD_MB = 10
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Small model: a couple of CPU threads is plenty and keeps memory low on free tiers.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

CLASS_EMOJI = {
    "airplane":    "✈️",  "automobile": "🚗",  "bird":  "🐦",
    "cat":         "🐱",  "deer":       "🦌",  "dog":   "🐶",
    "frog":        "🐸",  "horse":      "🐴",  "ship":  "🚢",
    "truck":       "🚛",
}

# ─── App factory ──────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder = os.path.join(os.path.dirname(__file__), "templates"),
            static_folder   = os.path.join(os.path.dirname(__file__), "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Global model holder (loaded once, on first request, thread-safe)
_model, _device = None, None
_lock = threading.Lock()


def get_model():
    global _model, _device
    if _model is None:
        with _lock:
            if _model is None:
                logger.info("Loading ViT model from %s …", MODEL_PATH)
                _model, _device, _ = load_model(MODEL_PATH)
                logger.info("Model loaded ✓ on %s", _device)
    return _model, _device


def build_response(probs, demo=False):
    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    pred_cls = CLASS_NAMES[pred_idx]
    body = {
        "class":         pred_cls,
        "emoji":         CLASS_EMOJI.get(pred_cls, ""),
        "confidence":    float(probs[pred_idx]),
        "probabilities": {c: round(float(p), 6) for c, p in zip(CLASS_NAMES, probs)},
        "top3":          [{"class": c, "emoji": CLASS_EMOJI.get(c, ""),
                           "confidence": round(p * 100, 1)} for c, p in top_k(probs, 3)],
    }
    if demo:
        body["demo"] = True
    return body


def validate_upload():
    """Return (file, None) or (None, (json, status))."""
    if "image" not in request.files:
        return None, (jsonify({"error": "No image field in request"}), 400)
    file = request.files["image"]
    if file.filename == "":
        return None, (jsonify({"error": "Empty filename"}), 400)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return None, (jsonify({"error": f"Unsupported file type: {ext}"}), 415)
    return file, None


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)"}), 413


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/health")
def health():
    return jsonify({"status": "ok",
                    "model_loaded": _model is not None,
                    "model_available": os.path.exists(MODEL_PATH)})


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
    file, err = validate_upload()
    if err:
        return err

    try:
        model, device = get_model()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return jsonify({"error": "Model not loaded on the server. "
                                 "Add models/vit_cifar10_torch.pth and restart."}), 503

    try:
        x = preprocess_image(file.read())
    except Exception:
        return jsonify({"error": "Could not read that file as an image"}), 400

    try:
        probs = predict_probs(model, x, device)
        return jsonify(build_response(probs))
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


# ─── Demo endpoint (no real model needed — returns mock data) ─────────────────

@app.route("/predict/demo", methods=["POST"])
def predict_demo():
    """Returns plausible fake predictions. Useful for UI testing without a trained model."""
    import random
    pred_idx  = random.randint(0, 9)
    raw = [abs(random.gauss(0, 1)) for _ in range(10)]
    raw[pred_idx] *= 5
    total = sum(raw)
    return jsonify(build_response([r / total for r in raw], demo=True))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting ViT CIFAR-10 app on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)