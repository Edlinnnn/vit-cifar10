"""
test_pipeline.py
----------------
Unit tests for the ViT CIFAR-10 pipeline.

Run with:
    pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Data tests ───────────────────────────────────────────────────────────────

class TestDataPreprocessing:

    def test_load_returns_correct_shapes(self):
        from src.data_preprocessing import load_cifar10
        (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_cifar10()

        assert x_tr.shape[1:]  == (32, 32, 3)
        assert x_val.shape[1:] == (32, 32, 3)
        assert x_te.shape      == (10000, 32, 32, 3)

        assert len(x_tr) + len(x_val) == 50000
        assert len(x_te)              == 10000

    def test_pixel_range(self):
        from src.data_preprocessing import load_cifar10
        (x_tr, _), _, (x_te, _) = load_cifar10()
        assert x_tr.min() >= 0.0
        assert x_tr.max() <= 1.0
        assert x_te.min() >= 0.0
        assert x_te.max() <= 1.0

    def test_label_range(self):
        from src.data_preprocessing import load_cifar10
        (_, y_tr), (_, y_val), (_, y_te) = load_cifar10()
        for y in [y_tr, y_val, y_te]:
            assert y.min() >= 0
            assert y.max() <= 9

    def test_dataset_pipeline_batches(self):
        import tensorflow as tf
        from src.data_preprocessing import make_dataset
        x = np.random.rand(200, 32, 32, 3).astype("float32")
        y = np.random.randint(0, 10, 200)
        ds = make_dataset(x, y, augment=False, batch_size=32)
        batch_imgs, batch_lbls = next(iter(ds))
        assert batch_imgs.shape == (32, 32, 32, 3)
        assert batch_lbls.shape == (32,)

    def test_augmented_dataset_same_shape(self):
        import tensorflow as tf
        from src.data_preprocessing import make_dataset
        x = np.random.rand(64, 32, 32, 3).astype("float32")
        y = np.random.randint(0, 10, 64)
        ds = make_dataset(x, y, augment=True, batch_size=16)
        batch_imgs, batch_lbls = next(iter(ds))
        assert batch_imgs.shape == (16, 32, 32, 3)


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestViTModel:

    def test_output_shape(self):
        from src.vit_model import build_vit
        model = build_vit(image_size=32, patch_size=4, num_classes=10,
                          embed_dim=32, num_heads=4, num_layers=2)
        dummy = np.random.rand(4, 32, 32, 3).astype("float32")
        out   = model(dummy, training=False)
        assert out.shape == (4, 10)

    def test_output_sums_to_one(self):
        from src.vit_model import build_vit
        model = build_vit(image_size=32, patch_size=4, num_classes=10,
                          embed_dim=32, num_heads=4, num_layers=2)
        dummy = np.random.rand(8, 32, 32, 3).astype("float32")
        out   = model(dummy, training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(8), atol=1e-5)

    def test_all_probs_in_range(self):
        from src.vit_model import build_vit
        model = build_vit(image_size=32, patch_size=4, num_classes=10,
                          embed_dim=32, num_heads=4, num_layers=2)
        dummy = np.random.rand(4, 32, 32, 3).astype("float32")
        out   = model(dummy, training=False).numpy()
        assert (out >= 0).all() and (out <= 1).all()

    def test_patch_embed_shape(self):
        from src.vit_model import PatchEmbedding
        pe = PatchEmbedding(image_size=32, patch_size=4, embed_dim=64)
        dummy = np.random.rand(2, 32, 32, 3).astype("float32")
        out   = pe(dummy)
        # (32/4)^2 = 64 patches
        assert out.shape == (2, 64, 64)

    def test_positional_embedding_shape(self):
        from src.vit_model import AddPositionalEmbedding
        ape = AddPositionalEmbedding(num_patches=64, embed_dim=64)
        dummy = np.zeros((3, 64, 64), dtype="float32")
        out   = ape(dummy)
        # Should prepend [CLS]: 64 + 1 = 65 tokens
        assert out.shape == (3, 65, 64)

    def test_transformer_block_residual(self):
        from src.vit_model import TransformerBlock
        block = TransformerBlock(embed_dim=32, num_heads=4, mlp_dim=64)
        dummy = np.random.rand(2, 10, 32).astype("float32")
        out   = block(dummy, training=False)
        assert out.shape == (2, 10, 32)

    def test_different_patch_sizes(self):
        from src.vit_model import build_vit
        for ps in [2, 4, 8]:
            model = build_vit(image_size=32, patch_size=ps, num_classes=10,
                              embed_dim=32, num_heads=4, num_layers=1)
            dummy = np.random.rand(2, 32, 32, 3).astype("float32")
            out   = model(dummy, training=False)
            assert out.shape == (2, 10), f"Failed for patch_size={ps}"


# ─── App tests (no model needed — uses demo endpoint) ────────────────────────

class TestFlaskApp:

    @pytest.fixture
    def client(self):
        import importlib
        # Patch model loading to avoid needing a saved model
        app_module = importlib.import_module("app.app")
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def test_index_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_classes_endpoint(self, client):
        resp = client.get("/classes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["classes"]) == 10

    def test_predict_no_file_400(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 400

    def test_demo_predict(self, client):
        """Demo endpoint returns plausible fake data without a real model."""
        import io
        from PIL import Image as PILImage
        # Create a tiny in-memory PNG
        img = PILImage.new("RGB", (32, 32), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/predict/demo",
            data={"image": (buf, "test.png")},
            content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "class"         in data
        assert "confidence"    in data
        assert "probabilities" in data
        assert len(data["probabilities"]) == 10
        assert data["demo"] is True


# ─── Image preprocessing tests ───────────────────────────────────────────────

class TestPreprocessing:

    def test_preprocess_output_shape(self):
        import io
        from PIL import Image as PILImage
        from app.app import preprocess_image

        img = PILImage.new("RGB", (256, 256), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out = preprocess_image(buf.getvalue())

        assert out.shape == (1, 32, 32, 3)

    def test_preprocess_normalised(self):
        import io
        from PIL import Image as PILImage
        from app.app import preprocess_image

        img = PILImage.new("RGB", (64, 64), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out = preprocess_image(buf.getvalue())

        assert out.max() <= 1.0
        assert out.min() >= 0.0