"""
test_pipeline.py
----------------
Unit tests for the ViT CIFAR-10 pipeline (PyTorch). No dataset download or GPU needed.

Run with:
    pytest tests/ -v
"""

import io
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vit_model import (DEFAULT_CONFIG, PatchEmbedding, TransformerBlock,
                           build_vit, count_parameters)
from src.data_preprocessing import CLASS_NAMES, eval_transform, train_transform
from src.inference import load_model, preprocess_image, predict_probs, top_k


def png_bytes(size=(64, 64), color=(100, 150, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def checkpoint(tmp_path):
    """A randomly initialised checkpoint in the same format train.py writes."""
    torch.manual_seed(0)
    model = build_vit()
    path = tmp_path / "vit.pth"
    torch.save({"epoch": 1, "config": DEFAULT_CONFIG, "class_names": CLASS_NAMES,
                "model_state_dict": model.state_dict(), "val_acc": 0.1, "val_loss": 2.3}, path)
    return str(path)


# ─── Model ────────────────────────────────────────────────────────────────────

class TestViTModel:

    def test_output_shape(self):
        out = build_vit()(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_parameter_count(self):
        assert count_parameters(build_vit()) == 326_602

    def test_softmax_of_logits_sums_to_one(self):
        model = build_vit().eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.randn(8, 3, 32, 32)), dim=-1)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(8))
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_patch_embed_shape(self):
        out = PatchEmbedding(image_size=32, patch_size=4, embed_dim=64)(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 64, 64)          # (32/4)^2 = 64 patches of dim 64

    def test_cls_token_and_positions(self):
        model = build_vit()
        assert model.cls_token.shape == (1, 1, 64)
        assert model.pos_embed.shape == (1, 65, 64)   # 64 patches + [CLS]

    def test_transformer_block_keeps_shape(self):
        block = TransformerBlock(embed_dim=32, num_heads=4, mlp_dim=64).eval()
        assert block(torch.randn(2, 10, 32)).shape == (2, 10, 32)

    @pytest.mark.parametrize("patch_size", [2, 4, 8])
    def test_different_patch_sizes(self, patch_size):
        model = build_vit(patch_size=patch_size, embed_dim=32, num_heads=4, num_layers=1)
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    def test_backward_pass(self):
        model = build_vit(embed_dim=32, num_heads=4, num_layers=1)
        loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 32, 32)),
                                                 torch.tensor([0, 1, 2, 3]))
        loss.backward()
        assert model.patch_embed.projection.weight.grad is not None


# ─── Transforms ───────────────────────────────────────────────────────────────

class TestTransforms:

    def test_eval_transform_resizes_and_normalises(self):
        x = eval_transform()(Image.new("RGB", (256, 128), color=(255, 0, 0)))
        assert x.shape == (3, 32, 32)
        assert x[0].mean() > 1.5 and x[1].mean() < 0     # red channel high, green low after normalise

    def test_train_transform_shape(self):
        x = train_transform()(Image.new("RGB", (32, 32), color=(10, 20, 30)))
        assert x.shape == (3, 32, 32)


# ─── Inference helpers ────────────────────────────────────────────────────────

class TestInference:

    def test_preprocess_output_shape(self):
        assert preprocess_image(png_bytes((256, 256))).shape == (1, 3, 32, 32)

    def test_preprocess_accepts_rgba_and_grayscale(self):
        for mode in ("RGBA", "L"):
            buf = io.BytesIO()
            Image.new(mode, (40, 40)).save(buf, format="PNG")
            assert preprocess_image(buf.getvalue()).shape == (1, 3, 32, 32)

    def test_load_checkpoint_and_predict(self, checkpoint):
        model, device, ckpt = load_model(checkpoint, device=torch.device("cpu"))
        assert not model.training and ckpt["epoch"] == 1
        probs = predict_probs(model, preprocess_image(png_bytes()), device)
        assert len(probs) == 10 and abs(sum(probs) - 1) < 1e-5

    def test_load_bare_state_dict(self, tmp_path):
        path = tmp_path / "bare.pth"
        torch.save(build_vit().state_dict(), path)
        model, _, _ = load_model(str(path), device=torch.device("cpu"))
        assert count_parameters(model) == 326_602

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "nope.pth"))

    def test_top_k(self):
        probs = [0.0] * 10
        probs[5], probs[3], probs[1] = 0.6, 0.3, 0.1
        assert [c for c, _ in top_k(probs, 3)] == ["dog", "cat", "automobile"]


# ─── Flask app ────────────────────────────────────────────────────────────────

@pytest.fixture
def app_module():
    import app.app as app_module
    app_module.app.config["TESTING"] = True
    app_module._model, app_module._device = None, None
    yield app_module
    app_module._model, app_module._device = None, None


@pytest.fixture
def client(app_module):
    with app_module.app.test_client() as c:
        yield c


class TestFlaskApp:

    def test_index_200(self, client):
        assert client.get("/").status_code == 200

    def test_health(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_classes_endpoint(self, client):
        assert len(client.get("/classes").get_json()["classes"]) == 10

    def test_predict_no_file_400(self, client):
        assert client.post("/predict").status_code == 400

    def test_predict_bad_extension_415(self, client):
        resp = client.post("/predict", data={"image": (io.BytesIO(b"x"), "notes.txt")},
                           content_type="multipart/form-data")
        assert resp.status_code == 415

    def test_predict_without_model_503(self, client, app_module, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "MODEL_PATH", str(tmp_path / "missing.pth"))
        resp = client.post("/predict", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                           content_type="multipart/form-data")
        assert resp.status_code == 503

    def test_predict_with_model(self, client, app_module, checkpoint, monkeypatch):
        monkeypatch.setattr(app_module, "MODEL_PATH", checkpoint)
        resp = client.post("/predict", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["class"] in CLASS_NAMES
        assert len(data["probabilities"]) == 10 and len(data["top3"]) == 3
        assert abs(sum(data["probabilities"].values()) - 1) < 1e-4

    def test_predict_corrupt_image_400(self, client, app_module, checkpoint, monkeypatch):
        monkeypatch.setattr(app_module, "MODEL_PATH", checkpoint)
        resp = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "a.png")},
                           content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_demo_predict(self, client):
        resp = client.post("/predict/demo", data={"image": (io.BytesIO(png_bytes()), "t.png")},
                           content_type="multipart/form-data")
        data = resp.get_json()
        assert resp.status_code == 200 and data["demo"] is True
        assert len(data["probabilities"]) == 10
