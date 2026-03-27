from pathlib import Path

import pytest
import torch
import yaml

from src import utils


@pytest.mark.unit
def test_resolve_config_accepts_dict():
    cfg = {"model": {"image_size": 224}}
    assert utils._resolve_config(cfg) == cfg


@pytest.mark.unit
def test_resolve_config_accepts_yaml_path(tmp_path: Path):
    cfg = {"model": {"image_size": 128}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    resolved = utils._resolve_config(cfg_path)
    assert resolved == cfg


@pytest.mark.unit
def test_resolve_config_rejects_invalid_type():
    with pytest.raises(TypeError):
        utils._resolve_config(3.14)


@pytest.mark.unit
def test_image_to_tensor_returns_normalized_tensor_shape():
    image = torch.linspace(0, 100, steps=64 * 64, dtype=torch.float32).reshape(64, 64).numpy()

    tensor = utils.image_to_tensor(image=image, image_size=96)

    assert tensor.shape == (3, 96, 96)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


@pytest.mark.unit
def test_image_to_tensor_handles_uniform_image_without_nan():
    image = torch.full((32, 32), 5.0, dtype=torch.float32).numpy()

    tensor = utils.image_to_tensor(image=image, image_size=64)

    assert tensor.shape == (3, 64, 64)
    assert torch.isfinite(tensor).all()


@pytest.mark.unit
def test_freeze_backbone_only_freezes_vit():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(8, 6)

    model = DummyModel()
    utils.freeze_backbone(model)

    assert all(not p.requires_grad for p in model.vit.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


@pytest.mark.unit
def test_load_model_falls_back_to_backbone(monkeypatch):
    class DummyVit:
        def __init__(self):
            self.fallback_called = False

        def load_state_dict(self, state_dict, strict=False):
            self.fallback_called = True
            self.strict = strict

    class DummyModel:
        def __init__(self):
            self.vit = DummyVit()

        def load_state_dict(self, state_dict, strict=True):
            raise RuntimeError("forced full-load failure")

    dummy_model = DummyModel()

    monkeypatch.setattr(utils, "create_model", lambda config: dummy_model)
    monkeypatch.setattr(utils.torch, "load", lambda *args, **kwargs: {"w": 1})

    loaded = utils.load_model("fake_model.pth", {"model": {"backbone": "b", "hidden_dim": 8, "num_outputs": 6}})

    assert loaded is dummy_model
    assert loaded.vit.fallback_called is True
    assert loaded.vit.strict is False
