from types import SimpleNamespace

import pytest
import torch

from src.model import MaxSWIN, MaxViT, MaxViTMultiHead


class _DummyVitBackbone(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # Mimic HF ViT output shape: (B, seq_len, hidden_dim)
        return SimpleNamespace(last_hidden_state=torch.randn(batch_size, 197, self.hidden_dim))


class _DummySwinBackbone(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # Mimic HF Swin output shape: (B, num_tokens, hidden_dim)
        return SimpleNamespace(last_hidden_state=torch.randn(batch_size, 49, self.hidden_dim))


@pytest.mark.unit
@pytest.mark.model
def test_maxvit_forward_returns_expected_shape():
    hidden_dim = 32
    num_outputs = 6
    backbone = _DummyVitBackbone(hidden_dim=hidden_dim)
    head = torch.nn.Linear(hidden_dim, num_outputs)
    model = MaxViT(vit_backbone=backbone, regression_head=head)

    x = torch.randn(3, 3, 64, 64)
    y = model(x)

    assert y.shape == (3, num_outputs)


@pytest.mark.unit
@pytest.mark.model
def test_maxvit_multihead_forward_returns_expected_shape():
    hidden_dim = 48
    backbone = _DummyVitBackbone(hidden_dim=hidden_dim)
    model = MaxViTMultiHead(vit_backbone=backbone, hidden_dim=hidden_dim)

    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 6)


@pytest.mark.unit
@pytest.mark.model
def test_maxswin_forward_returns_expected_shape():
    hidden_dim = 40
    num_outputs = 6
    backbone = _DummySwinBackbone(hidden_dim=hidden_dim)
    head = torch.nn.Linear(hidden_dim, num_outputs)
    model = MaxSWIN(swin_backbone=backbone, regression_head=head)

    x = torch.randn(4, 3, 64, 64)
    y = model(x)

    assert y.shape == (4, num_outputs)
