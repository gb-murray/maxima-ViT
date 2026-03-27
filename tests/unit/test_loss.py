import pytest
import torch

from src.loss import Loss


@pytest.mark.unit
def test_loss_zero_for_identical_predictions():
    criterion = Loss()
    y = torch.tensor([[0.15, 0.0425, 0.0375, 0.0, 0.0, 0.0]], dtype=torch.float32)
    loss = criterion(y, y)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.unit
def test_loss_is_positive_for_mismatched_predictions():
    criterion = Loss()
    y_true = torch.zeros((2, 6), dtype=torch.float32)
    y_pred = torch.ones((2, 6), dtype=torch.float32)
    loss = criterion(y_pred, y_true)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


@pytest.mark.unit
def test_loss_backward_produces_gradients():
    criterion = Loss()
    y_true = torch.zeros((4, 6), dtype=torch.float32)
    y_pred = torch.randn((4, 6), dtype=torch.float32, requires_grad=True)

    loss = criterion(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
