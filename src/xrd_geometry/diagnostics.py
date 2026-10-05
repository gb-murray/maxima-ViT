"""Input-space diagnostics for network predictions and uncertainty."""

from __future__ import annotations

import torch

from .models import ImageRegressor, enable_dropout


def parameter_jacobians(model: ImageRegressor, images: torch.Tensor) -> torch.Tensor:
    """Return d(mean_j)/d(image) for every batch item and active parameter."""

    model.eval()
    inputs = images.detach().clone().requires_grad_(True)
    output = model(inputs)
    means = output[0] if isinstance(output, tuple) else output
    maps = []
    for parameter_index in range(means.shape[-1]):
        gradient = torch.autograd.grad(
            means[:, parameter_index].sum(),
            inputs,
            retain_graph=parameter_index < means.shape[-1] - 1,
        )[0]
        maps.append(gradient)
    return torch.stack(maps, dim=1)


def uncertainty_sensitivity(
    model: ImageRegressor,
    images: torch.Tensor,
    *,
    samples: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    """Gradient of normalized total covariance trace with respect to the input."""

    if samples < 2:
        raise ValueError("At least two samples are required for uncertainty sensitivity")
    model.eval()
    enable_dropout(model.head)
    inputs = images.detach().clone().requires_grad_(True)
    features = model.extract_features(inputs)
    devices = [inputs.device.index] if inputs.is_cuda and inputs.device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        outputs = [model.forward_head(features) for _ in range(samples)]
    if not all(isinstance(output, tuple) for output in outputs):
        raise TypeError("Uncertainty sensitivity requires a probabilistic head")
    means = torch.stack([output[0] for output in outputs])
    factors = torch.stack([output[1] for output in outputs])
    conditional_trace = torch.diagonal(factors @ factors.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1).mean(0)
    model_trace = means.var(dim=0, unbiased=True).sum(-1)
    score = conditional_trace + model_trace
    return torch.autograd.grad(score.sum(), inputs)[0]
