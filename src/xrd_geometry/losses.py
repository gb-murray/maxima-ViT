"""Objectives for deterministic and probabilistic training stages."""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn


class WeightedRegressionLoss(nn.Module):
    def __init__(self, kind: Literal["l1", "l2", "huber"], weights: Sequence[float], *, delta: float = 1.0):
        super().__init__()
        self.kind = kind
        self.delta = delta
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "l1":
            elementwise = torch.abs(prediction - target)
        elif self.kind == "l2":
            elementwise = torch.square(prediction - target)
        elif self.kind == "huber":
            elementwise = torch.nn.functional.huber_loss(prediction, target, delta=self.delta, reduction="none")
        else:
            raise ValueError(f"Unsupported deterministic loss {self.kind!r}")
        if elementwise.shape[-1] != self.weights.numel():
            raise ValueError("Loss weights do not match prediction dimension")
        return (elementwise * self.weights).mean()


class MultivariateGaussianNLL(nn.Module):
    """Stable full-covariance Gaussian negative log likelihood."""

    def forward(self, output: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        mean, scale_tril = output
        # Triangular solves and log determinants are sensitive to half precision.
        # Keep the probabilistic algebra in FP32 even when the model forward pass
        # runs under CUDA autocast.
        with torch.autocast(device_type=mean.device.type, enabled=False):
            distribution = torch.distributions.MultivariateNormal(
                mean.float(), scale_tril=scale_tril.float()
            )
            return -distribution.log_prob(target.float()).mean()
