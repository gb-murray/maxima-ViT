"""Backbone-independent deterministic and probabilistic regressors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


class DeterministicHead(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class FullCovarianceHead(nn.Module):
    """Predict a mean and Cholesky factor in normalized target coordinates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        minimum_diagonal: float = 1e-4,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.minimum_diagonal = minimum_diagonal
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.cholesky = nn.Linear(hidden_dim, output_dim * (output_dim + 1) // 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        mean = self.mean(features)
        raw = self.cholesky(features)
        batch_shape = raw.shape[:-1]
        factor = raw.new_zeros((*batch_shape, self.output_dim, self.output_dim))
        rows, columns = torch.tril_indices(self.output_dim, self.output_dim, device=raw.device)
        factor[..., rows, columns] = raw
        diagonal = torch.diagonal(factor, dim1=-2, dim2=-1)
        positive_diagonal = torch.nn.functional.softplus(diagonal) + self.minimum_diagonal
        factor = factor - torch.diag_embed(diagonal) + torch.diag_embed(positive_diagonal)
        return mean, factor


class ImageRegressor(nn.Module):
    """Compose an image backbone with a replaceable regression head."""

    def __init__(self, backbone: nn.Module, head: nn.Module, *, family: Literal["swin", "vit", "generic"] = "generic"):
        super().__init__()
        self.family = family
        if family == "swin":
            self.swin = backbone
        elif family == "vit":
            self.vit = backbone
        else:
            self.backbone = backbone
        self.head = head
        self.pooler = nn.AdaptiveAvgPool1d(1)

    @property
    def feature_extractor(self) -> nn.Module:
        return getattr(self, self.family if self.family in ("swin", "vit") else "backbone")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.family in ("swin", "vit"):
            outputs = self.feature_extractor(pixel_values=images)
            hidden = outputs.last_hidden_state
            if self.family == "vit":
                return hidden[:, 0, :]
            return self.pooler(hidden.transpose(1, 2)).flatten(1)
        features = self.feature_extractor(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def forward_head(self, features: torch.Tensor):
        return self.head(features)

    def forward(self, images: torch.Tensor):
        return self.forward_head(self.extract_features(images))


def set_trainable_scope(model: ImageRegressor, scope: Literal["head", "backbone", "all"]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"
    if scope == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    elif scope == "backbone":
        for parameter in model.feature_extractor.parameters():
            parameter.requires_grad = True


def enable_dropout(module: nn.Module) -> None:
    """Enable dropout layers while leaving all other modules in evaluation mode."""

    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.train()


def initialize_probabilistic_head(
    probabilistic: FullCovarianceHead,
    deterministic: DeterministicHead,
    *,
    active_indices: tuple[int, ...],
    initial_variance: float = 1.0,
) -> None:
    """Copy a legacy deterministic MLP into a full-covariance head."""

    if len(deterministic) != 4:
        raise ValueError("Expected the legacy Linear/GELU/Dropout/Linear head layout")
    with torch.no_grad():
        probabilistic.features[0].weight.copy_(deterministic[0].weight)
        probabilistic.features[0].bias.copy_(deterministic[0].bias)
        probabilistic.mean.weight.copy_(deterministic[3].weight[list(active_indices)])
        probabilistic.mean.bias.copy_(deterministic[3].bias[list(active_indices)])
        probabilistic.cholesky.weight.zero_()
        probabilistic.cholesky.bias.zero_()
        desired_diagonal = math.sqrt(initial_variance) - probabilistic.minimum_diagonal
        if desired_diagonal <= 0:
            raise ValueError("initial_variance is below the minimum covariance diagonal")
        inverse_softplus = math.log(math.expm1(desired_diagonal))
        diagonal_positions = []
        offset = 0
        for row in range(probabilistic.output_dim):
            diagonal_positions.append(offset + row)
            offset += row + 1
        probabilistic.cholesky.bias[diagonal_positions] = inverse_softplus
