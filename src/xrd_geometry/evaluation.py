"""Metrics for point predictions and full-covariance Gaussian predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PointMetrics:
    mae: torch.Tensor
    rmse: torch.Tensor


@dataclass(frozen=True)
class GaussianMetrics:
    nll: float
    marginal_coverage: dict[float, torch.Tensor]
    marginal_width: dict[float, torch.Tensor]
    joint_coverage: dict[float, float]


def point_metrics(mean: torch.Tensor, target: torch.Tensor) -> PointMetrics:
    if mean.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    residual = mean - target
    return PointMetrics(
        mae=residual.abs().mean(dim=0),
        rmse=residual.square().mean(dim=0).sqrt(),
    )


def gaussian_metrics(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> GaussianMetrics:
    """Evaluate calibrated intervals and ellipsoids in physical target units."""

    if mean.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    dimension = mean.shape[-1]
    if covariance.shape != (*mean.shape[:-1], dimension, dimension):
        raise ValueError("Covariance shape does not match predictions")
    checked_levels = tuple(float(level) for level in levels)
    if any(not 0 < level < 1 for level in checked_levels):
        raise ValueError("Coverage levels must lie strictly between zero and one")

    distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)
    nll = float((-distribution.log_prob(target)).mean())
    standard_deviation = torch.diagonal(covariance, dim1=-2, dim2=-1).sqrt()
    absolute_residual = (target - mean).abs()
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, device=mean.device, dtype=mean.dtype),
        torch.tensor(1.0, device=mean.device, dtype=mean.dtype),
    )
    residual = (target - mean).unsqueeze(-1)
    solved = torch.linalg.solve(covariance, residual)
    mahalanobis_squared = (residual.transpose(-1, -2) @ solved).squeeze(-1).squeeze(-1)

    marginal_coverage = {}
    marginal_width = {}
    joint_coverage = {}
    for level in checked_levels:
        probability = torch.tensor((1.0 + level) / 2.0, device=mean.device, dtype=mean.dtype)
        z_value = standard_normal.icdf(probability)
        half_width = z_value * standard_deviation
        marginal_coverage[level] = (absolute_residual <= half_width).to(mean.dtype).mean(dim=0)
        marginal_width[level] = (2.0 * half_width).mean(dim=0)

        chi_square = torch.distributions.Chi2(
            torch.tensor(float(dimension), device=mean.device, dtype=mean.dtype)
        )
        try:
            threshold = chi_square.icdf(torch.tensor(level, device=mean.device, dtype=mean.dtype))
        except NotImplementedError:
            from scipy.stats import chi2

            threshold = torch.tensor(
                chi2.ppf(level, dimension),
                device=mean.device,
                dtype=mean.dtype,
            )
        joint_coverage[level] = float((mahalanobis_squared <= threshold).to(mean.dtype).mean())
    return GaussianMetrics(nll, marginal_coverage, marginal_width, joint_coverage)
