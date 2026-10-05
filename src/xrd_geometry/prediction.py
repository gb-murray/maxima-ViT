"""Structured prediction outputs and Monte Carlo dropout inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .contracts import ParameterSpace
from .models import ImageRegressor, enable_dropout


@dataclass(frozen=True)
class GeometryPrediction:
    mean: torch.Tensor
    conditional_covariance: torch.Tensor
    model_covariance: torch.Tensor
    total_covariance: torch.Tensor
    parameter_names: tuple[str, ...]
    units: tuple[str, ...]
    samples: torch.Tensor | None = None


class UncertaintyEstimator(Protocol):
    """Extension point for epistemic estimators used by an XRD model."""

    def __call__(
        self,
        model: ImageRegressor,
        images: torch.Tensor,
        parameter_space: ParameterSpace,
    ) -> GeometryPrediction: ...


@dataclass(frozen=True)
class MCDropoutEstimator:
    samples: int = 32
    stochastic_scope: str = "head"
    return_samples: bool = False

    def __call__(
        self,
        model: ImageRegressor,
        images: torch.Tensor,
        parameter_space: ParameterSpace,
    ) -> GeometryPrediction:
        return predict_with_uncertainty(
            model,
            images,
            parameter_space,
            samples=self.samples,
            stochastic_scope=self.stochastic_scope,
            return_samples=self.return_samples,
        )


def build_estimator(config: dict | None = None) -> UncertaintyEstimator:
    values = dict(config or {})
    name = values.pop("estimator", "mc_dropout")
    if name != "mc_dropout":
        raise ValueError(f"Unsupported uncertainty estimator {name!r}")
    return MCDropoutEstimator(**values)


def _sample_covariance(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0, keepdim=True)
    denominator = max(samples.shape[0] - 1, 1)
    return torch.einsum("tbi,tbj->bij", centered, centered) / denominator


def predict_with_uncertainty(
    model: ImageRegressor,
    images: torch.Tensor,
    parameter_space: ParameterSpace,
    *,
    samples: int = 32,
    stochastic_scope: str = "head",
    return_samples: bool = False,
) -> GeometryPrediction:
    if samples < 1:
        raise ValueError("samples must be positive")
    if stochastic_scope not in {"head", "all"}:
        raise ValueError("stochastic_scope must be 'head' or 'all'")

    model.eval()
    with torch.no_grad():
        if stochastic_scope == "head":
            features = model.extract_features(images)
            enable_dropout(model.head)
            outputs = [model.forward_head(features) for _ in range(samples)]
        else:
            enable_dropout(model)
            outputs = [model(images) for _ in range(samples)]

    if not all(isinstance(output, tuple) and len(output) == 2 for output in outputs):
        raise TypeError("Uncertainty prediction requires a probabilistic head")
    means = torch.stack([output[0] for output in outputs])
    factors = torch.stack([output[1] for output in outputs])
    conditional = torch.matmul(factors, factors.transpose(-1, -2)).mean(dim=0)
    model_covariance = _sample_covariance(means)
    total = conditional + model_covariance
    mean = means.mean(dim=0)

    physical_mean = parameter_space.denormalize(mean)
    physical_conditional = parameter_space.covariance_to_physical(conditional)
    physical_model = parameter_space.covariance_to_physical(model_covariance)
    physical_total = parameter_space.covariance_to_physical(total)
    physical_samples = parameter_space.denormalize(means) if return_samples else None
    return GeometryPrediction(
        mean=physical_mean,
        conditional_covariance=physical_conditional,
        model_covariance=physical_model,
        total_covariance=physical_total,
        parameter_names=parameter_space.active_names,
        units=tuple(parameter.unit for parameter in parameter_space.active),
        samples=physical_samples,
    )
