"""Domain contracts shared by XRD data, models, and saved artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of one physical parameter in the inverse problem."""

    name: str
    unit: str
    center: float
    scale: float
    inferred: bool = True
    fixed_value: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"Parameter {self.name!r} must have a positive scale")
        if not self.inferred and self.fixed_value is None:
            raise ValueError(f"Fixed parameter {self.name!r} requires fixed_value")


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered physical parameter schema and its normalization transform."""

    parameters: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique")
        if not self.active:
            raise ValueError("At least one inferred parameter is required")

    @property
    def active(self) -> tuple[ParameterSpec, ...]:
        return tuple(parameter for parameter in self.parameters if parameter.inferred)

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.active)

    @property
    def active_indices(self) -> tuple[int, ...]:
        return tuple(index for index, parameter in enumerate(self.parameters) if parameter.inferred)

    @property
    def dimension(self) -> int:
        return len(self.active)

    def _stats(self, *, tensor: bool, reference: torch.Tensor | None = None):
        centers = [parameter.center for parameter in self.active]
        scales = [parameter.scale for parameter in self.active]
        if tensor:
            kwargs = {}
            if reference is not None:
                kwargs = {"device": reference.device, "dtype": reference.dtype}
            return torch.tensor(centers, **kwargs), torch.tensor(scales, **kwargs)
        return np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)

    def select_active(self, values: np.ndarray | torch.Tensor):
        if values.shape[-1] == self.dimension:
            return values
        if values.shape[-1] != len(self.parameters):
            raise ValueError(
                f"Expected {self.dimension} active or {len(self.parameters)} full parameters, "
                f"received {values.shape[-1]}"
            )
        return values[..., list(self.active_indices)]

    def normalize(self, values: np.ndarray | torch.Tensor):
        active = self.select_active(values)
        centers, scales = self._stats(tensor=torch.is_tensor(active), reference=active if torch.is_tensor(active) else None)
        return (active - centers) / scales

    def denormalize(self, values: np.ndarray | torch.Tensor):
        if values.shape[-1] != self.dimension:
            raise ValueError(f"Expected last dimension {self.dimension}, received {values.shape[-1]}")
        centers, scales = self._stats(tensor=torch.is_tensor(values), reference=values if torch.is_tensor(values) else None)
        return values * scales + centers

    def covariance_to_physical(self, covariance: torch.Tensor) -> torch.Tensor:
        if covariance.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("Covariance shape does not match active parameter dimension")
        _, scales = self._stats(tensor=True, reference=covariance)
        return covariance * scales.unsqueeze(-1) * scales.unsqueeze(-2)

    def expand(self, active_values: Sequence[float] | np.ndarray) -> dict[str, float]:
        values = np.asarray(active_values, dtype=float)
        if values.shape != (self.dimension,):
            raise ValueError(f"Expected {self.dimension} active values, received shape {values.shape}")
        active_iter = iter(values.tolist())
        return {
            parameter.name: float(next(active_iter) if parameter.inferred else parameter.fixed_value)
            for parameter in self.parameters
        }

    def to_dict(self) -> dict[str, Any]:
        return {"parameters": [asdict(parameter) for parameter in self.parameters]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterSpace":
        return cls(tuple(ParameterSpec(**item) for item in data["parameters"]))

    @classmethod
    def xrd_five_parameter(
        cls,
        centers: Sequence[float],
        scales: Sequence[float],
        *,
        rot3: float = 0.0,
    ) -> "ParameterSpace":
        if len(centers) != len(scales) or len(centers) not in (5, 6):
            raise ValueError("XRD normalization requires five active values or six legacy values")
        names = ("dist", "poni1", "poni2", "rot1", "rot2")
        units = ("m", "m", "m", "rad", "rad")
        parameters = [
            ParameterSpec(name, unit, float(centers[i]), float(scales[i]))
            for i, (name, unit) in enumerate(zip(names, units))
        ]
        parameters.append(ParameterSpec("rot3", "rad", rot3, 1.0, inferred=False, fixed_value=rot3))
        return cls(tuple(parameters))


@dataclass(frozen=True)
class ExperimentSpec:
    """Fixed assumptions that define where an image-only model is valid."""

    name: str
    detector: str
    wavelengths_m: tuple[float, ...]
    parameter_space: ParameterSpace
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detector": self.detector,
            "wavelengths_m": list(self.wavelengths_m),
            "parameter_space": self.parameter_space.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        return cls(
            name=data["name"],
            detector=data["detector"],
            wavelengths_m=tuple(float(value) for value in data["wavelengths_m"]),
            parameter_space=ParameterSpace.from_dict(data["parameter_space"]),
            metadata=data.get("metadata", {}),
        )
