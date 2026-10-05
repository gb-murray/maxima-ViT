"""One preprocessing implementation shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ImagePreprocessor:
    image_size: int = 1056
    lower_percentile: float = 0.1
    upper_percentile: float = 99.9
    mask_intensity: int | float = 4294967295
    channels: int = 3
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Percentiles must be ordered within [0, 100]")
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ValueError("Channel statistics do not match channels")

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32).copy()
        invalid = ~np.isfinite(array)
        inferred_mask = array == self.mask_intensity
        combined_mask = invalid | inferred_mask
        if mask is not None:
            if mask.shape != array.shape:
                raise ValueError("Mask shape must match image shape")
            combined_mask |= mask.astype(bool)

        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        array = np.clip(array, 0, None)
        array = np.log1p(array)
        valid = array[~combined_mask]
        if valid.size:
            low, high = np.percentile(valid, (self.lower_percentile, self.upper_percentile))
        else:
            low, high = 0.0, 1.0
        if high > low:
            array = np.clip((array - low) / (high - low), 0.0, 1.0)
        else:
            array = np.zeros_like(array)
        array[combined_mask] = 0.0
        array = array.astype(np.float32, copy=False)

        tensor = torch.from_numpy(array).unsqueeze(0).repeat(self.channels, 1, 1)
        _, height, width = tensor.shape
        size = max(height, width)
        tensor = F.pad(tensor, (0, size - width, 0, size - height))
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        mean = torch.tensor(self.mean, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype).view(-1, 1, 1)
        return (tensor - mean) / std

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImagePreprocessor":
        values = dict(data)
        for key in ("mean", "std"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)
