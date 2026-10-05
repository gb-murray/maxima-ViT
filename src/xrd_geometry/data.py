"""Storage adapters and measurement transforms for diffraction images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import ParameterSpace
from .preprocessing import ImagePreprocessor


@dataclass(frozen=True)
class MeasurementSample:
    """A generated or observed measurement before model preprocessing."""

    image: np.ndarray
    target: np.ndarray
    index: int
    group: str


class MeasurementTransform(Protocol):
    """Converts an expected detector image into a sampled measurement."""

    def __call__(self, expected_counts: np.ndarray, *, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class PoissonMeasurement:
    """Photon-counting measurement with clipping and an optional hardware mask."""

    mask: np.ndarray | None = None
    mask_intensity: int | float = 4294967295
    saturation: float | None = None

    def __call__(self, expected_counts: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        expectation = np.clip(expected_counts, 0, None)
        measured = rng.poisson(expectation).astype(np.float32)
        if self.saturation is not None:
            measured = np.minimum(measured, self.saturation)
        if self.mask is not None:
            measured[self.mask] = self.mask_intensity
        return measured


class EmpiricalBackgroundRandomizer:
    """Reconstruct an empirical NMF background using legacy count-space behavior."""

    def __init__(
        self,
        components: np.ndarray,
        weights: np.ndarray,
        global_range: tuple[float, float] = (0.9, 1.1),
        component_range: tuple[float, float] = (0.95, 1.05),
    ) -> None:
        if components.shape[0] != weights.shape[1]:
            raise ValueError("Background components and weights are incompatible")
        self.components = components
        self.weights = weights
        self.global_range = global_range
        self.component_range = component_range

    def __call__(self, *, rng: np.random.Generator) -> np.ndarray:
        index = int(rng.integers(0, self.weights.shape[0]))
        base = self.weights[index].copy()
        global_scale = rng.uniform(*self.global_range)
        component_scale = rng.uniform(*self.component_range, size=base.shape)
        return np.tensordot(base * global_scale * component_scale, self.components, axes=1)


class HDF5DiffractionDataset(Dataset):
    """Read legacy XRD HDF5 files and optionally randomize cached clean signals."""

    def __init__(
        self,
        path: str | Path,
        group: str,
        parameter_space: ParameterSpace,
        preprocessor: ImagePreprocessor,
        *,
        randomize: bool | None = None,
        seed: int = 0,
    ) -> None:
        self.path = str(path)
        self.group = group
        self.parameter_space = parameter_space
        self.preprocessor = preprocessor
        self.seed = int(seed)
        self.epoch = 0
        self._file: h5py.File | None = None

        with h5py.File(self.path, "r") as handle:
            if group not in handle:
                raise KeyError(f"HDF5 group {group!r} not found")
            self.length = len(handle[group]["images"])
            has_background = "background_model" in handle
            self.randomize = (group == "train" and has_background) if randomize is None else randomize
            if self.randomize and not has_background:
                raise ValueError("Randomization requested but background_model is absent")
            if has_background:
                background = handle["background_model"]
                components = np.asarray(background["H"], dtype=np.float32)
                sample_shape = tuple(handle[group]["images"].shape[1:])
                if components.ndim == 2:
                    components = components.reshape((components.shape[0], *sample_shape))
                self.background_randomizer = EmpiricalBackgroundRandomizer(
                    components,
                    np.asarray(background["W"], dtype=np.float32),
                )
                self.master_mask = np.asarray(background["master_mask"], dtype=bool)
                self.mask_intensity = float(np.asarray(background["mask_intensity"]).reshape(-1)[0])
            else:
                self.background_randomizer = None
                self.master_mask = None
                self.mask_intensity = preprocessor.mask_intensity

    def __len__(self) -> int:
        return self.length

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible but new domain-randomization realization."""

        self.epoch = int(epoch)

    def __del__(self) -> None:
        self.close()

    def raw_sample(self, index: int) -> MeasurementSample:
        handle = self._handle()
        image = np.asarray(handle[self.group]["images"][index], dtype=np.float32)
        target = np.asarray(handle[self.group]["labels"][index], dtype=np.float32)
        if self.randomize:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            expected = image + self.background_randomizer(rng=rng)  # type: ignore[misc]
            image = PoissonMeasurement(self.master_mask, self.mask_intensity)(expected, rng=rng)
        return MeasurementSample(image=image, target=target, index=index, group=self.group)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.raw_sample(index)
        image = self.preprocessor(sample.image, self.master_mask)
        target = torch.as_tensor(self.parameter_space.normalize(sample.target), dtype=torch.float32)
        return image, target


def parameter_space_from_hdf5(path: str | Path, *, rot3: float = 0.0) -> ParameterSpace:
    """Build the five-target schema from a legacy normalization group."""

    with h5py.File(path, "r") as handle:
        if "normalization" not in handle:
            raise ValueError(f"Normalization data not found in HDF5 file: {path}")
        centers = np.asarray(handle["normalization"]["centers"], dtype=np.float32)
        scales = np.asarray(handle["normalization"]["scale_factors"], dtype=np.float32)
    return ParameterSpace.xrd_five_parameter(centers, scales, rot3=rot3)
