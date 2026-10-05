"""Versioned, self-describing model artifacts with legacy state-dict support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import ExperimentSpec
from .preprocessing import ImagePreprocessor


ARTIFACT_FORMAT = "xrd_geometry"
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ArtifactMetadata:
    experiment: ExperimentSpec
    preprocessor: ImagePreprocessor
    model_config: Mapping[str, Any]
    training: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    format_version: int = ARTIFACT_VERSION


def save_artifact(
    path: str | Path,
    model: torch.nn.Module,
    *,
    experiment: ExperimentSpec,
    preprocessor: ImagePreprocessor,
    model_config: Mapping[str, Any],
    training: Mapping[str, Any] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "model_state": model.state_dict(),
        "experiment": experiment.to_dict(),
        "preprocessor": preprocessor.to_dict(),
        "model_config": dict(model_config),
        "training": dict(training or {}),
        "uncertainty": dict(uncertainty or {}),
    }
    torch.save(payload, path)


def read_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], ArtifactMetadata | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("format") == ARTIFACT_FORMAT:
        version = int(payload["format_version"])
        if version > ARTIFACT_VERSION:
            raise ValueError(f"Artifact version {version} is newer than supported version {ARTIFACT_VERSION}")
        metadata = ArtifactMetadata(
            experiment=ExperimentSpec.from_dict(payload["experiment"]),
            preprocessor=ImagePreprocessor.from_dict(payload["preprocessor"]),
            model_config=payload["model_config"],
            training=payload.get("training", {}),
            uncertainty=payload.get("uncertainty", {}),
            format_version=version,
        )
        return payload["model_state"], metadata
    if not isinstance(payload, dict) or not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError("Checkpoint is neither an xrd_geometry artifact nor a legacy state dict")
    return payload, None


def load_weights(model: torch.nn.Module, path: str | Path, *, strict: bool = True) -> ArtifactMetadata | None:
    state, metadata = read_checkpoint(path)
    model.load_state_dict(state, strict=strict)
    return metadata
