"""Configuration-driven construction without importing optional backbones at package import time."""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch.nn as nn

from .models import DeterministicHead, FullCovarianceHead, ImageRegressor


def _backbone_source(backbone_name: str) -> tuple[str, bool]:
    """Resolve an optional local backbone without changing artifact metadata."""

    override = os.environ.get("XRD_BACKBONE_PATH")
    source = override or backbone_name
    offline = bool(override) or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) == "1"
    return source, offline


def build_model(config: Mapping[str, Any], *, output_dim: int) -> ImageRegressor:
    model_config = config.get("model", config)
    backbone_name = model_config["backbone"]
    image_size = int(model_config.get("image_size", 1056))
    feature_dim = int(model_config.get("feature_dim", model_config.get("hidden_dim")))
    head_hidden_dim = int(model_config.get("head_hidden_dim", 512))
    dropout = float(model_config.get("dropout", 0.1))
    quality_dim = int(model_config.get("quality_dim", 0))
    quality_hidden_dim = int(model_config.get("quality_hidden_dim", 16))
    family = model_config.get("family") or ("swin" if "swin" in backbone_name.lower() else "vit")
    backbone_source, local_files_only = _backbone_source(backbone_name)

    from transformers import SwinConfig, SwinModel, ViTConfig, ViTModel

    if family == "swin":
        backbone_config = SwinConfig.from_pretrained(
            backbone_source,
            local_files_only=local_files_only,
        )
        backbone_config.image_size = image_size
        backbone = SwinModel.from_pretrained(
            backbone_source,
            config=backbone_config,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
    elif family == "vit":
        backbone_config = ViTConfig.from_pretrained(
            backbone_source,
            local_files_only=local_files_only,
        )
        backbone_config.image_size = image_size
        backbone = ViTModel.from_pretrained(
            backbone_source,
            config=backbone_config,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError(f"Unsupported configured backbone family {family!r}")

    head_type = model_config.get("head", "deterministic")
    head: nn.Module
    if head_type == "deterministic":
        head = DeterministicHead(feature_dim, output_dim, hidden_dim=head_hidden_dim, dropout=dropout)
    elif head_type == "full_covariance":
        head = FullCovarianceHead(
            feature_dim,
            output_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            quality_dim=quality_dim,
            quality_hidden_dim=quality_hidden_dim,
        )
    else:
        raise ValueError(f"Unsupported head type {head_type!r}")
    return ImageRegressor(backbone, head, family=family)
