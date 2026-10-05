"""Configuration-driven construction without importing optional backbones at package import time."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from .models import DeterministicHead, FullCovarianceHead, ImageRegressor


def build_model(config: Mapping[str, Any], *, output_dim: int) -> ImageRegressor:
    model_config = config.get("model", config)
    backbone_name = model_config["backbone"]
    image_size = int(model_config.get("image_size", 1056))
    feature_dim = int(model_config.get("feature_dim", model_config.get("hidden_dim")))
    head_hidden_dim = int(model_config.get("head_hidden_dim", 512))
    dropout = float(model_config.get("dropout", 0.1))
    family = model_config.get("family") or ("swin" if "swin" in backbone_name.lower() else "vit")

    from transformers import SwinConfig, SwinModel, ViTConfig, ViTModel

    if family == "swin":
        backbone_config = SwinConfig.from_pretrained(backbone_name)
        backbone_config.image_size = image_size
        backbone = SwinModel.from_pretrained(
            backbone_name,
            config=backbone_config,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
    elif family == "vit":
        backbone_config = ViTConfig.from_pretrained(backbone_name)
        backbone_config.image_size = image_size
        backbone = ViTModel.from_pretrained(
            backbone_name,
            config=backbone_config,
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"Unsupported configured backbone family {family!r}")

    head_type = model_config.get("head", "deterministic")
    head: nn.Module
    if head_type == "deterministic":
        head = DeterministicHead(feature_dim, output_dim, hidden_dim=head_hidden_dim, dropout=dropout)
    elif head_type == "full_covariance":
        head = FullCovarianceHead(feature_dim, output_dim, hidden_dim=head_hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unsupported head type {head_type!r}")
    return ImageRegressor(backbone, head, family=family)
