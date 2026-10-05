"""Configurable deterministic and probabilistic training stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import torch
from torch.optim import AdamW

from .losses import MultivariateGaussianNLL, WeightedRegressionLoss
from .models import (
    DeterministicHead,
    FullCovarianceHead,
    ImageRegressor,
    initialize_probabilistic_head,
    set_trainable_scope,
)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int
    min_delta: float = 0.0
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("Early-stopping patience must be non-negative")
        if self.min_delta < 0:
            raise ValueError("Early-stopping min_delta must be non-negative")


@dataclass(frozen=True)
class MixedPrecisionConfig:
    """CUDA automatic mixed-precision policy."""

    enabled: bool = True
    dtype: Literal["float16", "bfloat16"] = "float16"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MixedPrecisionConfig":
        return cls(**dict(data or {}))

    def __post_init__(self) -> None:
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("Mixed-precision dtype must be 'float16' or 'bfloat16'")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.dtype == "float16" else torch.bfloat16

    def active_on(self, device: torch.device) -> bool:
        return self.enabled and device.type == "cuda"


@dataclass(frozen=True)
class TrainingStage:
    name: str
    regime: Literal["deterministic", "probabilistic"]
    epochs: int
    learning_rate: float
    trainable: Literal["head", "backbone", "all"] = "all"
    loss: Literal["l1", "l2", "huber", "gaussian_nll"] = "huber"
    weights: tuple[float, ...] | None = None
    huber_delta: float = 1.0
    weight_decay: float = 0.01
    dataset: str = "train"
    validation_dataset: str = "validation"
    early_stopping: EarlyStoppingConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingStage":
        values = dict(data)
        if "weights" in values:
            values["weights"] = tuple(float(value) for value in values["weights"])
        if values.get("early_stopping") is not None:
            values["early_stopping"] = EarlyStoppingConfig(**values["early_stopping"])
        return cls(**values)


@dataclass(frozen=True)
class StageResult:
    name: str
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool


def build_loss(stage: TrainingStage, dimension: int) -> torch.nn.Module:
    if stage.loss == "gaussian_nll":
        if stage.regime != "probabilistic":
            raise ValueError("gaussian_nll requires a probabilistic stage")
        return MultivariateGaussianNLL()
    weights = stage.weights or tuple(1.0 for _ in range(dimension))
    return WeightedRegressionLoss(stage.loss, weights, delta=stage.huber_delta)


def _mean_epoch_loss(
    model: ImageRegressor,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    max_batches: int | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    training = optimizer is not None
    model.train(training)
    if training:
        if not any(parameter.requires_grad for parameter in model.feature_extractor.parameters()):
            model.feature_extractor.eval()
        if not any(parameter.requires_grad for parameter in model.head.parameters()):
            model.head.eval()
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = loss_fn(model(images), targets)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            total += float(loss.detach())
            count += 1
    if not count:
        raise ValueError("Data loader yielded no batches")
    return total / count


def _set_loader_epoch(loader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _ensure_regime(
    model: ImageRegressor,
    stage: TrainingStage,
    *,
    feature_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
) -> None:
    if stage.regime == "deterministic":
        if not isinstance(model.head, DeterministicHead):
            raise ValueError("Cannot return to deterministic training after probabilistic conversion")
        return
    if isinstance(model.head, FullCovarianceHead):
        return
    if not isinstance(model.head, DeterministicHead):
        raise TypeError("Probabilistic conversion requires a deterministic head")
    old_head = model.head
    new_head = FullCovarianceHead(feature_dim, output_dim, hidden_dim=hidden_dim, dropout=dropout)
    new_head.to(next(model.parameters()).device)
    initialize_probabilistic_head(new_head, old_head, active_indices=tuple(range(output_dim)))
    model.head = new_head


def run_stages(
    model: ImageRegressor,
    train_loader,
    validation_loader,
    stages: Iterable[TrainingStage],
    *,
    device: torch.device,
    feature_dim: int,
    output_dim: int,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    dry_run: bool = False,
    mixed_precision: MixedPrecisionConfig | None = None,
    writer: Any | None = None,
    progress: bool = False,
) -> tuple[StageResult, ...]:
    model.to(device)
    precision = mixed_precision or MixedPrecisionConfig()
    amp_enabled = precision.active_on(device)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if amp_enabled and precision.dtype == "float16"
        else None
    )
    results = []
    epoch_index = 0
    for stage in stages:
        stage_train_loader = train_loader[stage.dataset] if isinstance(train_loader, Mapping) else train_loader
        stage_validation_loader = (
            validation_loader[stage.validation_dataset]
            if isinstance(validation_loader, Mapping)
            else validation_loader
        )
        _ensure_regime(
            model,
            stage,
            feature_dim=feature_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        set_trainable_scope(model, stage.trainable)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError(f"Stage {stage.name!r} has no trainable parameters")
        optimizer = AdamW(trainable, lr=stage.learning_rate, weight_decay=stage.weight_decay)
        loss_fn = build_loss(stage, output_dim).to(device)
        train_losses, validation_losses = [], []
        best_validation_loss = float("inf")
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        stopped_early = False
        epochs = 1 if dry_run else stage.epochs
        if progress:
            precision_label = f"AMP/{precision.dtype}" if amp_enabled else "float32"
            print(
                f"Starting stage {stage.name!r}: regime={stage.regime}, "
                f"trainable={stage.trainable}, precision={precision_label}",
                flush=True,
            )
        for stage_epoch in range(epochs):
            _set_loader_epoch(stage_train_loader, epoch_index)
            train_losses.append(
                _mean_epoch_loss(
                    model,
                    stage_train_loader,
                    loss_fn,
                    device,
                    optimizer,
                    max_batches=1 if dry_run else None,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    amp_dtype=precision.torch_dtype,
                )
            )
            validation_losses.append(
                _mean_epoch_loss(
                    model,
                    stage_validation_loader,
                    loss_fn,
                    device,
                    None,
                    max_batches=1 if dry_run else None,
                    scaler=None,
                    amp_enabled=amp_enabled,
                    amp_dtype=precision.torch_dtype,
                )
            )
            validation_loss = validation_losses[-1]
            if writer is not None:
                writer.add_scalar(f"loss/{stage.name}/train", train_losses[-1], stage_epoch + 1)
                writer.add_scalar(f"loss/{stage.name}/validation", validation_loss, stage_epoch + 1)
                writer.add_scalar(
                    f"learning_rate/{stage.name}", optimizer.param_groups[0]["lr"], stage_epoch + 1
                )
                writer.flush()
            if progress:
                print(
                    f"{stage.name} epoch {stage_epoch + 1}/{epochs}: "
                    f"train={train_losses[-1]:.6f}, validation={validation_loss:.6f}",
                    flush=True,
                )
            minimum_improvement = stage.early_stopping.min_delta if stage.early_stopping else 0.0
            if validation_loss < best_validation_loss - minimum_improvement:
                best_validation_loss = validation_loss
                best_epoch = stage_epoch + 1
                epochs_without_improvement = 0
                if stage.early_stopping and stage.early_stopping.restore_best:
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            else:
                epochs_without_improvement += 1
            epoch_index += 1
            if (
                stage.early_stopping is not None
                and epochs_without_improvement > 0
                and epochs_without_improvement >= stage.early_stopping.patience
            ):
                stopped_early = True
                if progress:
                    print(
                        f"{stage.name}: early stopping after epoch {stage_epoch + 1}; "
                        f"best validation={best_validation_loss:.6f} at epoch {best_epoch}",
                        flush=True,
                    )
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        results.append(
            StageResult(
                stage.name,
                tuple(train_losses),
                tuple(validation_losses),
                best_epoch,
                best_validation_loss,
                stopped_early,
            )
        )
    return tuple(results)
