"""Train deterministic and probabilistic XRD geometry stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..checkpoints import read_checkpoint, save_artifact
from ..config import build_experiment, build_preprocessor, load_config, resolve_dataset_configs
from ..data import HDF5DiffractionDataset
from ..factory import build_model
from ..models import FullCovarianceHead
from ..training import MixedPrecisionConfig, TrainingStage, run_stages


def train_from_config(
    config: dict,
    *,
    checkpoint: str | None = None,
    start_stage: str | None = None,
    dry_run: bool = False,
):
    experiment = build_experiment(config)
    preprocessor = build_preprocessor(config)
    parameter_space = experiment.parameter_space
    seed = int(config.get("seed", 0))

    training_config = config["training"]
    workers = 0 if dry_run else int(training_config.get("num_workers", 0))
    persistent_workers = bool(training_config.get("persistent_workers", False))
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(training_config.get("batch_size", 8)),
        "num_workers": workers,
        "pin_memory": bool(training_config.get("pin_memory", torch.cuda.is_available())),
        "persistent_workers": workers > 0 and persistent_workers,
    }
    dataset_configs = resolve_dataset_configs(config)
    if persistent_workers and any(definition.get("randomize") for definition in dataset_configs.values()):
        raise ValueError("persistent_workers cannot propagate epoch state to randomized datasets")
    datasets = {
        name: HDF5DiffractionDataset(
            definition["hdf5_path"],
            definition["group"],
            parameter_space,
            preprocessor,
            randomize=definition.get("randomize"),
            seed=seed,
        )
        for name, definition in dataset_configs.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            shuffle=bool(dataset_configs[name].get("shuffle", False)),
            generator=generator if dataset_configs[name].get("shuffle", False) else None,
            **loader_args,
        )
        for name, dataset in datasets.items()
    }

    stages = tuple(TrainingStage.from_dict(stage) for stage in training_config["stages"])
    mixed_precision = MixedPrecisionConfig.from_dict(training_config.get("mixed_precision"))
    model_config = dict(config["model"])
    checkpoint_state = None
    checkpoint_metadata = None
    if checkpoint:
        checkpoint_state, checkpoint_metadata = read_checkpoint(checkpoint)
        if checkpoint_metadata is not None:
            model_config = dict(checkpoint_metadata.model_config)
    if start_stage is not None:
        matching_indices = [index for index, stage in enumerate(stages) if stage.name == start_stage]
        if not matching_indices:
            raise ValueError(f"Unknown start stage {start_stage!r}")
        stages = stages[matching_indices[0] :]
    elif model_config.get("head") == "full_covariance":
        stages = tuple(stage for stage in stages if stage.regime == "probabilistic")
    if not stages:
        raise ValueError("No training stages remain after applying resume settings")
    model_config.setdefault("head", "full_covariance" if stages[0].regime == "probabilistic" else "deterministic")
    model = build_model(model_config, output_dim=parameter_space.dimension)
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(config["paths"]["output_dir"]) / config["paths"].get("artifact_name", "xrd_geometry.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output_path.parent / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))
    amp_active = mixed_precision.active_on(device)
    print(f"Using device: {device}", flush=True)
    print(
        f"Mixed precision: {'enabled (' + mixed_precision.dtype + ')' if amp_active else 'disabled'}",
        flush=True,
    )
    print(f"TensorBoard logs: {log_dir}", flush=True)
    try:
        results = run_stages(
            model,
            loaders,
            loaders,
            stages,
            device=device,
            feature_dim=int(model_config.get("feature_dim", model_config.get("hidden_dim"))),
            output_dim=parameter_space.dimension,
            hidden_dim=int(model_config.get("head_hidden_dim", 512)),
            dropout=float(model_config.get("dropout", 0.1)),
            dry_run=dry_run,
            mixed_precision=mixed_precision,
            writer=writer,
            progress=True,
        )
        model_config["head"] = "full_covariance" if isinstance(model.head, FullCovarianceHead) else "deterministic"
        save_artifact(
            output_path,
            model,
            experiment=experiment,
            preprocessor=preprocessor,
            model_config=model_config,
            training={
                "stages": [asdict(stage) for stage in stages],
                "results": [asdict(result) for result in results],
                "dry_run": dry_run,
                "device_type": device.type,
                "mixed_precision": {
                    **asdict(mixed_precision),
                    "active": amp_active,
                },
                "tensorboard_log_dir": str(log_dir),
            },
            uncertainty=config.get("uncertainty", {}),
        )
    finally:
        writer.close()
        for dataset in datasets.values():
            dataset.close()
    return output_path, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XRD detector-geometry model")
    parser.add_argument("config", help="Path to a YAML configuration")
    parser.add_argument("--load-checkpoint", help="Artifact or legacy state dict to resume")
    parser.add_argument("--start-stage", help="Resume at this configured stage name")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch and epoch per stage")
    arguments = parser.parse_args()
    output, results = train_from_config(
        load_config(arguments.config),
        checkpoint=arguments.load_checkpoint,
        start_stage=arguments.start_stage,
        dry_run=arguments.dry_run,
    )
    for result in results:
        status = "early stop" if result.stopped_early else "complete"
        print(
            f"{result.name}: train={result.train_losses[-1]:.6f}, "
            f"validation={result.validation_losses[-1]:.6f}, "
            f"best={result.best_validation_loss:.6f} at epoch {result.best_epoch} ({status})"
        )
    print(f"Saved artifact: {output}")


if __name__ == "__main__":
    main()
