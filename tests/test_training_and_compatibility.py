from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from xrd_geometry.checkpoints import load_weights, read_checkpoint, save_artifact
from xrd_geometry.contracts import ExperimentSpec, ParameterSpace
from xrd_geometry.models import DeterministicHead, FullCovarianceHead, ImageRegressor
from xrd_geometry.losses import MultivariateGaussianNLL
from xrd_geometry.preprocessing import ImagePreprocessor
from xrd_geometry import training as training_module
from xrd_geometry.training import EarlyStoppingConfig, MixedPrecisionConfig, TrainingStage, run_stages

from .helpers import CountingBackbone


def _space():
    return ParameterSpace.xrd_five_parameter([0.1, 0.02, 0.03, 0.0, 0.0], [0.01] * 5)


def _model():
    return ImageRegressor(CountingBackbone(), DeterministicHead(6, 5, hidden_dim=10, dropout=0.2))


def test_deterministic_to_probabilistic_dry_run():
    torch.manual_seed(3)
    model = _model()
    loader = DataLoader(TensorDataset(torch.randn(6, 3, 8, 8), torch.randn(6, 5)), batch_size=2)
    stages = [
        TrainingStage("deterministic", "deterministic", 5, 1e-3, loss="huber"),
        TrainingStage("probabilistic", "probabilistic", 5, 1e-3, trainable="head", loss="gaussian_nll"),
    ]
    results = run_stages(
        model,
        loader,
        loader,
        stages,
        device=torch.device("cpu"),
        feature_dim=6,
        output_dim=5,
        hidden_dim=10,
        dropout=0.2,
        dry_run=True,
    )
    assert len(results) == 2
    assert all(len(result.train_losses) == 1 for result in results)
    assert isinstance(model.head, FullCovarianceHead)
    assert all(not parameter.requires_grad for parameter in model.feature_extractor.parameters())


def test_versioned_artifact_round_trip(tmp_path: Path):
    model = _model()
    space = _space()
    experiment = ExperimentSpec("fixture", "detector", (1e-10,), space)
    processor = ImagePreprocessor(image_size=8)
    path = tmp_path / "artifact.pt"
    save_artifact(
        path,
        model,
        experiment=experiment,
        preprocessor=processor,
        model_config={"backbone": "fixture", "feature_dim": 6, "head": "deterministic"},
        uncertainty={"estimator": "mc_dropout", "samples": 7, "stochastic_scope": "head"},
    )
    state, metadata = read_checkpoint(path)
    assert metadata is not None
    assert metadata.experiment == experiment
    assert metadata.uncertainty["samples"] == 7
    clone = _model()
    loaded_metadata = load_weights(clone, path)
    assert loaded_metadata == metadata
    for key, value in model.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[key], value)


def test_legacy_raw_state_dict_is_supported(tmp_path: Path):
    model = _model()
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)
    state, metadata = read_checkpoint(path)
    assert metadata is None
    assert set(state) == set(model.state_dict())


def test_stage_early_stopping_restores_best_weights(monkeypatch):
    model = _model()
    loader = DataLoader(TensorDataset(torch.zeros(1, 3, 8, 8), torch.zeros(1, 5)), batch_size=1)
    tracked_parameter = model.head[0].weight
    initial_value = tracked_parameter.detach().clone()
    validation_losses = iter((1.0, 1.1, 1.2, 0.5))

    def fake_epoch_loss(
        model,
        loader,
        loss_fn,
        device,
        optimizer,
        *,
        max_batches,
        scaler,
        amp_enabled,
        amp_dtype,
    ):
        if optimizer is not None:
            with torch.no_grad():
                tracked_parameter.add_(1.0)
            return 0.0
        return next(validation_losses)

    monkeypatch.setattr(training_module, "_mean_epoch_loss", fake_epoch_loss)
    stage = TrainingStage(
        "early-stop",
        "deterministic",
        epochs=10,
        learning_rate=1e-3,
        early_stopping=EarlyStoppingConfig(patience=2, min_delta=0.0, restore_best=True),
    )
    result = run_stages(
        model,
        loader,
        loader,
        [stage],
        device=torch.device("cpu"),
        feature_dim=6,
        output_dim=5,
        hidden_dim=10,
    )[0]

    assert result.stopped_early is True
    assert result.best_epoch == 1
    assert result.best_validation_loss == 1.0
    assert len(result.validation_losses) == 3
    torch.testing.assert_close(tracked_parameter, initial_value + 1.0)


def test_stage_config_parses_early_stopping():
    stage = TrainingStage.from_dict(
        {
            "name": "configured",
            "regime": "deterministic",
            "epochs": 10,
            "learning_rate": 1e-3,
            "early_stopping": {"patience": 3, "min_delta": 0.01, "restore_best": False},
        }
    )
    assert stage.early_stopping == EarlyStoppingConfig(3, 0.01, False)


def test_mixed_precision_policy_is_cuda_only():
    assert MixedPrecisionConfig.from_dict(None) == MixedPrecisionConfig(enabled=True, dtype="float16")
    precision = MixedPrecisionConfig.from_dict({"enabled": True, "dtype": "bfloat16"})
    assert precision.torch_dtype is torch.bfloat16
    assert precision.active_on(torch.device("cuda")) is True
    assert precision.active_on(torch.device("cpu")) is False
    assert MixedPrecisionConfig(enabled=False).active_on(torch.device("cuda")) is False
    with pytest.raises(ValueError, match="dtype"):
        MixedPrecisionConfig(dtype="float32")  # type: ignore[arg-type]


def test_gaussian_nll_promotes_half_precision_inputs_to_float32():
    mean = torch.zeros(2, 5, dtype=torch.float16, requires_grad=True)
    scale = torch.eye(5, dtype=torch.float16).expand(2, -1, -1).clone().requires_grad_()
    target = torch.ones(2, 5, dtype=torch.float16)
    loss = MultivariateGaussianNLL()((mean, scale), target)
    assert loss.dtype is torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert mean.grad is not None
    assert scale.grad is not None


def test_epoch_metrics_are_written_and_printed(capsys):
    class RecordingWriter:
        def __init__(self):
            self.scalars = []
            self.flushes = 0

        def add_scalar(self, name, value, step):
            self.scalars.append((name, value, step))

        def flush(self):
            self.flushes += 1

    model = _model()
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 8, 8), torch.randn(2, 5)), batch_size=1)
    writer = RecordingWriter()
    run_stages(
        model,
        loader,
        loader,
        [TrainingStage("monitored", "deterministic", 1, 1e-3, loss="huber")],
        device=torch.device("cpu"),
        feature_dim=6,
        output_dim=5,
        hidden_dim=10,
        writer=writer,
        progress=True,
    )
    names = {entry[0] for entry in writer.scalars}
    assert names == {
        "loss/monitored/train",
        "loss/monitored/validation",
        "learning_rate/monitored",
    }
    assert writer.flushes == 1
    assert "monitored epoch 1/1" in capsys.readouterr().out
