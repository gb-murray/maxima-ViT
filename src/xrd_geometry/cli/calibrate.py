"""Image-only network inference for XRD detector geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..checkpoints import read_checkpoint
from ..config import build_experiment, build_preprocessor, load_config
from ..factory import build_model
from ..prediction import build_estimator


def _write_poni(path: Path, geometry: dict[str, float], detector: str, wavelength: float) -> None:
    lines = [
        "# Calibration produced by xrd_geometry network inference",
        f"Detector: {detector}",
        f"Distance: {geometry['dist']}",
        f"Poni1: {geometry['poni1']}",
        f"Poni2: {geometry['poni2']}",
        f"Rot1: {geometry['rot1']}",
        f"Rot2: {geometry['rot2']}",
        f"Rot3: {geometry['rot3']}",
        f"Wavelength: {wavelength}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_image(
    image: np.ndarray,
    checkpoint_path: str | Path,
    *,
    legacy_config: dict | None = None,
    samples: int | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, float], dict | None]:
    state, metadata = read_checkpoint(checkpoint_path)
    if metadata is None:
        if legacy_config is None:
            raise ValueError("Legacy state dicts require the original YAML configuration")
        experiment = build_experiment(legacy_config)
        preprocessor = build_preprocessor(legacy_config)
        model_config = legacy_config["model"]
        output_dim = int(model_config.get("num_outputs", 6))
        uncertainty_config = legacy_config.get("uncertainty", {})
    else:
        experiment = metadata.experiment
        preprocessor = metadata.preprocessor
        model_config = metadata.model_config
        output_dim = experiment.parameter_space.dimension
        uncertainty_config = metadata.uncertainty
    model = build_model(model_config, output_dim=output_dim)
    model.load_state_dict(state)
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device).eval()
    tensor = preprocessor(image).unsqueeze(0).to(target_device)

    uncertainty = None
    if model_config.get("head", "deterministic") == "full_covariance":
        estimator_config = dict(uncertainty_config)
        if samples is not None:
            estimator_config["samples"] = samples
        prediction = build_estimator(estimator_config)(model, tensor, experiment.parameter_space)
        active = prediction.mean[0].detach().cpu().numpy()
        uncertainty = {
            "parameter_names": list(prediction.parameter_names),
            "units": list(prediction.units),
            "conditional_covariance": prediction.conditional_covariance[0].detach().cpu().tolist(),
            "model_covariance": prediction.model_covariance[0].detach().cpu().tolist(),
            "total_covariance": prediction.total_covariance[0].detach().cpu().tolist(),
        }
    else:
        with torch.no_grad():
            normalized = model(tensor)
        normalized = experiment.parameter_space.select_active(normalized)
        active = experiment.parameter_space.denormalize(normalized)[0].cpu().numpy()
    return experiment.parameter_space.expand(active), uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer XRD detector geometry from one image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", "--model-path", dest="checkpoint", required=True)
    parser.add_argument("--config", help="Required only for a legacy state-dict checkpoint")
    parser.add_argument("--samples", type=int, help="Override the artifact's MC sample count")
    parser.add_argument("--output", help="Optional .poni or .json output")
    arguments = parser.parse_args()

    import fabio

    image = fabio.open(arguments.image).data.astype(np.float32)
    legacy_config = load_config(arguments.config) if arguments.config else None
    geometry, uncertainty = infer_image(
        image,
        arguments.checkpoint,
        legacy_config=legacy_config,
        samples=arguments.samples,
    )
    result = {"geometry": geometry, "uncertainty": uncertainty}
    if arguments.output:
        output = Path(arguments.output)
        if output.suffix.lower() == ".poni":
            state, metadata = read_checkpoint(arguments.checkpoint)
            experiment = metadata.experiment if metadata else build_experiment(legacy_config)
            _write_poni(output, geometry, experiment.detector, experiment.wavelengths_m[0])
        else:
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
