from pathlib import Path

import pytest
import torch
import yaml

from scripts.interpolate_weights import interpolate_pos_embeddings


@pytest.mark.integration
def test_interpolate_pos_embeddings_resizes_grid(tmp_path: Path):
    config = {
        "model": {
            "image_size": 448,
            "patch_size": 16,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    state = {
        "vit.embeddings.position_embeddings": torch.randn(1, 197, 32),
    }
    model_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_out.pth"
    torch.save(state, model_path)

    interpolate_pos_embeddings(str(config_path), str(model_path), str(output_path))

    assert output_path.exists()

    saved_state = torch.load(output_path, map_location=torch.device("cpu"))
    pos = saved_state["vit.embeddings.position_embeddings"]
    assert pos.shape == (1, 785, 32)
