import pytest
import torch

from src.data_pipeline import DiffractionDataset


@pytest.mark.integration
@pytest.mark.data
def test_diffraction_dataset_len_matches_hdf5_samples(tiny_hdf5_file):
    dataset = DiffractionDataset(str(tiny_hdf5_file), group="train", image_size=64)
    assert len(dataset) == 4


@pytest.mark.integration
@pytest.mark.data
def test_diffraction_dataset_getitem_returns_expected_tensors(tiny_hdf5_file):
    dataset = DiffractionDataset(str(tiny_hdf5_file), group="train", image_size=64)

    image_tensor, label_tensor = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 64, 64)
    assert label_tensor.shape == (6,)
    assert torch.isfinite(image_tensor).all()
    assert torch.isfinite(label_tensor).all()


@pytest.mark.integration
@pytest.mark.data
def test_diffraction_dataset_can_disable_dynamic_randomization(tiny_hdf5_file):
    dataset = DiffractionDataset(
        str(tiny_hdf5_file),
        group="train",
        image_size=64,
        apply_dynamic_randomization=False,
    )

    image_tensor, _ = dataset[0]
    assert image_tensor.shape == (3, 64, 64)
