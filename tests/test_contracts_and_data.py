import h5py
import numpy as np
import torch

from xrd_geometry.contracts import ExperimentSpec, ParameterSpace
from xrd_geometry.data import HDF5DiffractionDataset, parameter_space_from_hdf5
from xrd_geometry.preprocessing import ImagePreprocessor


def test_parameter_space_round_trip_and_fixed_parameter(tiny_hdf5_file):
    space = parameter_space_from_hdf5(tiny_hdf5_file, rot3=0.0)
    full = np.array([0.18, 0.04, 0.01, 0.01, -0.01, 9.9], dtype=np.float32)
    normalized = space.normalize(full)
    np.testing.assert_allclose(space.denormalize(normalized), full[:5], rtol=1e-6)
    expanded = space.expand(space.denormalize(normalized))
    assert expanded["rot3"] == 0.0
    assert tuple(expanded) == ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")


def test_covariance_unit_transform(tiny_hdf5_file):
    space = parameter_space_from_hdf5(tiny_hdf5_file)
    covariance = torch.eye(5).unsqueeze(0)
    physical = space.covariance_to_physical(covariance)
    expected = torch.tensor([0.03, 0.01, 0.01, 0.02, 0.02]).square()
    torch.testing.assert_close(torch.diagonal(physical[0]), expected)


def test_experiment_serialization(tiny_hdf5_file):
    space = parameter_space_from_hdf5(tiny_hdf5_file)
    original = ExperimentSpec("fixture", "detector", (1e-10, 1.01e-10), space, {"calibrant": "alpha"})
    restored = ExperimentSpec.from_dict(original.to_dict())
    assert restored == original


def test_preprocessor_masks_before_percentile_scaling():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    image[0, 0] = 4294967295
    output = ImagePreprocessor(image_size=8)(image)
    assert output.shape == (3, 8, 8)
    assert torch.isfinite(output).all()
    expected_mask_value = torch.tensor((0.0 - 0.485) / 0.229)
    torch.testing.assert_close(output[0, 0, 0], expected_mask_value)


def test_legacy_hdf5_adapter_selects_five_targets_and_is_reproducible(tiny_hdf5_file):
    space = parameter_space_from_hdf5(tiny_hdf5_file)
    processor = ImagePreprocessor(image_size=16)
    first = HDF5DiffractionDataset(tiny_hdf5_file, "train", space, processor, seed=17)
    second = HDF5DiffractionDataset(tiny_hdf5_file, "train", space, processor, seed=17)
    first_image, first_target = first[2]
    second_image, second_target = second[2]
    assert first_image.shape == (3, 16, 16)
    assert first_target.shape == (5,)
    torch.testing.assert_close(first_image, second_image)
    torch.testing.assert_close(first_target, second_target)
    first.close()
    second.close()


def test_validation_group_does_not_randomize_by_default(tiny_hdf5_file):
    space = parameter_space_from_hdf5(tiny_hdf5_file)
    dataset = HDF5DiffractionDataset(
        tiny_hdf5_file,
        "validation",
        space,
        ImagePreprocessor(image_size=8),
    )
    assert dataset.randomize is False
    dataset.close()


def test_randomization_changes_reproducibly_by_epoch(tiny_hdf5_file):
    space = parameter_space_from_hdf5(tiny_hdf5_file)
    dataset = HDF5DiffractionDataset(
        tiny_hdf5_file,
        "train",
        space,
        ImagePreprocessor(image_size=16),
        seed=4,
    )
    epoch_zero = dataset[0][0]
    dataset.set_epoch(1)
    epoch_one = dataset[0][0]
    dataset.set_epoch(0)
    repeated_zero = dataset[0][0]
    assert not torch.equal(epoch_zero, epoch_one)
    torch.testing.assert_close(epoch_zero, repeated_zero)
    dataset.close()
