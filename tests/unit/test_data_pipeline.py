import numpy as np
import pytest

from src.data_pipeline import DomainRandomizer


@pytest.mark.unit
@pytest.mark.data
def test_sample_int_handles_inverted_bounds():
    rng = np.random.default_rng(0)
    for _ in range(25):
        value = DomainRandomizer._sample_int(10, 3, rng)
        assert 3 <= value <= 10


@pytest.mark.unit
@pytest.mark.data
def test_domain_randomizer_apply_non_negative_output():
    randomizer = DomainRandomizer(image_shape=(32, 32))
    image = np.ones((32, 32), dtype=np.float32)

    out = randomizer.apply(image=image, imin=1.0, randomize_occlusions=False)

    assert out.shape == image.shape
    assert np.isfinite(out).all()
    assert (out >= 0).all()


@pytest.mark.unit
@pytest.mark.data
def test_domain_randomizer_deterministic_with_seeded_rng():
    image = np.ones((32, 32), dtype=np.float32)

    rand_a = DomainRandomizer(image_shape=(32, 32))
    rand_b = DomainRandomizer(image_shape=(32, 32))

    rand_a.rng = np.random.default_rng(123)
    rand_b.rng = np.random.default_rng(123)

    out_a = rand_a.apply(image=image, imin=1.0, randomize_occlusions=False)
    out_b = rand_b.apply(image=image, imin=1.0, randomize_occlusions=False)

    assert np.allclose(out_a, out_b)
