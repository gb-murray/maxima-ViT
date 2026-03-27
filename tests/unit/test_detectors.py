import pytest

from src.detectors import DetectorProfile, _normalize_profile_name, build_detector_profile


@pytest.mark.unit
def test_normalize_profile_name_replaces_dash_and_lowercases():
    assert _normalize_profile_name(" Eiger2Cdte-1M ") == "eiger2cdte_1m"


@pytest.mark.unit
def test_build_detector_profile_none_returns_default():
    profile = build_detector_profile(None)
    assert isinstance(profile, DetectorProfile)
    assert profile == DetectorProfile()


@pytest.mark.unit
def test_build_detector_profile_dict_overrides_defaults():
    profile = build_detector_profile({"detector": "default", "beamstop_prob": 0.25})
    assert isinstance(profile, DetectorProfile)
    assert profile.beamstop_prob == 0.25


@pytest.mark.unit
def test_build_detector_profile_invalid_type_raises():
    with pytest.raises(TypeError):
        build_detector_profile(123)
