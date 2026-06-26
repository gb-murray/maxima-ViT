from dataclasses import dataclass, fields, replace
from typing import Any, Tuple


@dataclass(frozen=True)
class DetectorProfile:
    """Base class defining parameters for dynamic domain randomization."""

    imin_range: Tuple[float, float] = (0.0, 0.5)
    beamstop_prob: float = 1.0
    deadzone_prob: float = 0.0
    deadzone_count_range: Tuple[int, int] = (1, 3)
    deadzone_width_range: Tuple[int, int] = (2, 8)
    flux_range: Tuple[float, float] = (1.0, 5.0)
    scatter_intensity_range: Tuple[float, float] = (0.0, 10.0)
    scatter_decay_range: Tuple[float, float] = (3e-5, 5e-5)
    texture_floor_range: Tuple[float, float] = (0.55, 0.75)
    readout_noise_std_range: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Eiger2Cdte1MProfile(DetectorProfile):
    """Default profile for Eiger2Cdte_1M; values can be overridden from YAML."""


_PROFILE_REGISTRY: dict[str, DetectorProfile] = {
    "default": DetectorProfile(),
    "Eiger2Cdte_1M": Eiger2Cdte1MProfile()
}


def _normalize_profile_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def build_detector_profile(detector_config: Any) -> DetectorProfile:
    """
    Build a detector profile from YAML config.

    Supported shapes:
    - None -> default profile
    - str -> profile name (e.g. "default", "Eiger2Cdte_1M")
    - dict -> optional "detector" + any DetectorProfile fields as overrides
    """
    if detector_config is None:
        return DetectorProfile()

    if isinstance(detector_config, str):
        key = _normalize_profile_name(detector_config)
        return _PROFILE_REGISTRY.get(key, DetectorProfile())

    if isinstance(detector_config, dict):
        profile_name = str(detector_config.get("detector", "default"))
        base = _PROFILE_REGISTRY.get(_normalize_profile_name(profile_name), DetectorProfile())
        profile_field_names = {f.name for f in fields(DetectorProfile)}
        overrides = {k: v for k, v in detector_config.items() if k in profile_field_names}
        return replace(base, **overrides)

    raise TypeError("'detector' must be null, string, or mapping in YAML config.")
