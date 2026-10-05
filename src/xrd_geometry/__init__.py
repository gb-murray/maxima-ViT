"""Image-based probabilistic detector geometry inference for XRD."""

from .contracts import ExperimentSpec, ParameterSpace, ParameterSpec
from .prediction import GeometryPrediction

__all__ = [
    "ExperimentSpec",
    "GeometryPrediction",
    "ParameterSpace",
    "ParameterSpec",
]

__version__ = "0.2.0"
