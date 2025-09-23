# Helper functions

from pyFAI.calibrant import Calibrant, CALIBRANT_FACTORY
from pyFAI.detectors import Detector, detector_factory

def get_calibrant(alias: str, wavelength: float) -> Calibrant:

    calibrant = CALIBRANT_FACTORY(alias)
    calibrant.wavelength = wavelength

    return calibrant

def get_detector(alias:str) -> Detector:

    detector = detector_factory(alias)

    return detector