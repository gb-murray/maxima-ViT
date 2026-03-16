# Data generation, augmentation, and Dataset classes

import copy
import numpy as np
from pyFAI.calibrant import Calibrant
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from typing import cast
import torch
from torch.utils.data import Dataset
import h5py
from torchvision import transforms
from .utils import image_to_tensor

class CalibrantSim:
    """
    Generates a 2D diffraction pattern using a 2-Gaussian approximation of a pseudo-Voigt peak profile and Poisson noise.

    Attributes:
        calibrant (pyFAI.calibrant.Calibrant): The calibrant object. Must have an assigned wavelength attr.
        detector (pyFAI.detector.Detector): The detector object.
        geometry (pyFAI.geometry.Geometry): The ideal detector geometry used in poni format.
    """

    def __init__(self, calibrant: Calibrant, detector: Detector, geometry: dict, wavelength: float):
        self.geometry = geometry
        self.wavelength = wavelength
        self.calibrant = calibrant
        self.primary_calibrant = copy.deepcopy(calibrant)
        self.secondary_calibrant = copy.deepcopy(calibrant)
        self.detector = detector
        self.image = None
        self.rng = np.random.default_rng()

        # Clear any detector-level mask (for example a built-in beamstop) so occlusions
        # can be randomized at the image level during simulation.
        self.detector.mask = np.zeros(detector.shape, dtype=int)

        # peak splitting
        lambda_1 = self.wavelength
        lambda_2 = self.wavelength + 0.00443e-10  # K-alpha 2 offset for In K-alpha source
        self.primary_calibrant.wavelength = lambda_1
        self.secondary_calibrant.wavelength = lambda_2

        self.ai_primary = AzimuthalIntegrator(
            detector=detector, wavelength=lambda_1, **geometry
        )

        self.ai_secondary = AzimuthalIntegrator(
            detector=detector, wavelength=lambda_2, **geometry
        )

    @staticmethod
    def _sample_int(low: int, high: int, rng: np.random.Generator) -> int:
        if low > high:
            low, high = high, low
        return int(rng.integers(low, high + 1))

    def _apply_random_occlusions(
        self,
        image: np.ndarray,
        imin: float,
        beamstop_prob: float,
        deadzone_prob: float,
        deadzone_count_range: tuple[int, int],
        deadzone_width_range: tuple[int, int],
    ) -> np.ndarray:
        h, w = image.shape
        cy, cx = h // 2, w // 2
        occlusion_mask = np.zeros((h, w), dtype=bool)

        if self.rng.random() < beamstop_prob:
            # Horizontal bar near the beam center.
            bar_h = self._sample_int(10, 36, self.rng)
            y_jitter = self._sample_int(-20, 20, self.rng)
            y0 = int(np.clip(cy + y_jitter - bar_h // 2, 0, h - 1))
            y1 = int(np.clip(y0 + bar_h, 0, h))
            occlusion_mask[y0:y1, :] = True

            # Vertical support bar from center downward (matches common beamstop geometry).
            support_w = self._sample_int(3, 10, self.rng)
            x_jitter = self._sample_int(-20, 20, self.rng)
            x0 = int(np.clip(cx + x_jitter - support_w // 2, 0, w - 1))
            x1 = int(np.clip(x0 + support_w, 0, w))
            # Force the support to start at/under the beamstop and extend through the lower half.
            y_start = int(np.clip(max(cy, y1) + self._sample_int(0, 12, self.rng), 0, h))
            occlusion_mask[y_start:, x0:x1] = True

        if self.rng.random() < deadzone_prob:
            n_dead = self._sample_int(deadzone_count_range[0], deadzone_count_range[1], self.rng)
            for _ in range(n_dead):
                gap_w = self._sample_int(deadzone_width_range[0], deadzone_width_range[1], self.rng)
                gap_w = min(gap_w, w)
                if self.rng.random() < 0.5:
                    # Left edge deadzone
                    occlusion_mask[:, :gap_w] = True
                else:
                    # Right edge deadzone
                    occlusion_mask[:, w - gap_w:] = True

        if np.any(occlusion_mask):
            # Keep occluded regions dark but not perfectly constant.
            blocked_signal = self.rng.normal(
                loc=max(0.0, imin),
                scale=max(1.0, 0.2 * max(imin, 1.0)),
                size=image.shape,
            )
            image[occlusion_mask] = np.clip(blocked_signal[occlusion_mask], 0.0, None)

        return image

    def __repr__(self):
        status = "generated" if self.image is not None else "not run"
        return (f"FakeCalibrantImage(calibrant='{self.calibrant.name}', "
                f"detector='{self.detector.name}', status='{status}')")

    def get_geometry(self) -> dict:
        return self.geometry.copy()

    def run(self,
            imax: float = 1000.0,
            imin: float = 5.0,
            fwhm: float = 0.15,
            k_alpha_ratio: float = 0.5,
            poisson_exposure_scale: float = 0.6,
            randomize_occlusions: bool = True,
            beamstop_prob: float = 1.0,
            deadzone_prob: float = 0.8,
            deadzone_count_range: tuple[int, int] = (1, 3),
            deadzone_width_range: tuple[int, int] = (4, 16)) -> np.ndarray: #TODO implement generic separation logic for other sources
            """
            Executes the simulation with the given parameters.

            Args:
                imax (float): Peak intensity (counts)
                imin (float): Background level (counts)
                w_sharp (float): Peak width (Caglioti W)
                k_alpha_ratio (float): Intensity of K-alpha 2 relative to K-alpha 1
                poisson_exposure_scale (float): Lower values (<1) increase shot noise while keeping mean intensity similar.
                k_alpha_separation (float): Wavelength difference in meters
                randomize_occlusions (bool): Add randomized beamstop + deadzone artifacts.
                beamstop_prob (float): Probability of adding a beamstop to an image.
                deadzone_prob (float): Probability of adding detector deadzones.
                deadzone_count_range (tuple[int, int]): Number of deadzones to sample.
                deadzone_width_range (tuple[int, int]): Deadzone width range in pixels.
            """
                        
            # K-alpha 1
            img_k1 = self.primary_calibrant.fake_calibration_image(
                self.ai_primary,
                Imax=imax,
                Imin=0,
                resolution=fwhm
            )
            
            # K-alpha 2
            img_k2 = self.secondary_calibrant.fake_calibration_image(
                self.ai_secondary,
                Imax=imax * k_alpha_ratio,
                Imin=0,
                resolution=fwhm
            )

            # combine signals
            clean_signal = img_k1 + img_k2 + imin
            
            # add poisson shot noise
            clean_signal[clean_signal < 0] = 0
            poisson_exposure_scale = max(float(poisson_exposure_scale), 1e-3)
            poisson_lambda = clean_signal * poisson_exposure_scale
            self.image = (
                np.random.poisson(poisson_lambda).astype(np.float32) / poisson_exposure_scale
            )

            # add gaussian readout noise
            readout_noise = np.random.normal(0, 10.0, self.image.shape)
            self.image = self.image + readout_noise
        
            # clip negative values from Gauss
            self.image[self.image < 0] = 0

            if randomize_occlusions:
                self.image = self._apply_random_occlusions(
                    self.image,
                    imin=imin,
                    beamstop_prob=beamstop_prob,
                    deadzone_prob=deadzone_prob,
                    deadzone_count_range=deadzone_count_range,
                    deadzone_width_range=deadzone_width_range,
                )
        
            return self.image
    
class HDF5Dataset(Dataset):
    """
    Handles large training datasets in an HDF5 format.
    """
    def __init__(self, hdf5_path: str, group: str, image_size: int = 224):
        self.hdf5_path = hdf5_path  
        self.group = group
        self.image_size = image_size
        self.file = None 

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            return len(f[self.group]['images']) #type: ignore

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')

        images = self.file[self.group]['images'] #type: ignore
        labels = self.file[self.group]['labels'] #type: ignore
        
        image = images[idx] #type: ignore

        final_tensor = image_to_tensor(image, self.image_size) #type:ignore
        label_tensor = torch.from_numpy(labels[idx]).float() #type: ignore
        
        return final_tensor, label_tensor