# Data generation, augmentation, and Dataset classes

import copy
import numpy as np
from pyFAI.calibrant import Calibrant
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import torch
from torch.utils.data import Dataset
import h5py
from scipy.ndimage import gaussian_filter
from .utils import image_to_tensor
from .detectors import DetectorProfile


class DomainRandomizer:
    """Applies on-the-fly detector/domain randomization to a diffraction image."""

    def __init__(self, image_shape: tuple[int, int], profile: DetectorProfile | None = None):
        self.image_shape = image_shape
        self.profile = profile if profile is not None else DetectorProfile()
        self.rng = np.random.default_rng()
        h, w = image_shape
        self.cy = h / 2.0
        self.cx = w / 2.0

        y, x = np.ogrid[:h, :w]
        self._dist_sq = (x - self.cx) ** 2 + (y - self.cy) ** 2

    @staticmethod
    def _sample_int(low: int, high: int, rng: np.random.Generator) -> int:
        if low > high:
            low, high = high, low
        return int(rng.integers(low, high + 1))

    def _apply_photometric_degradation(
        self,
        image: np.ndarray,
        imin: float,
        profile: DetectorProfile,
    ) -> np.ndarray:
        """
        Injects real-world photometric degradation: air scatter, azimuthal texture,
        photon shot noise, and detector readout noise.
        """
        work = image.astype(np.float32, copy=True)
        h, w = work.shape

        raw_noise = self.rng.random((h, w), dtype=np.float32)
        # A large sigma creates massive, smooth, organic clouds
        smooth_noise = gaussian_filter(raw_noise, sigma=w // 30) 
        # Normalize strictly between 0 and 1
        smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())

        texture_floor = self.rng.uniform(*profile.texture_floor_range)
        texture_floor = float(np.clip(texture_floor, 0.0, 1.0))
        texture_map = texture_floor + (1.0 - texture_floor) * smooth_noise
        work *= texture_map

        scatter_intensity = self.rng.uniform(*profile.scatter_intensity_range)
        decay_rate = self.rng.uniform(*profile.scatter_decay_range)
        background = scatter_intensity * np.exp(-decay_rate * self._dist_sq)
        work += background.astype(np.float32)

        global_baseline = self.rng.uniform(2.0, 5.0)
        work += global_baseline

        flux = max(self.rng.uniform(*profile.flux_range), 1e-3)
        work = np.clip(work * flux, 0.0, None)
        work = self.rng.poisson(work).astype(np.float32)

        readout_std = max(self.rng.uniform(*profile.readout_noise_std_range), 0.0)
        if readout_std > 0:
            readout_noise = self.rng.normal(0.0, readout_std, size=work.shape)
            work += readout_noise.astype(np.float32)

        baseline_shift = self.rng.uniform(0.8, 1.2) * max(imin, 0.0)
        work += np.float32(baseline_shift)
        work[work < 0] = 0
        return work

    def _apply_random_occlusions(
        self,
        image: np.ndarray,
        imin: float,
        profile: DetectorProfile,
        randomize_occlusions: bool,
    ) -> np.ndarray:
        h, w = image.shape
        cy, cx = h // 2, w // 2

        beamstop_prob = profile.beamstop_prob if randomize_occlusions else 0.0
        deadzone_prob = profile.deadzone_prob if randomize_occlusions else 0.0

        # --- 1. The Beamstop (Bright & Speckled) ---
        if self.rng.random() < beamstop_prob:
            beamstop_mask = np.zeros((h, w), dtype=bool)
            
            # Horizontal bar
            bar_h = self._sample_int(35, 55, self.rng)
            y_jitter = self._sample_int(-20, 20, self.rng)
            y0 = int(np.clip(cy + y_jitter - bar_h // 2, 0, h - 1))
            y1 = int(np.clip(y0 + bar_h, 0, h))
            beamstop_mask[y0:y1, :] = True

            # # Vertical support
            # support_w = self._sample_int(3, 10, self.rng)
            # x_jitter = self._sample_int(-20, 20, self.rng)
            # x0 = int(np.clip(cx + x_jitter - support_w // 2, 0, w - 1))
            # x1 = int(np.clip(x0 + support_w, 0, w))
            # y_start = int(np.clip(max(cy, y1) + self._sample_int(0, 12, self.rng), 0, h))
            # beamstop_mask[y_start:, x0:x1] = True

            # Fill beamstop with independent, bright Poisson scatter
            max_dist_sq = cx**2 + cy**2
            normalized_dist = self._dist_sq / max_dist_sq
            
            # The beamstop glows warmest near the direct beam and fades to the edges
            bs_center_glow = self.rng.uniform(400.0, 500.0)
            bs_edge_glow = self.rng.uniform(50.0, 150.0)
            
            # Create a smooth linear fade outward
            bs_profile = bs_center_glow * (1.0 - normalized_dist) + bs_edge_glow
            bs_noise = self.rng.poisson(bs_profile).astype(np.float32)
            
            image[beamstop_mask] = bs_noise[beamstop_mask]

        # --- 2. The Deadzones (Pitch Black) ---
        # if self.rng.random() < deadzone_prob:
        #     deadzone_mask = np.zeros((h, w), dtype=bool)
        #     n_dead = self._sample_int(profile.deadzone_count_range[0], profile.deadzone_count_range[1], self.rng)
            
        #     for _ in range(n_dead):
        #         gap_w = self._sample_int(profile.deadzone_width_range[0], profile.deadzone_width_range[1], self.rng)
        #         gap_w = min(gap_w, w)
        #         if self.rng.random() < 0.5:
        #             deadzone_mask[:, :gap_w] = True
        #         else:
        #             deadzone_mask[:, w - gap_w:] = True

        #     # Fill deadzones with absolute minimal noise (near zero)
        #     dz_noise = self.rng.normal(loc=max(0.0, imin), scale=max(0.5, 0.2 * imin), size=(h, w))
        #     image[deadzone_mask] = np.clip(dz_noise[deadzone_mask], 0.0, None)

        return image

    def apply(
        self,
        image: np.ndarray,
        imin: float | None = None,
        randomize_occlusions: bool = True,
        profile_override: DetectorProfile | None = None,
    ) -> np.ndarray:
        """Applies photometric degradation first, then detector occlusions."""
        profile = profile_override if profile_override is not None else self.profile
        if imin is None:
            imin = float(self.rng.uniform(*profile.imin_range))

        degraded = self._apply_photometric_degradation(
            image=image,
            imin=imin,
            profile=profile,
        )
        degraded = self._apply_random_occlusions(
            image=degraded,
            imin=imin,
            profile=profile,
            randomize_occlusions=randomize_occlusions,
        )
        degraded[degraded < 0] = 0
        return degraded

class CalibrantSimulator:
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
        self.domain_randomizer = DomainRandomizer(detector.shape)

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

    def _generate_base_pattern(
        self,
        imax: float,
        fwhm: float,
        k_alpha_ratio: float,
    ) -> np.ndarray:
        """Runs the expensive pyFAI synthesis once (K-alpha split included)."""
        img_k1 = self.primary_calibrant.fake_calibration_image(
            self.ai_primary,
            Imax=imax,
            Imin=0,
            resolution=fwhm,
        )

        img_k2 = self.secondary_calibrant.fake_calibration_image(
            self.ai_secondary,
            Imax=imax * k_alpha_ratio,
            Imin=0,
            resolution=fwhm,
        )

        base_pattern = (img_k1 + img_k2).astype(np.float32)
        base_pattern[base_pattern < 0] = 0
        return base_pattern

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
            apply_domain_randomization: bool = True,
            randomize_occlusions: bool = True,
            beamstop_prob: float = 1.0,
            deadzone_prob: float = 0.8,
            deadzone_count_range: tuple[int, int] = (1, 3),
            deadzone_width_range: tuple[int, int] = (4, 16),
            flux_range: tuple[float, float] = (0.5, 5.0),
            scatter_intensity_range: tuple[float, float] = (10.0, 100.0),
            scatter_decay_range: tuple[float, float] = (1e-5, 5e-5),
            texture_floor_range: tuple[float, float] = (0.25, 0.55),
            readout_noise_std_range: tuple[float, float] = (6.0, 14.0)) -> np.ndarray: #TODO implement generic separation logic for other sources
            """
            Executes the simulation with the given parameters.

            Args:
                imax (float): Peak intensity (counts)
                imin (float): Background level (counts)
                w_sharp (float): Peak width (Caglioti W)
                k_alpha_ratio (float): Intensity of K-alpha 2 relative to K-alpha 1
                k_alpha_separation (float): Wavelength difference in meters
                apply_domain_randomization (bool): Adds dynamic photometric and occlusion artifacts.
                randomize_occlusions (bool): Add randomized beamstop + deadzone artifacts.
                beamstop_prob (float): Probability of adding a beamstop to an image.
                deadzone_prob (float): Probability of adding detector deadzones.
                deadzone_count_range (tuple[int, int]): Number of deadzones to sample.
                deadzone_width_range (tuple[int, int]): Deadzone width range in pixels.
            """

            self.image = self._generate_base_pattern(
                imax=imax,
                fwhm=fwhm,
                k_alpha_ratio=k_alpha_ratio,
            )

            if apply_domain_randomization:
                profile_override = DetectorProfile(
                    imin_range=(float(imin), float(imin)),
                    beamstop_prob=float(beamstop_prob),
                    deadzone_prob=float(deadzone_prob),
                    deadzone_count_range=deadzone_count_range,
                    deadzone_width_range=deadzone_width_range,
                    flux_range=flux_range,
                    scatter_intensity_range=scatter_intensity_range,
                    scatter_decay_range=scatter_decay_range,
                    texture_floor_range=texture_floor_range,
                    readout_noise_std_range=readout_noise_std_range,
                )
                self.image = self.domain_randomizer.apply(
                    image=self.image,
                    imin=imin,
                    randomize_occlusions=randomize_occlusions,
                    profile_override=profile_override,
                )
        
            return self.image
    
class DiffractionDataset(Dataset):
    """
    Handles large training datasets in an HDF5 format and applies dynamic domain randomization.
    """
    def __init__(
        self,
        hdf5_path: str,
        group: str,
        image_size: int = 224,
        apply_dynamic_randomization: bool = True,
        detector_profile: DetectorProfile | None = None,
    ):
        self.hdf5_path = hdf5_path  
        self.group = group
        self.image_size = image_size
        self.file = None
        self.apply_dynamic_randomization = apply_dynamic_randomization
        self.detector_profile = detector_profile if detector_profile is not None else DetectorProfile()
        self.randomizer = None

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            return len(f[self.group]['images']) #type: ignore

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')

        images = self.file[self.group]['images'] #type: ignore
        labels = self.file[self.group]['labels'] #type: ignore

        image = np.array(images[idx], dtype=np.float32) #type: ignore

        if self.apply_dynamic_randomization:
            if self.randomizer is None:
                self.randomizer = DomainRandomizer(image.shape, profile=self.detector_profile)
            image = self.randomizer.apply(
                image=image,
            )

        final_tensor = image_to_tensor(image, self.image_size) #type:ignore
        label_tensor = torch.from_numpy(labels[idx]).float() #type: ignore
        
        return final_tensor, label_tensor