# Data generation, augmentation, and Dataset classes

import numpy as np
from pyFAI.calibrant import Calibrant
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from typing import cast
import torch
from torch.utils.data import Dataset
import h5py
from torchvision import transforms

class CalibrantSim:
    """
    Generates a 2D diffraction pattern using a 2-Gaussian approximation of a pseudo-Voigt peak profile and Poisson noise.

    Attributes:
        calibrant (pyFAI.calibrant.Calibrant): The calibrant object. Must have an assigned wavelength attr.
        detector (pyFAI.detector.Detector): The detector object.
        geometry (pyFAI.geometry.Geometry): The ideal detector geometry used in poni format.
    """

    def __init__(self, calibrant: Calibrant, detector: Detector, geometry: dict):
        self.geometry = geometry
        self.calibrant = calibrant
        self.detector = detector
        self.image = None
        self.azimuthal_integrator = AzimuthalIntegrator(
            detector=detector,
            wavelength=calibrant.wavelength,
            **geometry
        )

    def __repr__(self):
        status = "generated" if self.image is not None else "not run"
        return (f"FakeCalibrantImage(calibrant='{self.calibrant.name}', "
                f"detector='{self.detector.name}', status='{status}')")

    def get_geometry(self) -> dict:
        return self.geometry.copy()

    def run(self,
            snr: float = float('inf'),
            eta: float = 0.5,
            w_sharp: float = 1e-6,
            broad_factor: float = 10.0,
            imin: float = 0.0,
            imax: float = 1.0):
        """
        Executes the simulation.

        Args:
            snr (float): Desired signal-to-noise-ratio.
            eta (float): Mixing parameter for pseudo-Voigt approximation.
            w_sharp (float): W Caglioti parameter for the sharp component.
            broad_factor (float): Multiplier for w_sharp to create the broad component.
            imin (float): The absolute background intensity for the image.
            imax (float): The absolute maximum peak intensity for the image.
        """
        if self.azimuthal_integrator is None:
            from pyFAI.integrator.azimuthal import AzimuthalIntegrator
            self.azimuthal_integrator = AzimuthalIntegrator(
                detector=self.detector,
                wavelength=self.calibrant.wavelength,
                **self.geometry
            )

        img_sharp = self.calibrant.fake_calibration_image(
            self.azimuthal_integrator,
            Imax=imax * (1.0 - eta),
            Imin=imin / 2.0,  
            W=w_sharp
        )

        w_broad = w_sharp * broad_factor
        img_broad = self.calibrant.fake_calibration_image(
            self.azimuthal_integrator,
            Imax=imax * eta,
            Imin=imin / 2.0, 
            W=w_broad
        )

        combined_img = img_sharp + img_broad

        noisy_img = self._add_noise(combined_img, snr)
        self.frame = noisy_img
        return cast(np.ndarray, self.frame)

    def _add_noise(self, image: np.ndarray, snr: float) -> np.ndarray:
        image[image < 0] = 0

        if snr == float('inf'):
            return image.astype(np.float32)

        noisy_image = np.random.poisson(image * snr) / snr
        return noisy_image.astype(np.float32)
    
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
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
        _ , h, w = image_tensor.shape
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        
        padding_transform = transforms.Pad(padding=(pad_w, pad_h))
        padded_tensor = padding_transform(image_tensor)

        resize_transform = transforms.Resize((self.image_size, self.image_size), antialias=True)
        final_tensor = resize_transform(padded_tensor)

        final_tensor = final_tensor.repeat(3, 1, 1)
        label_tensor = torch.from_numpy(labels[idx]).float() #type: ignore
        
        return final_tensor, label_tensor