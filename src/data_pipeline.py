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
from src.utils import image_to_tensor

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
        self.detector = detector
        self.image = None

        self.calibrant.set_wavelength(wavelength)
        self.ai = AzimuthalIntegrator(
            detector=detector,
            wavelength=wavelength,
            **geometry
        )

    def __repr__(self):
        status = "generated" if self.image is not None else "not run"
        return (f"FakeCalibrantImage(calibrant='{self.calibrant.name}', "
                f"detector='{self.detector.name}', status='{status}')")

    def get_geometry(self) -> dict:
        return self.geometry.copy()

    def run(self,
            imax: float = 1000.0,
            imin: float = 5.0,
            w_sharp: float = 1e-4,
            k_alpha_ratio: float = 0.5,
            k_alpha_separation: float = 0.00443e-10) -> np.ndarray: #TODO implement generic separation logic for other sources
            """
            Executes the simulation with the given parameters.

            Args:
                imax (float): Peak intensity (counts)
                imin (float): Background level (counts)
                w_sharp (float): Peak width (Caglioti W)
                k_alpha_ratio (float): Intensity of K-alpha 2 relative to K-alpha 1
                k_alpha_separation (float): Wavelength difference in meters
            """
            
            # peak splitting
            lambda_1 = self.wavelength
            lambda_2 = self.wavelength + k_alpha_separation
            
            # K-alpha 1
            self.ai.wavelength = lambda_1
            img_k1 = self.calibrant.fake_calibration_image(
                self.ai, 
                Imax=imax, 
                Imin=0,     
                W=w_sharp
            )
            
            # K-alpha 2
            self.ai.wavelength = lambda_2
            img_k2 = self.calibrant.fake_calibration_image(
                self.ai, 
                Imax=imax * k_alpha_ratio, 
                Imin=0, 
                W=w_sharp
            )
            
            self.ai.wavelength = lambda_1 # reset to original wavelength
            
            # combine signals
            clean_signal = img_k1 + img_k2 + imin
            
            # add poisson noise
            clean_signal[clean_signal < 0] = 0
            self.image = np.random.poisson(clean_signal).astype(np.float32)
            
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