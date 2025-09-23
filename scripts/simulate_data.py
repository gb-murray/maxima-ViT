# Simulate 2D XRD patterns and write to appropriate H5, separate for training, validation

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import yaml 

from src.data_pipeline import CalibrantSim
from src.utils import get_calibrant, get_detector

def sample_geometry(config: dict) -> dict:
    """Samples a random geometry dictionary from ranges defined in the config."""
    return {
        "dist": np.random.uniform(*config['geometry_ranges']['dist']),
        "poni1": np.random.uniform(*config['geometry_ranges']['poni1']),
        "poni2": np.random.uniform(*config['geometry_ranges']['poni2']),
        "rot1": np.random.uniform(*config['geometry_ranges']['rot1']),
        "rot2": np.random.uniform(*config['geometry_ranges']['rot2']),
        "rot3": np.random.uniform(*config['geometry_ranges']['rot3']),
    }

def generate_sample(config: dict):
    """
    Worker function to generate a single image and its label.
    This function will be called by each parallel process.
    """
    # Each worker gets its own instances to avoid thread-safety issues
    calibrant = get_calibrant(*config['calibrant'], *config['wavelengths']) 
    detector = get_detector(*config['detector'])
    
    # Generate a random geometry for this sample
    geometry_params = sample_geometry(config)
    
    # Instantiate and run the simulation
    sim = CalibrantSim(calibrant, detector, geometry_params)
    snr = np.random.uniform(*config['generation']['snr_range'])
    eta = np.random.uniform(*config['generation']['eta_range'])
    w_sharp = np.random.uniform(*config['generation']['w_range'])
    broad_factor = np.random.uniform(*config['generation']['broad_range'])
    imin = np.random.uniform(*config['generation']['imin_range'])
    imax = np.random.uniform(*config['generation']['imax_range'])
    image = sim.run(
        snr=snr,
        eta=eta,
        w_sharp=w_sharp,
        broad_factor=broad_factor,
        imin=imin,
        imax=imax
    )
    
    # Return the image and its corresponding label
    label = np.array(list(geometry_params.values()), dtype=np.float32)
    return image, label

def generate_dataset(config: dict):
    """
    Main function to orchestrate the parallel generation of the dataset.
    """
    num_samples = config['generation']['num_images']
    hdf5_path = config['paths']['hdf5_file']
    image_shape = get_detector(*config['detector']).shape
    
    # Use a multiprocessing Pool to parallelize generation
    num_workers = cpu_count() - 1
    
    with h5py.File(hdf5_path, "w") as hf:
        # Create datasets for images and labels with chunking for efficiency
        dset_images = hf.create_dataset("images", (num_samples, *image_shape), dtype='f4', chunks=(1, *image_shape)) # type: ignore
        dset_labels = hf.create_dataset("labels", (num_samples, 6), dtype='f4', chunks=(1, 6))

        with Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(generate_sample, [config] * num_samples)
            
            for i, (image, label) in enumerate(tqdm(results, total=num_samples)):
                dset_images[i] = image
                dset_labels[i] = label

    print(f"\nDataset with {num_samples} samples successfully generated at {hdf5_path}")

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    generate_dataset(config)