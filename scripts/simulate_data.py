import os
import sys
import yaml
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.data_pipeline import CalibrantSim
from src.utils import get_calibrant, get_detector

# Data Generation
def sample_geometry(config: dict) -> dict:
    """
    Samples a random geometry dictionary from ranges defined in the config.
    """
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
    """
    calibrant = get_calibrant(config['calibrant'], config['wavelength'])
    detector = get_detector(config['detector'])
    geometry_params = sample_geometry(config)
    
    sim = CalibrantSim(calibrant, detector, geometry_params)
    sim_params = {k: np.random.uniform(*v) for k, v in config['simulation_ranges'].items()}
    image = sim.run(**sim_params)
    
    label = np.array(list(geometry_params.values()), dtype=np.float32)
    return image, label

def _populate_group(h5_group, num_samples: int, config: dict, desc: str):
    """
    Helper to run the simulation and fill an HDF5 group.
    """
    if num_samples <= 0:
        return
        
    print(f"Generating {num_samples} samples for the '{desc}' set...")
    image_shape = get_detector(config['detector']).shape
    dset_images = h5_group.create_dataset("images", (num_samples, *image_shape), dtype='f4', chunks=(1, *image_shape)) #type: ignore
    dset_labels = h5_group.create_dataset("labels", (num_samples, 6), dtype='f4', chunks=(1, 6))

    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.imap_unordered(generate_sample, [config] * num_samples)
        for i, (image, label) in enumerate(tqdm(results, total=num_samples, desc=desc)):
            dset_images[i] = image
            dset_labels[i] = label

def generate_dataset(config: dict):
    """
    Main function to generate a dataset and save it to a HDF5 file.
    """
    output_path = config['paths']['output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_total = config['generation']['num_images']
    test_ratio = config['generation']['test_split_ratio']
    n_test = int(n_total * test_ratio)
    n_train_pool = n_total - n_test

    print(f"Generating dataset at: {output_path}")
    print(f"Total samples: {n_total} (Training Pool: {n_train_pool}, Test: {n_test})")

    with h5py.File(output_path, "w") as hf:
        train_pool_group = hf.create_group('training_pool')
        test_group = hf.create_group('test')
        
        _populate_group(train_pool_group, n_train_pool, config, "Training Pool")
        _populate_group(test_group, n_test, config, "Test Set")

    print(f"\nDataset generation complete. File saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an XRD simulation dataset for k-fold cross-validation.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    generate_dataset(config)