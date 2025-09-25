import os
import sys
import yaml
import h5py
import uuid
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from girder_client import GirderClient

# Project Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.data_pipeline import CalibrantSim
from src.utils import get_calibrant, get_detector

class GirderUploader:
    """
    Handles connection and file uploads to a Girder instance.
    """
    def __init__(self, api_url: str, api_key: str):
        self.client = GirderClient(apiUrl=api_url)
        self.client.authenticate(apiKey=api_key)
        print(f"Successfully authenticated to Girder as '{self.client.get('user/me')['login']}'")

    def upload(self, local_path: str, parent_folder_id: str, filename: str):
        """Uploads a single file to a specific folder on Girder."""
        print(f"\nUploading {filename} to Girder folder ID {parent_folder_id}...")
        
        self.client.uploadFileToFolder(
            folderId=parent_folder_id,
            filepath=local_path,
            filename=filename
        )
        
        print("Upload complete.")

# Data Generation
def sample_geometry(config: dict) -> dict:
    """Samples a random geometry dictionary from ranges defined in the config."""
    return {
        "dist": np.random.normal(*config['geometry_ranges']['dist']),
        "poni1": np.random.normal(*config['geometry_ranges']['poni1']),
        "poni2": np.random.normal(*config['geometry_ranges']['poni2']),
        "rot1": np.random.uniform(*config['geometry_ranges']['rot1']),
        "rot2": np.random.uniform(*config['geometry_ranges']['rot2']),
        "rot3": np.random.uniform(*config['geometry_ranges']['rot3']),
    }

def generate_sample(config: dict):
    """Worker function to generate a single image and its label."""
    calibrant = get_calibrant(config['calibrant'], config['wavelength'])
    detector = get_detector(config['detector'])
    geometry_params = sample_geometry(config)
    
    sim = CalibrantSim(calibrant, detector, geometry_params)
    sim_params = {k: np.random.uniform(*v) for k, v in config['simulation_ranges'].items()}
    image = sim.run(**sim_params)
    
    label = np.array(list(geometry_params.values()), dtype=np.float32)
    return image, label

# Orchestration Logic

def _populate_group(h5_group, num_samples: int, config: dict, desc: str):
    """Helper to run the simulation and fill an HDF5 group."""
    if num_samples == 0:
        return
        
    print(f"Generating {num_samples} samples for the '{desc}' set...")
    image_shape = get_detector(config['detector']).shape
    dset_images = h5_group.create_dataset("images", (num_samples, *image_shape), dtype='f4', chunks=(1, *image_shape)) # type: ignore
    dset_labels = h5_group.create_dataset("labels", (num_samples, 6), dtype='f4', chunks=(1, 6))

    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.imap_unordered(generate_sample, [config] * num_samples)
        for i, (image, label) in enumerate(tqdm(results, total=num_samples, desc=desc)):
            dset_images[i] = image
            dset_labels[i] = label

def generate_and_upload(config: dict):
    """
    Main function to generate the dataset with splits, save to a temporary
    HDF5 file, and upload it to Girder.
    """
    n_total = config['generation']['num_images']
    split_ratio = config['generation']['split_ratio']
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val

    tmp_dir = config['paths']['tmp_dir']
    os.makedirs(tmp_dir, exist_ok=True)
    
    tmp_filename = f"tmp_dataset_{uuid.uuid4().hex}.hdf5"
    tmp_path = os.path.join(tmp_dir, tmp_filename)
    
    print(f"Generating dataset in temporary file: {tmp_path}")

    try:
        with h5py.File(tmp_path, "w") as hf:
            train_group = hf.create_group('train')
            val_group = hf.create_group('validation')
            test_group = hf.create_group('test')
            
            _populate_group(train_group, n_train, config, "Training")
            _populate_group(val_group, n_val, config, "Validation")
            _populate_group(test_group, n_test, config, "Testing")

        print("\nLocal HDF5 file generation complete.")
        
    #     girder_cfg = config['girder']
    #     api_key = os.environ.get("HTMDEC_API_KEY")
    #     if not api_key:
    #         raise ValueError("HTMDEC_API_KEY environment variable not set.")
            
    #     uploader = GirderUploader(api_url=girder_cfg['api_url'], api_key=api_key)
    #     uploader.upload(
    #         local_path=tmp_path,
    #         parent_folder_id=girder_cfg['parent_folder_id'],
    #         filename=girder_cfg['filename']
    #     )
    # finally:
    #     if os.path.exists(tmp_path):
    #         os.remove(tmp_path)
    #         print(f"Removed {tmp_path}.")

    finally:
        pass

    print("Dataset generation and upload complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and upload an XRD simulation dataset.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    generate_and_upload(config)