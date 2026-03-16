# Uses a trained Maxima-ViT model to predict the 6D geometry of a diffraction pattern.
# NOT intended for high-throughput inferences, just one-shots

# things from the config:
# - model version
# - image size
# - architecture specs for create_model
# - calibrant and detector info

import os
import argparse
import yaml
import fabio
import numpy as np
import torch

from pyFAI.geometry import Geometry

from maxima_vit.utils import load_model, image_to_tensor, get_calibrant, get_detector, PeakOptimizer

def save_poni_file(output_path: str, params: np.ndarray, config: dict):
    """
    Saves the predicted geometry parameters to a .poni file.
    """
    param_names = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
    
    # Need to add detector and wavelength info
    header = f"# Calibration results from Max-ViT\n"
    header += f"# Model checkpoint: {config['model_path']}\n"
    
    with open(output_path, "w") as f:
        f.write(header)
        for name, value in zip(param_names, params):
            f.write(f"{name.capitalize()}: {value}\n")
            
    print(f"Saved calibration to: {output_path}")

def main():
    """
    Main function to run the calibration.
    """
    parser = argparse.ArgumentParser(description="Calibrate an XRD image using a trained Max-ViT model.")
    parser.add_argument("--image", required=True, help="Path to the input 2D XRD image file.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--config", required=True, help="Path to the YAML training config file used for the model.")
    parser.add_argument("--output", help="Optional path to save the output .poni file.")
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # config['model_path'] = args.model_path # store for header

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the pattern
    image = fabio.open(args.image).data.astype(np.float32)

    image = np.clip(image, 30.0, 300.0) # clip zingers, beamstop
    image_size = config['model'].get('image_size', 1056)
    
    tensor = image_to_tensor(args.image, image_size)
    tensor = tensor.unsqueeze(0).to(device)

    # load model weights
    model = load_model(args.model_path, config)
    model.to(device)
    model.eval() 

    # run inference
    with torch.no_grad():
        prediction = model(tensor).cpu().numpy().flatten()

    calibrant = get_calibrant(config['calibrant'], config['wavelength'])
    detector = get_detector(config['detector'])
    
    geometry = Geometry(
        dist=prediction[0],
        poni1=prediction[1],
        poni2=prediction[2],
        rot1=prediction[3],
        rot2=prediction[4],\
        rot3=prediction[5],
        wavelength=calibrant.wavelength,
        detector=detector
    )

    optimizer = PeakOptimizer(
        image=image,
        initial_geometry=geometry,
        calibrant=calibrant
    )

    refiner = optimizer.optimize()

    refined_dist = refiner.dist
    refined_poni1 = refiner.poni1
    refined_poni2 = refiner.poni2
    refined_rot1 = refiner.rot1
    refined_rot2 = refiner.rot2
    refined_rot3 = refiner.rot3

    refined_geometry = Geometry(
        dist=refined_dist,
        poni1=refined_poni1,
        poni2=refined_poni2,
        rot1=refined_rot1,
        rot2=refined_rot2,
        rot3=refined_rot3,
        wavelength=calibrant.wavelength,
        detector=detector
    )

    # save output file
    if args.output:
        save_poni_file(args.output, refined_geometry.as_array(), config)

    else: 
        print("\nOutput .poni file not specified. Results will not be saved.")
        print("\n--- Calibration Results ---")
        param_names = ["Distance (m)", "PONI1 (m)", "PONI2 (m)", "Rotation 1 (rad)", "Rotation 2 (rad)", "Rotation 3 (rad)"]
        for name, value in zip(param_names, prediction):
            print(f"{name:<10}: {value:.6f}")

if __name__ == '__main__':
    main()