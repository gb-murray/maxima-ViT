# Uses a trained Maxima-ViT model to predict the 6D geometry of a diffraction pattern.
# NOT intended for high-throughput inferences, just one-shots

import os
import sys
import argparse
import yaml
import fabio
import numpy as np
import torch
from torchvision.transforms import Resize

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.utils import create_model, image_to_tensor

def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    """
    Loads and preprocesses a single 2D diffraction pattern.
    """
    print(f"Loading image: {image_path}")
    
    image = fabio.open(image_path).data.astype(np.float32)
    tensor = image_to_tensor(image, image_size)

    return tensor.to(device)

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

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config['model_path'] = args.model_path # store for header

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    model = create_model(config)
    state_dict = torch.load(args.model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[10:] if k.startswith('_orig_mod.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval() 

    # Preprocess the image
    image_size = config['model'].get('image_size', 224)
    input_tensor = preprocess_image(args.image, image_size, device)

    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        predicted_params_tensor = model(input_tensor)

    # Post-process and display results
    predicted_params = predicted_params_tensor.cpu().numpy().flatten()
    
    print("\n--- Calibration Results ---")
    param_names = ["Distance (m)", "PONI1 (m)", "PONI2 (m)", "Rotation 1 (rad)", "Rotation 2 (rad)", "Rotation 3 (rad)"]
    for name, value in zip(param_names, predicted_params):
        print(f"{name:<20}: {value:.6f}")

    # Save output file
    if args.output:
        save_poni_file(args.output, predicted_params, config)

if __name__ == '__main__':
    main()