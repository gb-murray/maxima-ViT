import torch
import torch.nn.functional as F
import argparse
import yaml

def interpolate_pos_embeddings(
    config_path: str,
    model_path: str,
    output_path: str
):
    """
    Loads a trained ViT state_dict, interpolates its positional embeddings to a new
    resolution specified in a config file, and saves the new state_dict.
    """
    # Load the configuration to get the target image size
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    image_size = config['model']['image_size']
    patch_size = config['model'].get('patch_size', 16) 
    target_image_size = (image_size, image_size) # Ensure a square target

    print(f"Loading original low-resolution model from: {model_path}")
    low_res_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    pos_embeddings_tensor = low_res_state_dict['vit.embeddings.position_embeddings']
    cls_token = pos_embeddings_tensor[:, 0, :]
    patch_embeddings = pos_embeddings_tensor[:, 1:, :]
    
    original_grid_size = int((patch_embeddings.shape[1])**0.5)
    new_grid_size = (target_image_size[0] // patch_size, target_image_size[1] // patch_size)
    
    print(f"Original patch grid size: {original_grid_size}x{original_grid_size}")
    print(f"New target patch grid size from config: {new_grid_size[0]}x{new_grid_size[1]}")

    patch_embeddings_2d = patch_embeddings.view(1, original_grid_size, original_grid_size, -1).permute(0, 3, 1, 2)
    
    interpolated_embeddings_2d = F.interpolate(
        patch_embeddings_2d, size=new_grid_size, mode='bicubic', align_corners=False
    )
    
    interpolated_embeddings = interpolated_embeddings_2d.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embeddings = torch.cat((cls_token.unsqueeze(1), interpolated_embeddings), dim=1)
    
    low_res_state_dict['vit.embeddings.position_embeddings'] = new_pos_embeddings

    print(f"Saving new high-resolution state dictionary to: {output_path}")
    torch.save(low_res_state_dict, output_path)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate ViT positional embeddings for a new resolution.")
    parser.add_argument("config", help="Path to the training YAML config file (e.g., config_hires_tune.yaml).")
    parser.add_argument("model_path", help="Path to the trained low-resolution model (.pth file).")
    parser.add_argument("output_path", help="Path to save the new high-resolution model (.pth file).")
    args = parser.parse_args()
    interpolate_pos_embeddings(args.config, args.model_path, args.output_path)