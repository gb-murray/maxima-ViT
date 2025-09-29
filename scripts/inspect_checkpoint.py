import torch
import argparse

def inspect_checkpoint(checkpoint_path: str):
    """Loads a PyTorch state_dict and prints the shape of its positional embeddings."""
    try:
        print(f"--- Inspecting: {checkpoint_path} ---")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # The key for the positional embeddings in your model
        embedding_key = 'vit.embeddings.position_embeddings'
        
        if embedding_key in state_dict:
            embedding_shape = state_dict[embedding_key].shape
            print(f"Found positional embeddings with shape: {embedding_shape}")
        else:
            print(f"Error: Could not find key '{embedding_key}' in this checkpoint.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a ViT model checkpoint.")
    parser.add_argument("checkpoint_path", help="Path to the model checkpoint (.pth file).")
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path)