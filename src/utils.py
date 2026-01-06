# Helper functions

from pyFAI.calibrant import Calibrant, CALIBRANT_FACTORY
from pyFAI.detectors import Detector, detector_factory
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, SwinModel, SwinConfig
from .model import MaxViT, MaxViTMultiHead, MaxSWIN
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torchvision import transforms
import numpy as np
from sklearn.metrics import mean_absolute_error

def get_calibrant(alias: str, wavelength: float) -> Calibrant:

    calibrant = CALIBRANT_FACTORY(alias)
    calibrant.wavelength = wavelength

    return calibrant

def get_detector(alias:str) -> Detector:

    detector = detector_factory(alias)

    return detector

# def create_model(config: dict) -> nn.Module:

#     """
#     Instantiates a model with the specified architecture and random weights.
#     """
#     print(f"Creating new model architecture from config...")
    
#     # Load the configuration of the pretrained model
#     model_config = ViTConfig.from_pretrained(
#         config['model']['backbone'],
#         image_size=config['model'].get('image_size', 224)
#     )
    
#     # Build the model from the configuration (initializes with random weights)
#     vit_backbone = ViTModel(model_config)

#     use_multi_head = config['model'].get('multi_head', False)

#     if use_multi_head:
#         print("Initializing multi-regression architecture...")
#         return MaxViTMultiHead(vit_backbone, hidden_dim=config['model']['vit_hidden_dim'])
    
#     else:
#         print("Initializing single-regression architecture...")
#         regression_head = nn.Sequential(
#             nn.Linear(config['model']['vit_hidden_dim'], 512),
#             nn.GELU(), nn.Dropout(0.1),
#             nn.Linear(512, config['model']['num_outputs'])
#         )
#         return MaxViTModel(vit_backbone, regression_head)
    
def create_model(config: dict) -> nn.Module:
    """
    Instantiates a ViT or Swin model with the specified architecture.
    """
    backbone_name = config['model']['backbone']
    image_size = config['model'].get('image_size', 224)
    hidden_dim = config['model']['hidden_dim']
    num_outputs = config['model']['num_outputs']
    use_multi_head = config['model'].get('multi_head', False)

    print(f"Initilizing model architecture: {backbone_name}")
    print(f"Target Resolution: {image_size}x{image_size}")

    # build the regression head
    if use_multi_head:
        print("Initializing multi-regression architecture...")
        if "swin" in backbone_name.lower():
            raise NotImplementedError("Multi-head logic not yet ported to Swin factory.")
        else:
            pass #TODO: Reimplement multi-head

    else:
        print("Initializing regression architecture...")
        regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_outputs)
        )

    # build the backbone
    if "swin" in backbone_name.lower():
        print("Loading SWIN backbone...")
        
        swin_config = SwinConfig.from_pretrained(backbone_name)
        swin_config.image_size = image_size
        
        backbone = SwinModel.from_pretrained(
            backbone_name, 
            config=swin_config,
            ignore_mismatched_sizes=True
        )
        
        return MaxSWIN(backbone, regression_head)

    else:
        print("Loading ViT backbone...")
        
        vit_config = ViTConfig.from_pretrained(backbone_name)
        vit_config.image_size = image_size
        
        backbone = ViTModel.from_pretrained(
            backbone_name, 
            config=vit_config,
            ignore_mismatched_sizes=True
        )
        
        return MaxViT(backbone, regression_head)

def load_model(model_path: str, config: dict) -> nn.Module:
    """
    Loads pre-trained weights into a fresh model architecture.
    """
    print(f"Loading weights from: {model_path}")
    
    model = create_model(config)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    try: 
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded full model state.")
    except RuntimeError as e:
        print(f"Could not load full state dict ({e}). Trying to load backbone only...")
        model.vit.load_state_dict(state_dict, strict=False)
        print("Successfully loaded backbone weights only.")
    
    return model

def freeze_backbone(model: nn.Module):
    """
    Freezes the parameters of the ViT backbone.
    """
    print("Freezing backbone weights. Only the regression head will be trained.")
    for param in model.vit.parameters(): #type: ignore
        param.requires_grad = False

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler, writer, epoch):
    """
    Worker function to train model weights for a single epoch.
    """
    model.train()
    total_loss = 0

    num_batches = len(dataloader)

    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        is_last_batch = (i == num_batches - 1)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            predictions = model(images)
            loss = loss_fn(predictions, labels)
        
        scaler.scale(loss).backward()

        if is_last_batch:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / num_batches

def validate(model, dataloader, loss_fn, device, writer, epoch):
    """
    Returns validation loss for the current model/epoch.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    """
    Converts a 2D diffraction pattern from an array to a tensor.
    The image is padded, resampled, and normalized to fit model specifications.
    """
    image = np.log1p(image)

    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)

    image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        
    _ , h, w = image_tensor.shape
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    padding_transform = transforms.Pad(padding=(pad_w, pad_h))
    padded_tensor = padding_transform(image_tensor)

    resize_transform = transforms.Resize((image_size, image_size), antialias=True)
    final_tensor = resize_transform(padded_tensor)

    final_tensor = transforms.Normalize( # normalize to ImageNet stats
        final_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return final_tensor