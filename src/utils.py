# Helper functions

from pyFAI.calibrant import Calibrant, CALIBRANT_FACTORY
from pyFAI.detectors import Detector, detector_factory
from pyFAI.geometry import Geometry
from pyFAI.geometryRefinement import GeometryRefinement
from skimage.feature import peak_local_max
from skimage import exposure
from scipy.optimize import minimize
import torch
import torch.nn as nn
import yaml
from transformers import ViTModel, ViTConfig, SwinModel, SwinConfig
from .model import MaxViT, MaxViTMultiHead, MaxSWIN
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torchvision import transforms
import numpy as np
from sklearn.metrics import mean_absolute_error
from pathlib import Path

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
            ignore_mismatched_sizes=True,
            use_safetensors=True
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

def _resolve_config(config: dict | str | Path) -> dict:
    if isinstance(config, dict):
        return config

    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"Config file must deserialize to a dict, got {type(loaded)}")
        return loaded

    raise TypeError(f"config must be a dict or YAML path, got {type(config)}")


def load_model(model_path: str, config: dict | str | Path) -> nn.Module:
    """
    Loads pre-trained weights into a fresh model architecture.
    """
    print(f"Loading weights from: {model_path}")
    
    resolved_config = _resolve_config(config)
    model = create_model(resolved_config)
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

    img_min, img_max = np.percentile(image, 1), np.percentile(image, 98.0)
    if img_max > img_min:
        image = np.clip((image - img_min) / (img_max - img_min), 0.0, 1.0)
    else:
        image = np.zeros_like(image)

    image = exposure.equalize_adapthist(image, clip_limit=0.05)

    image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).float()
        
    _ , h, w = image_tensor.shape
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    padding_transform = transforms.Pad(padding=(pad_w, pad_h))
    padded_tensor = padding_transform(image_tensor)

    resize_transform = transforms.Resize((image_size, image_size), antialias=True)
    final_tensor = resize_transform(padded_tensor)

    final_tensor = transforms.functional.normalize( # normalize to ImageNet stats
        final_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return final_tensor

def image_to_tensor_legacy(image: np.ndarray, image_size: int) -> torch.Tensor:
    """
    Legacy preprocessing for v1.6 models (Linear scaling, no ImageNet norm).
    """

    image_tensor = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)

    from torchvision.transforms import functional as F
    _, h, w = image_tensor.shape
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (pad_w, pad_h))
        
    final_tensor = F.resize(image_tensor, (image_size, image_size), antialias=True)

    return final_tensor

class PeakOptimizer:
    """
    Class to optimize peak detection and geometry refinement parameters using Nelder-Mead method.
    Arguments:
        image (np.ndarray): 2D XRD image data.
        initial_geometry (pyFAI.geometry.Geometry): Initial geometry parameters.
        calibrant (pyFAI.calibrant.Calibrant): Calibrant object.
        exclude_border (int): Number of pixels to exclude from image border during peak detection.
    """
    def __init__(self, image, initial_geometry, calibrant, exclude_border=300):
        self.image = image
        self.geo = initial_geometry
        self.calibrant = calibrant
        self.exclude_border = exclude_border
        
        self.best_params = None
        self.best_error = float('inf')
        self.best_peaks = None
        self.best_refiner = None
        self.best_geometry = None

    def _objective(self, params):    
        # min_distance must be >= 1 and an integer
        min_dist = int(max(1, round(params[0])))
        
        # threshold must be between 0 and 1
        thresh = np.clip(params[1], 0.001, 1.0)
        
        # tolerance must be positive (degrees)
        tol_deg = max(0.1, params[2])

        # peak detection
        try:
            peaks = peak_local_max(
                self.image, 
                min_distance=min_dist, 
                threshold_rel=thresh,
                exclude_border=self.exclude_border
            )
        except Exception:
            return 1e6 # penalty for crash

        if len(peaks) < 5:
            return 1e5  # penalty for too few peaks

        tth_measured = self.geo.tth(peaks[:, 0], peaks[:, 1])
        tth_expected = np.array(self.calibrant.get_2th())
        
        diff_matrix = np.abs(tth_measured[:, None] - tth_expected[None, :])
        
        min_diffs = diff_matrix.min(axis=1)
        ring_indices = diff_matrix.argmin(axis=1)
        
        mask = min_diffs < np.deg2rad(tol_deg)
        
        if np.sum(mask) < 6: 
            return 1e4 # penalty for insufficient labeled peaks

        # labeled data: [y, x, ring_index]
        valid_peaks = peaks[mask]
        valid_indices = ring_indices[mask]
        
        data = np.column_stack((valid_peaks[:, 0], valid_peaks[:, 1], valid_indices))

        # refinement
        try:
            refiner = GeometryRefinement(
                data=data,  
                dist=self.geo.dist,
                poni1=self.geo.poni1,
                poni2=self.geo.poni2,
                rot1=self.geo.rot1,
                rot2=self.geo.rot2,
                rot3=self.geo.rot3,
                pixel1=self.geo.detector.pixel1,
                pixel2=self.geo.detector.pixel2,
                detector=self.geo.detector,
                wavelength=self.calibrant.wavelength,
                calibrant=self.calibrant
            )
            
            error = refiner.refine2()
            
            if error < self.best_error:
                self.best_error = error
                self.best_params = (min_dist, thresh, tol_deg)
                self.best_peaks = data
                self.best_refiner = refiner
                self.best_geometry = Geometry(
                    dist=refiner.dist,
                    poni1=refiner.poni1,
                    poni2=refiner.poni2,
                    rot1=refiner.rot1,
                    rot2=refiner.rot2,
                    rot3=refiner.rot3,
                    wavelength=self.calibrant.wavelength,
                    detector=self.geo.detector
                )
                
            return error

        except Exception as e:
            return 1e6 # penalty for refinement failure

    def optimize(self, initial_guess=[5, 0.1, 1.0]):
        """
        Runs the optimizer.
        Initial Guess: [min_distance, threshold_rel, tolerance_degrees]
        """
        
        result = minimize( # uses nelder-mead for non-smooth objective 
            self._objective, 
            x0=initial_guess, 
            method='Nelder-Mead', 
            tol=1e-4,
            options={'maxiter': 50, 'disp': True}
        )
        
        return self.best_refiner
    
    def get_best_geometry(self):
        return self.best_geometry
    
    def get_best_refiner(self):
        return self.best_refiner