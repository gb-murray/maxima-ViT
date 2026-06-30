# Data generation, augmentation, and Dataset classes

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import torch.nn.functional as F
from torchvision import transforms
   
class DiffractionDataset(Dataset):
    """
    Handles large training datasets in an HDF5 format and applies dynamic domain randomization.
    """
    def __init__(
        self,
        hdf5_path: str,
        group: str = 'train',
        image_size: int = 1056,
    ):
        self.hdf5_path = hdf5_path  
        self.group = group
        self.image_size = image_size
        self.file = None
        self.digital_twin = False
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = len(f[self.group]['images'])

            if 'normalization' in f:
                self.centers = np.array(f['normalization']['centers'], dtype=np.float32)
                self.scale_factors = np.array(f['normalization']['scale_factors'], dtype=np.float32)
            else:
                raise ValueError(f"Normalization data not found in HDF5 file: {self.hdf5_path}")

            if 'background_model' in f and self.group == 'train':
                self.digital_twin = True
                
                self.W = np.array(f['background_model']['W'], dtype=np.float32) 
                self.H = np.array(f['background_model']['H'], dtype=np.float32)
                self.master_mask = np.array(f['background_model']['master_mask'], dtype=bool)
                self.mask_intensity = int(f['background_model']['mask_intensity'][0])

                self.num_bg_samples = self.W.shape[0]
                print(f"[{group.upper()}] Loaded synthetic training set.")
            elif 'background_model' in f and self.group == 'test':
                self.digital_twin = False
                print(f"[{group.upper()}] Loaded synthetic test set.")
            else:
                self.digital_twin = False
                print(f"[{group.upper()}] Loaded experimental training set.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')

        image = self.file[self.group]['images'][idx] 
        label = self.file[self.group]['labels'][idx] 

        if self.digital_twin:
            background = self.__generate_background__(self.H, self.W)[0]
            map = np.clip(image + background, a_min=0, a_max=None)
            pattern = np.random.poisson(map).astype(np.float32)
            pattern[self.master_mask] = self.mask_intensity

        else:
            pattern = image.astype(np.float32)

        image_tensor = self.__to_tensor__(pattern) #type: ignore
        label_tensor = torch.from_numpy(label).float() #type: ignore
        
        return image_tensor, label_tensor
    
    def __generate_background__(self, components, weights):
        n_frames, n_comps = weights.shape
        
        random_idx = np.random.randint(0, n_frames)
        base_weights = weights[random_idx].copy()
        
        global_exposure_jitter = np.random.uniform(0.9, 1.1)
        independent_jitter = np.random.uniform(0.95, 1.05, size=n_comps)
        final_weights = base_weights * global_exposure_jitter * independent_jitter

        synthetic_bg = final_weights @ components
        synthetic_bg = synthetic_bg.reshape(components.shape[1], components.shape[2])

        return synthetic_bg, final_weights

    def __to_tensor__(self, image: np.ndarray) -> torch.Tensor:
        """Converts a numpy array to a PyTorch tensor with shape (1, H, W)."""
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0, None)
        
        image = np.log1p(image)

        img_min, img_max = np.percentile(image, 1), np.percentile(image, 98.0)
        if img_max > img_min:
            image = np.clip((image - img_min) / (img_max - img_min), 0.0, 1.0)
        else:
            image = np.zeros_like(image)

        image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).float()
            
        _ , h, w = image_tensor.shape
        max_dim = max(h, w)

        pad_bottom = max_dim - h
        pad_right = max_dim - w
        
        padded_tensor = F.pad(image_tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=0.0)

        resize_transform = transforms.Resize((self.image_size, self.image_size), antialias=True)
        final_tensor = resize_transform(padded_tensor)

        final_tensor = transforms.functional.normalize( # normalize to ImageNet stats
            final_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return final_tensor