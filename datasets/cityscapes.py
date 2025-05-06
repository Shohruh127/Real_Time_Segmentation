# datasets/cityscapes.py
import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T # Use v2 for better joint image/label transforms

# Define the transformations
# Normalization constants for ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create separate transforms for train and validation/test
# (We might add augmentations like flips later for training if needed,
# but for Step 2a, let's keep it simple)

def create_transforms(split='train', target_size=(512, 1024)):
    """Creates appropriate transforms for Cityscapes."""
    transforms_list = []
    
    # Resizing: PIL Images expected by Resize
    # Note: target_size is usually (Height, Width) for PyTorch transforms
    transforms_list.append(T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR)) # Resize image
    # For labels, use NEAREST neighbor interpolation to avoid creating new label values
    # We need a way to apply resize differently to image and label,
    # so we'll handle label resize manually in __getitem__ after initial object creation.
    # Or better, use functional transforms within __getitem__ if PIL resize is needed,
    # or use torchvision.transforms.v2 which handles this better. Let's use v2.

    # Convert PIL Image to tensor (doesn't normalize yet)
    # v2.ToImage() converts PIL/numpy to torch Tensor. v2.ToDtype handles type conversion.
    transforms_list.append(T.ToImage()) # Converts PIL image to Tensor HWC uint8
    transforms_list.append(T.ToDtype(torch.float32, scale=True)) # Converts to float32 and scales [0, 1]
    
    # Normalization (only for image)
    transforms_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms_list)


class CityScapes(Dataset):
    ignore_index = 255
    id_to_trainid = {
        0: ignore_index, 1: ignore_index, 2: ignore_index, 3: ignore_index, 4: ignore_index, 5: ignore_index, 
        6: ignore_index, 7: 0, 8: 1, 9: ignore_index, 10: ignore_index, 11: 2, 12: 3, 13: 4, 
        14: ignore_index, 15: ignore_index, 16: ignore_index, 17: 5, 18: ignore_index, 19: 6, 20: 7, 
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_index, 
        30: ignore_index, 31: 16, 32: 17, 33: 18, -1: ignore_index
    }
    target_size = (512, 1024) # H, W for transforms

    def __init__(self, root_dir, split='train', transform_mode='train'):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split
        # Use 'images' based on user's path instead of 'leftImg8bit'
        self.image_dir_name = 'images' 
        self.label_dir_name = 'gtFine'
        self.files = []
        self.labels = []
        
        # Create transforms based on split
        # For now, train and val transforms are the same basic ones
        self.transform = create_transforms(split=transform_mode, target_size=self.target_size)

        self._set_files()

    def _set_files(self):
        img_dir = os.path.join(self.root_dir, self.image_dir_name, self.split)
        lbl_dir = os.path.join(self.root_dir, self.label_dir_name, self.split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        # Removed label dir check here, handle missing labels in loop

        search_pattern_img = os.path.join(img_dir, '*', f'*_{self.image_dir_name}.png') # Adapt pattern
        found_files = glob(search_pattern_img)
        found_files.sort()

        if not found_files:
             print(f"Warning: No image files found matching pattern: {search_pattern_img}")
             return

        for img_path in found_files:
            base_name = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))
            
            # Use the correct label suffix
            lbl_name_base = base_name.replace(f'_{self.image_dir_name}.png', '')
            lbl_name = f"{lbl_name_base}_{self.label_dir_name}_labelIds.png" 
            lbl_path = os.path.join(lbl_dir, city, lbl_name)

            if os.path.exists(lbl_path):
                self.files.append(img_path)
                self.labels.append(lbl_path)
            elif self.split == 'test':
                 self.files.append(img_path)
                 self.labels.append(None)
            # else: # Optional: Add warning back if needed for train/val
            #      print(f"Warning: Label file not found for image {img_path}. Expected at: {lbl_path}")

        print(f"Found {len(self.files)} images and {len([l for l in self.labels if l is not None])} labels in split '{self.split}'")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx]

        image_pil = Image.open(img_path).convert('RGB')
        
        if lbl_path is None:
            label_pil = None # Keep label as None if no path (test set)
            label_remapped_np = None
        else:
            label_pil = Image.open(lbl_path) # Load original label IDs
            
            # Remap IDs using numpy for efficiency
            label_np = np.array(label_pil, dtype=np.uint8)
            label_remapped_np = np.full(label_np.shape, self.ignore_index, dtype=np.uint8) # Start with ignore_index
            for k, v in self.id_to_trainid.items():
                mask = (label_np == k)
                label_remapped_np[mask] = v
            
            # Convert remapped numpy array back to PIL for potential transforms (if needed)
            # or keep as numpy/tensor if transforms handle it directly.
            # For v2 transforms, we can pass tensors directly.
            # Let's resize here manually using PIL for label to ensure NEAREST interpolation
            label_pil_remapped = Image.fromarray(label_remapped_np)
            # Resize Label using NEAREST
            label_pil_resized = label_pil_remapped.resize((self.target_size[1], self.target_size[0]), Image.NEAREST) 
            label_tensor = torch.from_numpy(np.array(label_pil_resized)).long() # Convert to LongTensor HxW

        # Apply composed transforms to the image
        # Transforms expect PIL Image, convert PIL to Tensor, normalize
        image_tensor = self.transform(image_pil) # This resizes, converts to float tensor [0,1], normalizes
        
        # Handle None label case
        if label_pil is None:
             # Return a dummy label or handle appropriately later
             # For loss calculation, we need a tensor, maybe zeros or ignore_index filled
             dummy_label = torch.full((self.target_size[0], self.target_size[1]), self.ignore_index, dtype=torch.long)
             return image_tensor, dummy_label

        return image_tensor, label_tensor # Return transformed image tensor and remapped/resized label tensor
