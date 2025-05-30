# datasets/cityscapes.py
import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T 

# --- Normalization Constants ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# This function already accepts target_size
def create_transforms(split='train', target_size=(512, 1024)): # H, W
    """Creates appropriate transforms for Cityscapes."""
    transforms_list = []
    transforms_list.append(T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR)) 
    transforms_list.append(T.ToImage()) 
    transforms_list.append(T.ToDtype(torch.float32, scale=True)) 
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
    default_cityscapes_target_size = (512, 1024) # Default H, W for Cityscapes itself

    # --- MODIFY THIS __INIT__ METHOD ---
    def __init__(self, root_dir, split='train', transform_mode='train', target_size=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_dir_name = 'images' 
        self.label_dir_name = 'gtFine'
        self.files = []
        self.labels = []
        
        # Use the passed target_size if provided, otherwise use the class default
        current_target_size = target_size if target_size is not None else self.default_cityscapes_target_size
        self.target_size_for_getitem = current_target_size # Store for label resizing in __getitem__

        # Pass the determined target_size to create_transforms for the image
        self.transform = create_transforms(split=transform_mode, target_size=current_target_size)
        
        print(f"Initializing CityScapes dataset for split '{split}' with target_size {current_target_size}...")
        self._set_files() 
    # --- END MODIFICATION ---

    def _set_files(self):
        # ... (your _set_files method from Turn 55 - it's correct) ...
        img_dir = os.path.join(self.root_dir, self.image_dir_name, self.split)
        lbl_dir = os.path.join(self.root_dir, self.label_dir_name, self.split) 

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        search_pattern_img = os.path.join(img_dir, '*', '*_leftImg8bit.png') 
        
        found_files = glob(search_pattern_img)
        print(f"Glob search found {len(found_files)} image files initially for pattern: {search_pattern_img}")
        found_files.sort()

        if not found_files:
             print(f"Warning: No image files found matching pattern: {search_pattern_img}")

        for img_path in found_files: 
            base_name = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))
            
            lbl_name_base = base_name.replace('_leftImg8bit.png', '') 
            lbl_name = f"{lbl_name_base}_{self.label_dir_name}_labelTrainIds.png" # Using _labelTrainIds.png
            lbl_path = os.path.join(lbl_dir, city, lbl_name) 

            if os.path.exists(lbl_path):
                self.files.append(img_path)
                self.labels.append(lbl_path)
            elif self.split == 'test':
                 self.files.append(img_path)
                 self.labels.append(None)
        
        print(f"Found {len(self.files)} images and {len([l for l in self.labels if l is not None])} labels in split '{self.split}'") 
        if len(self.files) == 0 and self.split != 'test':
             print(f"\nERROR: No image/label pairs found for Cityscapes split '{self.split}'. Check paths and label names.")


    def __len__(self):
        if len(self.files) == 0 and self.split != 'test':
            raise RuntimeError(f"CityScapes Dataset (split: {self.split}) initialization failed: No valid image/label pairs found.")
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx]

        image_pil = Image.open(img_path).convert('RGB')
        
        if lbl_path is None: 
            label_tensor = None
        else:
            try:
                label_pil = Image.open(lbl_path) 
            except Exception as e:
                 print(f"ERROR opening or processing label file {lbl_path}: {e}")
                 dummy_label = torch.full(self.target_size_for_getitem, self.ignore_index, dtype=torch.long)
                 return self.transform(image_pil), dummy_label

            # Assuming _labelTrainIds.png already contains train IDs (0-18, 255)
            # Resize Label using NEAREST
            label_pil_resized = label_pil.resize((self.target_size_for_getitem[1], self.target_size_for_getitem[0]), Image.NEAREST) 
            label_tensor = torch.from_numpy(np.array(label_pil_resized)).long() 

        # Apply transforms to the image
        image_tensor = self.transform(image_pil) 
        
        if label_tensor is None: # For test set
             dummy_label = torch.full(self.target_size_for_getitem, self.ignore_index, dtype=torch.long)
             return image_tensor, dummy_label

        return image_tensor, label_tensor
