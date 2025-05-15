# datasets/gta5.py
import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T # Using torchvision.transforms.v2

# --- Normalization Constants (same as Cityscapes) ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class GTA5(Dataset):
    ignore_index = 255
    gta5_id_to_train_id = { # Your verified mapping from Turn 74
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,  8: 1,
        9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
        17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
    }
    default_target_size = (720, 1280) # H, W for GTA5 training resolution

    def __init__(self, root_dir, split='train', 
                 target_size=None,
                 use_horizontal_flip=False, 
                 use_color_jitter=False, 
                 use_gaussian_blur=False,
                 aug_probability=0.5): # Default aug probability from project spec
        super(GTA5, self).__init__()
        self.root_dir = root_dir
        self.split = split # 'split' is for consistency, GTA5 usually uses all data
        self.image_dir_name = 'images' 
        self.label_dir_name = 'labels' 
        
        self.target_size = target_size if target_size is not None else self.default_target_size
        
        self.files = []
        self.labels = []
        
        # Store augmentation flags and probability
        self.is_train_split = (split == 'train') # Only apply augmentations during training
        self.use_horizontal_flip = use_horizontal_flip
        self.use_color_jitter = use_color_jitter
        self.use_gaussian_blur = use_gaussian_blur
        self.aug_probability = aug_probability

        # Define image-only augmentations (will be wrapped with RandomApply)
        self.color_jitter_transform = None
        if self.is_train_split and self.use_color_jitter:
            self.color_jitter_transform = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        self.gaussian_blur_transform = None
        if self.is_train_split and self.use_gaussian_blur:
            # Kernel size must be odd and positive. Example:
            self.gaussian_blur_transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            
        # Core image transforms (applied after augmentations and PIL image resizing)
        self.image_to_tensor_and_norm = T.Compose([
            T.ToImage(), # Converts PIL image to Tensor HWC uint8
            T.ToDtype(torch.float32, scale=True), # Converts to float32 and scales [0,1]
            T.Normalize(mean=mean, std=std)
        ])
        
        print(f"Initializing GTA5 dataset (Flip:{use_horizontal_flip}, ColorJitter:{use_color_jitter}, Blur:{use_gaussian_blur}, P={aug_probability})...")
        self._set_files()

    def _set_files(self):
        # (Your _set_files method from Turn 71/74 - it's correct for finding files)
        img_folder = os.path.join(self.root_dir, self.image_dir_name)
        lbl_folder = os.path.join(self.root_dir, self.label_dir_name)

        if not os.path.isdir(img_folder): raise FileNotFoundError(f"GTA5 Image directory not found: {img_folder}")
        if not os.path.isdir(lbl_folder): raise FileNotFoundError(f"GTA5 Label directory not found: {lbl_folder}")

        image_paths = sorted(glob(os.path.join(img_folder, "*.png"))) 
        if not image_paths: print(f"Warning: No image files found in {img_folder}"); return

        for img_path in image_paths:
            base_name = os.path.basename(img_path)
            lbl_path = os.path.join(lbl_folder, base_name) 
            if os.path.exists(lbl_path): self.files.append(img_path); self.labels.append(lbl_path)
        
        print(f"Found {len(self.files)} images and {len(self.labels)} corresponding labels for GTA5.")
        if len(self.files) == 0: print(f"ERROR: No image/label pairs found for GTA5 in {self.root_dir}")


    def __len__(self):
        if len(self.files) == 0 and self.is_train_split :
             raise RuntimeError("GTA5 Dataset initialization failed: No valid image/label pairs found for training.")
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx]

        try:
            image_pil = Image.open(img_path).convert('RGB')
            label_pil = Image.open(lbl_path) 
        except Exception as e:
            print(f"Error loading image/label: {img_path}, {lbl_path}. Error: {e}")
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]))
            dummy_lbl = torch.full((self.target_size[0], self.target_size[1]), self.ignore_index, dtype=torch.long)
            return dummy_img, dummy_lbl

        # --- Label Remapping ---
        label_np = np.array(label_pil, dtype=np.uint8)
        remapped_label_np = np.full(label_np.shape, self.ignore_index, dtype=np.uint8) 
        for gta_id, train_id in self.gta5_id_to_train_id.items():
            remapped_label_np[label_np == gta_id] = train_id
        label_pil_remapped = Image.fromarray(remapped_label_np)
        
        # --- Apply Augmentations (if training) ---
        if self.is_train_split:
            # 1. Geometric Augmentations (applied to both image and mask)
            if self.use_horizontal_flip and torch.rand(1) < self.aug_probability:
                image_pil = T.functional.hflip(image_pil)
                label_pil_remapped = T.functional.hflip(label_pil_remapped)
            
            # (Add other geometric augmentations like RandomRotation here, applied to both)

            # 2. Appearance Augmentations (applied only to image)
            if self.color_jitter_transform and torch.rand(1) < self.aug_probability:
                image_pil = self.color_jitter_transform(image_pil)
            
            if self.gaussian_blur_transform and torch.rand(1) < self.aug_probability:
                image_pil = self.gaussian_blur_transform(image_pil)
        
        # --- Resize (after augmentations) ---
        # Image with Bilinear, Label with Nearest
        image_pil_resized = image_pil.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        label_pil_resized = label_pil_remapped.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        
        # --- Convert to Tensor and Normalize Image ---
        image_tensor = self.image_to_tensor_and_norm(image_pil_resized)
        label_tensor = torch.from_numpy(np.array(label_pil_resized)).long()

        return image_tensor, label_tensor
