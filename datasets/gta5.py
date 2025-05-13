# datasets/gta5.py
import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

# --- Normalization Constants (same as Cityscapes) ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def create_gta5_transforms(target_size=(720, 1280)): # H, W for GTA5 training resolution
    """Creates appropriate transforms for GTA5 dataset."""
    transforms_list = []
    transforms_list.append(T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR))
    transforms_list.append(T.ToImage())
    transforms_list.append(T.ToDtype(torch.float32, scale=True))
    transforms_list.append(T.Normalize(mean=mean, std=std))
    return T.Compose(transforms_list)

class GTA5(Dataset):
    ignore_index = 255 # Cityscapes ignore index

    # Mapping from GTA5 original IDs to Cityscapes trainIds (0-18, 255 for ignore)
    # Derived from the structure of sarrrrry/PyTorchDL_GTA5/labels.py
    gta5_id_to_train_id = {
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
        7: 0,   # road
        8: 1,   # sidewalk
        9: 255, # parking
        10: 255,# rail track
        11: 2,  # building
        12: 3,  # wall
        13: 4,  # fence
        14: 255,# guard rail
        15: 255,# bridge
        16: 255,# tunnel
        17: 5,  # pole
        18: 255,# polegroup
        19: 6,  # traffic light
        20: 7,  # traffic sign
        21: 8,  # vegetation
        22: 9,  # terrain
        23: 10, # sky
        24: 11, # person
        25: 12, # rider
        26: 13, # car
        27: 14, # truck
        28: 15, # bus
        29: 255,# caravan
        30: 255,# trailer
        31: 16, # train
        32: 17, # motorcycle
        33: 18  # bicycle
    }

    # Project specifies GTA5 training resolution: 1280x720
    # PyTorch transforms: (height, width)
    default_target_size = (720, 1280) 

    def __init__(self, root_dir, split='train', transform=None, target_size=None):
        super(GTA5, self).__init__()
        self.root_dir = root_dir
        # GTA5 dataset from "Playing for Data" typically doesn't have standard 'train'/'val' splits in separate folders.
        # We'll assume all images from the provided paths are for training.
        # The 'split' argument is kept for consistency but might not be used to alter paths here.
        self.split = split 
        self.image_dir_name = 'images' 
        self.label_dir_name = 'labels' 
        
        self.target_size = target_size if target_size is not None else self.default_target_size
        self.transform = transform if transform is not None else create_gta5_transforms(target_size=self.target_size)
        
        self.files = []
        self.labels = []
        
        print(f"Initializing GTA5 dataset...") # Removed split from this message
        self._set_files()

    def _set_files(self):
        img_folder = os.path.join(self.root_dir, self.image_dir_name)
        lbl_folder = os.path.join(self.root_dir, self.label_dir_name)

        if not os.path.isdir(img_folder):
            raise FileNotFoundError(f"GTA5 Image directory not found: {img_folder}")
        if not os.path.isdir(lbl_folder):
            raise FileNotFoundError(f"GTA5 Label directory not found: {lbl_folder}")

        image_paths = sorted(glob(os.path.join(img_folder, "*.png"))) 
        
        if not image_paths:
            print(f"Warning: No image files found in {img_folder}")
            return

        for img_path in image_paths:
            base_name = os.path.basename(img_path)
            lbl_path = os.path.join(lbl_folder, base_name) 

            if os.path.exists(lbl_path):
                self.files.append(img_path)
                self.labels.append(lbl_path)
            # else: # Optional: Warn if a corresponding label is missing
            #     print(f"Warning: Label file not found for image {img_path} at {lbl_path}")
        
        print(f"Found {len(self.files)} images and {len(self.labels)} corresponding labels for GTA5.")
        if len(self.files) == 0:
            print(f"ERROR: No image/label pairs found for GTA5 in {self.root_dir}")


    def __len__(self):
        if len(self.files) == 0:
             raise RuntimeError("GTA5 Dataset initialization failed: No valid image/label pairs found.")
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx]

        try:
            image_pil = Image.open(img_path).convert('RGB')
            label_pil = Image.open(lbl_path) # Assuming labels are single-channel ID maps
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
        
        # Also map any other IDs not explicitly in our map to ignore_index
        # This is an extra check, as the np.full already initialized with ignore_index
        # but useful if some IDs from 0-33 were missed in the dict and are not meant to be a class.
        all_gta_ids_in_map = list(self.gta5_id_to_train_id.keys())
        # Create a mask for all pixels NOT in our defined gta_ids
        # For simplicity, we assume that any ID in label_np that is not a key in our map, 
        # or whose mapped value is 255, should be ultimately 255.
        # The current remapped_label_np has already handled this:
        # - initialized with 255
        # - overwritten with specific train_ids (0-18) for known gta_ids
        # - overwritten with 255 for gta_ids that map to ignore.

        label_pil_remapped = Image.fromarray(remapped_label_np)
        
        # Resize Label using NEAREST interpolation
        label_pil_resized = label_pil_remapped.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        label_tensor = torch.from_numpy(np.array(label_pil_resized)).long()

        # Apply transforms to the image (resize, ToTensor, Normalize)
        image_tensor = self.transform(image_pil)

        return image_tensor, label_tensor
