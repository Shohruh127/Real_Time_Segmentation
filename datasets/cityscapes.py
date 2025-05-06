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

def create_transforms(split='train', target_size=(512, 1024)):
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
    target_size = (512, 1024) # H, W for transforms

    def __init__(self, root_dir, split='train', transform_mode='train'):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_dir_name = 'images' 
        self.label_dir_name = 'gtFine'
        self.files = []
        self.labels = []
        self.transform = create_transforms(split=transform_mode, target_size=self.target_size)
        self._set_files() # Call helper function to populate files

    # --- REPLACED _set_files method with DEBUG version ---
    def _set_files(self):
        img_dir = os.path.join(self.root_dir, self.image_dir_name, self.split)
        lbl_dir = os.path.join(self.root_dir, self.label_dir_name, self.split) 

        # --- Start Debug Prints ---
        print(f"\n[DEBUG _set_files] Using root_dir: {self.root_dir}")
        print(f"[DEBUG _set_files] Checking img_dir: {img_dir}")
        
        if not os.path.isdir(img_dir):
            print(f"[DEBUG _set_files] ERROR: img_dir does NOT exist or is not a directory!")
            return 

        print(f"[DEBUG _set_files] img_dir confirmed to exist. Trying os.listdir...")
        try:
            contents = os.listdir(img_dir)
            print(f"[DEBUG _set_files] os.listdir found {len(contents)} items in img_dir.")
            # Optional: print first few items found by listdir
            # print(f"[DEBUG _set_files] First few items via listdir: {contents[:5]}") 
        except Exception as e:
            print(f"[DEBUG _set_files] ERROR listing contents of img_dir with os.listdir: {e}")
            return 

        # Define the pattern, ensure it's correct
        search_pattern_img = os.path.join(img_dir, '*', '*_leftImg8bit.png') 
        print(f"[DEBUG _set_files] Using glob pattern: {search_pattern_img}")
        
        found_files = [] # Initialize empty list
        try:
            # CRITICAL CHECK: Print length immediately after glob call
            found_files = glob.glob(search_pattern_img)
            print(f"[DEBUG _set_files] glob.glob returned {len(found_files)} files.") 
        except Exception as e:
             print(f"[DEBUG _set_files] ERROR during glob.glob execution: {e}")
             # found_files remains empty
        # --- End Debug Prints within glob section ---

        found_files.sort()

        if not found_files:
             if len(contents) > 0:
                  print(f"[DEBUG _set_files] WARNING: os.listdir found items, but glob found 0 files with pattern '{search_pattern_img}'. Check pattern/permissions?")
             else:
                  print(f"[DEBUG _set_files] WARNING: found_files list is empty and os.listdir was also empty or failed.")
        
        # --- Loop to find image/label pairs ---
        files_added_count = 0
        labels_added_count = 0
        for img_path in found_files: # Loop will not run if found_files is empty
            base_name = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))
            
            lbl_name_base = base_name.replace('_leftImg8bit.png', '') # Use correct suffix replacement
            lbl_name = f"{lbl_name_base}_{self.label_dir_name}_labelIds.png" 
            lbl_path = os.path.join(lbl_dir, city, lbl_name)

            if os.path.exists(lbl_path):
                self.files.append(img_path)
                self.labels.append(lbl_path)
                files_added_count += 1
                labels_added_count += 1
            elif self.split == 'test':
                 self.files.append(img_path)
                 self.labels.append(None)
                 files_added_count += 1
            # else: 
            #    pass # Skip image if label doesn't exist for train/val

        # Final summary print from within the method
        print(f"[DEBUG _set_files] FINAL Counts within _set_files: Images added={files_added_count}, Labels added={labels_added_count}")
        # Original print statement (will show 0 if self.files wasn't populated)
        print(f"Found {len(self.files)} images and {len([l for l in self.labels if l is not None])} labels in split '{self.split}'") 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx]

        image_pil = Image.open(img_path).convert('RGB')
        
        if lbl_path is None:
            label_pil = None 
            label_remapped_np = None
        else:
            label_pil = Image.open(lbl_path) 
            label_np = np.array(label_pil, dtype=np.uint8)
            label_remapped_np = np.full(label_np.shape, self.ignore_index, dtype=np.uint8) 
            for k, v in self.id_to_trainid.items():
                 mask = (label_np == k)
                 label_remapped_np[mask] = v
            
            label_pil_remapped = Image.fromarray(label_remapped_np)
            label_pil_resized = label_pil_remapped.resize((self.target_size[1], self.target_size[0]), Image.NEAREST) 
            label_tensor = torch.from_numpy(np.array(label_pil_resized)).long() 

        image_tensor = self.transform(image_pil) 
        
        if label_pil is None:
             dummy_label = torch.full((self.target_size[0], self.target_size[1]), self.ignore_index, dtype=torch.long)
             return image_tensor, dummy_label

        return image_tensor, label_tensor
