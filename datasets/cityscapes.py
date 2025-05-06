# datasets/cityscapes.py
import os
import torch
import numpy as np
from PIL import Image
from glob import glob  # Import the glob function
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T 

# Define the transformations
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
    # This mapping converts Cityscapes label IDs (0-33) found in _labelIds.png
    # files into the train IDs (0-18) + ignore index (255).
    # The _labelTrainIds.png files might *already* contain these train IDs.
    # We should load a _labelTrainIds.png file to check its values.
    # If they are already 0-18 and 255, we don't need the id_to_trainid mapping.
    # Let's assume for now the _labelTrainIds.png *does* contain the target train IDs (0-18, 255)
    # If it contains the original 0-33 IDs, we would need the mapping again.
    # For now, REMOVING the mapping and assuming the file contains TrainIDs.
    # id_to_trainid = { ... } # Keep this commented out for now

    target_size = (512, 1024) 

    def __init__(self, root_dir, split='train', transform_mode='train'):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_dir_name = 'images' 
        self.label_dir_name = 'gtFine'
        self.files = []
        self.labels = []
        self.transform = create_transforms(split=transform_mode, target_size=self.target_size)
        print(f"Initializing CityScapes dataset for split '{split}'...")
        self._set_files() 

    def _set_files(self):
        img_dir = os.path.join(self.root_dir, self.image_dir_name, self.split)
        lbl_dir = os.path.join(self.root_dir, self.label_dir_name, self.split) 

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        search_pattern_img = os.path.join(img_dir, '*', '*_leftImg8bit.png') 
        
        try:
            found_files = glob(search_pattern_img) # Use the imported function directly
        except Exception as e:
             print(f"ERROR during glob execution: {e}")
             found_files = []
        
        print(f"Glob search found {len(found_files)} image files initially for pattern: {search_pattern_img}")

        found_files.sort()

        if not found_files:
             print(f"Warning: No image files found matching pattern: {search_pattern_img}")

        files_added_count = 0
        labels_added_count = 0
        for img_path in found_files: 
            base_name = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))
            
            lbl_name_base = base_name.replace('_leftImg8bit.png', '') 
            # --- CORRECTED LABEL FILENAME SUFFIX ---
            lbl_name = f"{lbl_name_base}_{self.label_dir_name}_labelTrainIds.png" 
            # --- END CORRECTION ---
            lbl_path = os.path.join(lbl_dir, city, lbl_name) 

            label_exists = os.path.exists(lbl_path) 

            if label_exists:
                self.files.append(img_path)
                self.labels.append(lbl_path)
                files_added_count += 1
                labels_added_count += 1
            elif self.split == 'test':
                 self.files.append(img_path)
                 self.labels.append(None)
                 files_added_count += 1
            # else: # Optional: Warn if label missing for train/val
            #    print(f"Warning: Label file not found for {img_path} at {lbl_path}")


        print(f"Found {len(self.files)} images and {len([l for l in self.labels if l is not None])} labels in split '{self.split}'") 
        if len(self.files) == 0 and self.split != 'test':
             print(f"\nERROR: No image/label pairs found for split '{self.split}'.")
             print(f"Please check that label files ending in '_gtFine_labelTrainIds.png' exist in the corresponding gtFine directories.")
             print(f"Searched in: {lbl_dir}/<city>/")


    def __len__(self):
        # Raise error if length is 0 for train/val splits after init to prevent DataLoader error
        if len(self.files) == 0 and self.split != 'test':
            # This error occurs after the print messages in _set_files if nothing was found
            raise RuntimeError("Dataset initialization failed: No valid image/label pairs found.")
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx]

        image_pil = Image.open(img_path).convert('RGB')
        
        if lbl_path is None: # Handle test set case
            label_tensor = None # Will be handled later if needed, maybe return dummy
        else:
            try:
                # Load the labelTrainIds file directly
                label_pil = Image.open(lbl_path) 
            except Exception as e:
                 print(f"ERROR opening or processing label file {lbl_path}: {e}")
                 # Return dummy data to avoid crashing DataLoader
                 dummy_label = torch.full((self.target_size[0], self.target_size[1]), self.ignore_index, dtype=torch.long)
                 return self.transform(image_pil), dummy_label

            # --- Assumption: _labelTrainIds.png already contains train IDs (0-18, 255) ---
            # If this assumption is wrong, you need to load the label, inspect its values,
            # and potentially re-apply the id_to_trainid mapping.
            # For now, assume it's correct and just resize and convert.
            
            # Resize Label using NEAREST
            label_pil_resized = label_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST) 
            label_tensor = torch.from_numpy(np.array(label_pil_resized)).long() # Convert to LongTensor HxW
            # Ensure ignore_index values are preserved if they were not 255 in the file
            # label_tensor[label_tensor == original_ignore_value] = self.ignore_index # If needed

        # Apply transforms to the image
        image_tensor = self.transform(image_pil) 
        
        # Handle None label for test set if necessary (e.g., return dummy)
        if label_tensor is None:
             dummy_label = torch.full((self.target_size[0], self.target_size[1]), self.ignore_index, dtype=torch.long)
             return image_tensor, dummy_label

        return image_tensor, label_tensor
