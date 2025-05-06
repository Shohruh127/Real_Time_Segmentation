# val/evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm 
from fvcore.nn import FlopCountAnalysis, flop_count_table # For FLOPs/Params

# --- Configuration ---
CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityspaces/" # Adjust if needed
# --- IMPORTANT: Specify which checkpoint to load for validation ---
CHECKPOINT_PATH = "./checkpoints_deeplabv2/deeplabv2_run1/deeplabv2_latest.pth.tar" # Example: latest checkpoint
# CHECKPOINT_PATH = "./checkpoints_deeplabv2/deeplabv2_run1/deeplabv2_epoch_50.pth.tar" # Example: final epoch checkpoint

# Model & Dataset Config
NUM_CLASSES = 19
IGNORE_INDEX = 255
INPUT_SIZE = (512, 1024) # H, W

# Evaluation Hyperparameters
BATCH_SIZE = 4 # Adjust based on GPU memory for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Main Validation Function ---
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset and DataLoader (Validation Only)
    print("Loading validation dataset...")
    # Optional: Check if dataset directory exists
    if not os.path.isdir(CITYSCAPES_ROOT):
         print(f"ERROR: Cityscapes root directory not found at {CITYSCAPES_ROOT}")
         return
    val_dataset = CityScapes(root_dir=CITYSCAPES_ROOT, split='val', transform_mode='val') # Use 'val' split
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("Validation dataset loaded.")

    # 2. Model
    print("Initializing model...")
    # Initialize model structure (do not load Imagenet weights here, will load checkpoint)
    model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=False) 
    
    # Load checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return
        
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle potential DataParallel wrapping
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # 3. Initialize Confusion Matrix object
    conf_mat_calculator = ConfusionMatrix(num_classes=NUM_CLASSES, ignore_label=IGNORE_INDEX)

    # --- Validation Loop ---
    print("Starting validation...")
    progress_bar_val = tqdm(val_loader, desc="Validation Progress", unit="batch")
    
    with torch.no_grad():
        for images, labels in progress_bar_val:
            images = images.to(DEVICE)
            # labels stay on CPU/numpy for the update method
            
            outputs = model(images) # Get logits [B, C, H, W]
            preds = torch.argmax(outputs, dim=1) # Get predicted class index [B, H, W]

            # Update confusion matrix using the class method
            conf_mat_calculator.update(preds.cpu(), labels) # Pass cpu tensors or numpy arrays

    # --- Calculate and Print Metrics ---
    print("\nValidation Complete.")
    
    # mIoU - Use the method from the ConfusionMatrix object
    mean_iou, iou_per_class = conf_mat_calculator.compute_iou() 
    print(f"Mean Intersection over Union (mIoU): {mean_iou:.2f}%")
    # Optional: Print IoU per class
    # print("IoU per class:")
    # for i, iou in enumerate(iou_per_class):
    #    print(f"  Class {i}: {iou:.2f}%")

    # --- Calculate Other Metrics (Latency, FLOPs, Params) ---
    print("\nCalculating additional metrics...")
    
    # Latency 
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(DEVICE) 
    iterations = 100 
    latencies = []
    
    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)
        
    if DEVICE == torch.device("cuda"): torch.cuda.synchronize() 
        
    print(f"Measuring latency over {iterations} iterations...")
    for _ in tqdm(range(iterations), desc="Latency Test"):
        if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = model(dummy_input)
        if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) 

    avg_latency = np.mean(latencies)
    std_latency = np
