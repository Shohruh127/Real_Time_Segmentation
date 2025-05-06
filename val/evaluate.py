import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm 
from fvcore.nn import FlopCountAnalysis, flop_count_table # For FLOPs/Params

# Import project components
from datasets.cityscapes import CityScapes # Make sure this path is correct
from models.deeplabv2.deeplabv2 import get_deeplab_v2

# --- Configuration ---
CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityspaces/" # Adjust if needed
# --- IMPORTANT: Specify which checkpoint to load for validation ---
CHECKPOINT_PATH = "./checkpoints_deeplabv2/deeplabv2_run1/deeplabv2_latest.pth.tar" # Or point to the _best.pth.tar if you implement that logic back
# CHECKPOINT_PATH = "./checkpoints_deeplabv2/deeplabv2_run1/deeplabv2_epoch_50.pth.tar" # Or specific epoch

# Model & Dataset Config
NUM_CLASSES = 19
IGNORE_INDEX = 255
INPUT_SIZE = (512, 1024) # H, W

# Evaluation Hyperparameters
BATCH_SIZE = 4 # Adjust based on GPU memory for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Function: Robust mIoU Calculation ---
def compute_iou(cm, ignore_index=255):
    """Computes Intersection over Union (IoU) from confusion matrix."""
    cm = cm.astype(np.float64) # Use float64 for precision
    
    # Intersection is diagonal elements
    intersection = np.diag(cm)
    
    # Union is sum across columns (predictions) + sum across rows (ground truth) - intersection
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    # Compute IoU per class
    iou = intersection / union
    
    # Handle ignore index (if present) and NaN (division by zero if class has 0 union)
    # Find classes that are present in ground truth (non-zero sum across rows)
    present_classes_mask = (ground_truth_set > 0)
    
    # Exclude ignore_index if it falls within num_classes range or handle separately if outside
    if 0 <= ignore_index < len(iou):
        present_classes_mask[ignore_index] = False 
        
    # Filter IoU for present classes only
    iou_present = iou[present_classes_mask]
    
    # Calculate mean IoU over present classes
    mean_iou = np.nanmean(iou_present) # Use nanmean to ignore NaN results (e.g., class with 0 union)
    
    return mean_iou * 100, iou * 100 # Return mean IoU and per-class IoU as percentages


# --- Main Validation Function ---
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset and DataLoader (Validation Only)
    print("Loading validation dataset...")
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
    
    # Handle potential DataParallel wrapping if checkpoint was saved with it
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.` prefix if it exists
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # 3. Confusion Matrix for mIoU
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # --- Validation Loop ---
    print("Starting validation...")
    progress_bar_val = tqdm(val_loader, desc="Validation Progress", unit="batch")
    
    with torch.no_grad():
        for images, labels in progress_bar_val:
            images = images.to(DEVICE)
            labels = labels.cpu().numpy() # Keep labels on CPU as numpy for confusion matrix update

            outputs = model(images) # Get logits [B, C, H, W]
            preds = torch.argmax(outputs, dim=1).cpu().numpy() # Get predicted class index [B, H, W]

            # Update confusion matrix
            for i in range(images.shape[0]): # Iterate over batch
                # Mask out ignored pixels
                valid_mask = (labels[i] != IGNORE_INDEX)
                pred_valid = preds[i][valid_mask]
                label_valid = labels[i][valid_mask]
                
                # Add to confusion matrix
                # Ensure labels are within [0, NUM_CLASSES-1]
                label_valid = np.clip(label_valid, 0, NUM_CLASSES - 1)
                pred_valid = np.clip(pred_valid, 0, NUM_CLASSES - 1)
                
                # Calculate histogram efficiently
                count = np.bincount(
                    NUM_CLASSES * label_valid.astype(np.int64) + pred_valid,
                    minlength=NUM_CLASSES**2,
                )
                conf_matrix += count.reshape(NUM_CLASSES, NUM_CLASSES)


    # --- Calculate and Print Metrics ---
    print("\nValidation Complete.")
    
    # mIoU
    mean_iou, iou_per_class = compute_iou(conf_matrix, ignore_index=IGNORE_INDEX)
    print(f"Mean Intersection over Union (mIoU): {mean_iou:.2f}%")
    # Optional: Print IoU per class
    # for i, iou in enumerate(iou_per_class):
    #    print(f"  Class {i}: {iou:.2f}%")

    # --- Calculate Other Metrics (Latency, FLOPs, Params) ---
    print("\nCalculating additional metrics...")
    
    # Latency (using README pseudo-code)
    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(DEVICE) # Single image batch
    iterations = 100 # Use fewer iterations for quicker check, 1000 recommended
    latencies = []
    
    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)
        
    if DEVICE == torch.device("cuda"): torch.cuda.synchronize() # Ensure accurate timing on GPU
        
    print(f"Measuring latency over {iterations} iterations...")
    for _ in tqdm(range(iterations), desc="Latency Test"):
        if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = model(dummy_input)
        if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Store in ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    avg_fps = 1000.0 / avg_latency
    print(f"Latency: {avg_latency:.2f} +/- {std_latency:.2f} ms")
    print(f"FPS: {avg_fps:.2f}")

    # FLOPs and Parameters (using fvcore from README)
    print("\nCalculating FLOPs and Parameters...")
    try:
        # Ensure model is on CPU for fvcore if facing issues, then move back
        model.to('cpu') 
        dummy_input_cpu = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        flops = FlopCountAnalysis(model, dummy_input_cpu)
        print(flop_count_table(flops))
        # Optionally, get total FLOPs and Params numerically
        total_flops = flops.total()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Or all params if needed
        print(f"Total GFLOPs: {total_flops / 1e9:.2f}")
        print(f"Total Trainable Params (M): {total_params / 1e6:.2f}")
        model.to(DEVICE) # Move model back to original device
    except Exception as e:
        print(f"Could not calculate FLOPs/Params: {e}")
        model.to(DEVICE) # Ensure model is back on device

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
