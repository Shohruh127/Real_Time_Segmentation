# val/evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm 
from fvcore.nn import FlopCountAnalysis, flop_count_table 
import argparse 
from collections import OrderedDict

# Import project components (using absolute paths from project root)
from datasets.cityscapes import CityScapes 
from models.deeplabv2.deeplabv2 import get_deeplab_v2 # For DeepLabV2
from models.bisenet.build_bisenet import BiSeNet    # For BiSeNet
from utils.metrics import ConfusionMatrix             # For mIoU calculation

# --- Default Configurations ---
DEFAULT_CITYSCAPES_ROOT = "/kaggle/input/datasetcityscapes/Cityscapes/Cityscapes/Cityspaces/" # Example Kaggle path
DEFAULT_MODEL_TYPE = "deeplabv2" 
DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_INPUT_SIZE = (512, 1024) # H, W (for Cityscapes evaluation)
DEFAULT_BATCH_SIZE = 8 # Can be adjusted based on GPU memory during evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation on Cityscapes")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=['deeplabv2', 'bisenet_resnet18'],
                        help='Type of model to evaluate (deeplabv2 or bisenet_resnet18)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, 
                        help='Evaluation batch size')
    parser.add_argument('--data_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, 
                        help='Path to Cityscapes root directory for validation data')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES, help='Number of classes')
    parser.add_argument('--ignore_index', type=int, default=DEFAULT_IGNORE_INDEX, help='Ignore index for loss/metrics')
    # Add input_size if it can vary per model, though usually fixed for eval on Cityscapes
    # parser.add_argument('--input_height', type=int, default=DEFAULT_INPUT_SIZE[0])
    # parser.add_argument('--input_width', type=int, default=DEFAULT_INPUT_SIZE[1])
    args = parser.parse_args()
    return args

# --- Main Validation Function ---
def main(args): 
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {args}")

    if not os.path.isdir(args.data_root):
         print(f"ERROR: Cityscapes root directory for validation not found at {args.data_root}")
         return

    print("Loading validation dataset...")
    # Cityscapes evaluation resolution is 1024x512 (H,W -> (512, 1024))
    val_dataset = CityScapes(root_dir=args.data_root, split='val', 
                             transform_mode='val', target_size=DEFAULT_INPUT_SIZE)
    if len(val_dataset) == 0:
        print("ERROR: Validation dataset loaded 0 samples.")
        return
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")

    print(f"Initializing model type: {args.model_type}...")
    if args.model_type == 'deeplabv2':
        model = get_deeplab_v2(num_classes=args.num_classes, pretrain=False) 
    elif args.model_type == 'bisenet_resnet18':
        model = BiSeNet(num_classes=args.num_classes, context_path='resnet18')
    else:
        print(f"ERROR: Unknown model_type '{args.model_type}'")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at '{args.checkpoint}'")
        return
        
    print(f"Loading checkpoint: {args.checkpoint}")
    # --- CORRECTED torch.load call ---
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    # --- END CORRECTION ---
    
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval() 
    print("Model loaded and set to evaluation mode.")

    conf_mat_calculator = ConfusionMatrix(num_classes=args.num_classes, ignore_label=args.ignore_index)

    print("Starting validation...")
    progress_bar_val = tqdm(val_loader, desc="Validation Progress", unit="batch")
    
    with torch.no_grad():
        for images, labels in progress_bar_val:
            images = images.to(DEVICE)
            outputs = model(images) 
            preds = torch.argmax(outputs, dim=1) 
            conf_mat_calculator.update(preds.cpu(), labels.cpu()) # Ensure labels are also on CPU for numpy update

    print("\nValidation Complete.")
    
    mean_iou, iou_per_class = conf_mat_calculator.compute_iou() 
    print(f"Mean Intersection over Union (mIoU): {mean_iou:.2f}%")
    
    print("\nIoU per class:")
    cityscapes_class_names = [ # Standard Cityscapes 19-class order
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 
        'person', 'rider', 'car', 'truck', 'bus', 'train', 
        'motorcycle', 'bicycle'
    ]
    
    # Classes required for Table 3 (Domain Shift)
    required_classes_for_table3 = {'road', 'sidewalk', 'wall', 'fence', 'bicycle'}
    
    print(f"\nMetrics for Table 3 (BiSeNet trained on GTA5, evaluated on Cityscapes):")
    print(f"  Overall mIoU: {mean_iou:.2f}%")

    for i, class_iou in enumerate(iou_per_class):
        if i < len(cityscapes_class_names):
            class_name = cityscapes_class_names[i]
            print(f"  Class '{class_name}' (ID {i}): {class_iou:.2f}%")
            if class_name in required_classes_for_table3:
                print(f"    -> For Table 3 ({class_name}): {class_iou:.2f}%")
        else: # Should not happen if num_classes is 19
            print(f"  Class ID {i}: {class_iou:.2f}%")


    print("\nCalculating additional metrics (Latency, FLOPs, Params)...")
    
    # Input size for these metrics should match the model's expected eval resolution on Cityscapes
    metric_input_size = DEFAULT_INPUT_SIZE 
    dummy_input = torch.randn(1, 3, metric_input_size[0], metric_input_size[1]).to(DEVICE) 
    iterations = 100 
    latencies = []
    
    for _ in range(10): # Warm-up
        _ = model(dummy_input)
    if DEVICE.type == 'cuda': torch.cuda.synchronize() 
        
    print(f"Measuring latency over {iterations} iterations...")
    for _ in tqdm(range(iterations), desc="Latency Test"):
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = model(dummy_input)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) 

    avg_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    print(f"Latency: {avg_latency:.2f} +/- {std_latency:.2f} ms")
    print(f"FPS: {avg_fps:.2f}")

    print("\nCalculating FLOPs and Parameters...")
    try:
        model_cpu = model.to('cpu') # fvcore often prefers CPU model
        dummy_input_cpu = torch.randn(1, 3, metric_input_size[0], metric_input_size[1])
        flops_analyzer = FlopCountAnalysis(model_cpu, dummy_input_cpu)
        print(flop_count_table(flops_analyzer))
        total_flops = flops_analyzer.total()
        total_params = sum(p.numel() for p in model.parameters()) 
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total GFLOPs: {total_flops / 1e9:.2f}")
        print(f"Total Parameters (M): {total_params / 1e6:.2f}")
        print(f"Trainable Parameters (M): {trainable_params / 1e6:.2f}")
        model.to(DEVICE) # Move model back to original device
    except Exception as e:
        print(f"Could not calculate FLOPs/Params: {e}")
        model.to(DEVICE) # Ensure model is back on device if error occurred

# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args()
    main(cmd_args)
