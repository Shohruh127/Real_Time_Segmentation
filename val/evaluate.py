# val/evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm 
from fvcore.nn import FlopCountAnalysis, flop_count_table 
import argparse # Add argparse
from collections import OrderedDict

# Import project components
from datasets.cityscapes import CityScapes 
# Import both model types
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.bisenet.build_bisenet import BiSeNet 
from utils.metrics import ConfusionMatrix 

# --- Default Configurations ---
DEFAULT_CITYSCAPES_ROOT = "/content/data_local/CityscapesDataset/"
# DEFAULT_CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityscapes/" 
DEFAULT_CHECKPOINT_PATH = None # Will be set by argument
DEFAULT_MODEL_TYPE = "deeplabv2" # Default to deeplabv2

DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_INPUT_SIZE = (512, 1024) # H, W
DEFAULT_BATCH_SIZE = 2 # Adjust based on GPU memory for validation

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
                        help='Path to Cityscapes root directory')
    # Add other relevant arguments if needed, e.g., num_classes, input_size
    args = parser.parse_args()
    return args

# --- Main Validation Function ---
def main(args): 
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {args}")

    if not os.path.isdir(args.data_root):
         print(f"ERROR: Cityscapes root directory not found at {args.data_root}")
         return

    print("Loading validation dataset...")
    val_dataset = CityScapes(root_dir=args.data_root, split='val', transform_mode='val')
    if len(val_dataset) == 0:
        print("ERROR: Validation dataset loaded 0 samples.")
        return
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")

    print(f"Initializing model type: {args.model_type}...")
    # --- Conditional Model Initialization ---
    if args.model_type == 'deeplabv2':
        model = get_deeplab_v2(num_classes=DEFAULT_NUM_CLASSES, pretrain=False) # pretrain=False as we load a checkpoint
    elif args.model_type == 'bisenet_resnet18':
        model = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
        # Ensure BiSeNet's build_contextpath uses pretrained=True for ResNet18 from torchvision
    else:
        print(f"ERROR: Unknown model_type '{args.model_type}'")
        return
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at '{args.checkpoint}'")
        return
        
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval() 
    print("Model loaded and set to evaluation mode.")

    # Initialize Confusion Matrix object
    conf_mat_calculator = ConfusionMatrix(num_classes=DEFAULT_NUM_CLASSES, ignore_label=DEFAULT_IGNORE_INDEX)

    print("Starting validation...")
    progress_bar_val = tqdm(val_loader, desc="Validation Progress", unit="batch")
    
    with torch.no_grad():
        for images, labels in progress_bar_val:
            images = images.to(DEVICE)
            # labels stay on CPU/numpy for the update method
            
            # Both DeepLabV2 and BiSeNet (in eval mode) should return only main logits
            outputs = model(images) 
            preds = torch.argmax(outputs, dim=1) 

            conf_mat_calculator.update(preds.cpu(), labels) 

    print("\nValidation Complete.")
    
    mean_iou, iou_per_class = conf_mat_calculator.compute_iou() 
    print(f"Mean Intersection over Union (mIoU): {mean_iou:.2f}%")
    # Optional: Print IoU per class
    # print("IoU per class:")
    # for i, iou_val in enumerate(iou_per_class): # Renamed iou to iou_val to avoid conflict
    #    print(f"  Class {i}: {iou_val:.2f}%")


    print("\nCalculating additional metrics...")
    
    # Latency 
    # Assuming DEFAULT_INPUT_SIZE is correct for both models for this metric
    dummy_input = torch.randn(1, 3, DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1]).to(DEVICE) 
    iterations = 100 
    latencies = []
    
    for _ in range(10): # Warm-up
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
    std_latency = np.std(latencies)
    avg_fps = 1000.0 / avg_latency
    print(f"Latency: {avg_latency:.2f} +/- {std_latency:.2f} ms")
    print(f"FPS: {avg_fps:.2f}")

    # FLOPs and Parameters
    print("\nCalculating FLOPs and Parameters...")
    try:
        model.to('cpu') 
        dummy_input_cpu = torch.randn(1, 3, DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1])
        # If model has specific FLOPs counting method, use it, else fvcore
        # For BiSeNet, you might need to ensure it's compatible or implement a counter
        flops = FlopCountAnalysis(model, dummy_input_cpu)
        print(flop_count_table(flops))
        total_flops = flops.total()
        total_params = sum(p.numel() for p in model.parameters()) 
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total GFLOPs: {total_flops / 1e9:.2f}")
        print(f"Total Parameters (M): {total_params / 1e6:.2f}")
        print(f"Trainable Parameters (M): {trainable_params / 1e6:.2f}")
        model.to(DEVICE) 
    except Exception as e:
        print(f"Could not calculate FLOPs/Params: {e}")
        model.to(DEVICE) 

# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args()
    main(cmd_args)
