# train/train_deeplabv2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm # For progress bars
import numpy as np 
import argparse # Needed for command-line arguments (like --resume)
from collections import OrderedDict # Needed for fixing state_dict keys

# Import project components using absolute paths from project root
from datasets.cityscapes import CityScapes 
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils.lr_scheduler import adjust_learning_rate # Assuming lr_poly is also in this file if needed by adjust_learning_rate

# --- Configuration ---
# These might be overridden by command-line args or a config file later
CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityspaces/" # Adjust if needed
PRETRAINED_WEIGHTS_PATH = "/content/drive/MyDrive/deeplab_resnet_pretrained_imagenet.pth" # Your path
CHECKPOINT_DIR = "./checkpoints_deeplabv2" # Directory to save model checkpoints
RUN_NAME = "deeplabv2_run1" # Name for this specific training run's checkpoints

# Model & Dataset Config
NUM_CLASSES = 19
IGNORE_INDEX = 255
INPUT_SIZE = (512, 1024) # H, W

# Training Hyperparameters
NUM_EPOCHS = 50 # As per project spec
BATCH_SIZE = 2 # Reduced based on previous OOM error, adjust if needed
LEARNING_RATE = 1e-3 # Initial LR for backbone
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Other Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 100 # Print training loss every N batches

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabV2 Training on Cityscapes")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the checkpoint to resume from (e.g., ./checkpoints_deeplabv2/deeplabv2_run1/deeplabv2_latest.pth.tar)')
    # Add arguments for other configs if desired
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Total number of epochs to train')
    parser.add_argument('--data_root', type=str, default=CITYSCAPES_ROOT, help='Path to Cityscapes root directory')
    parser.add_argument('--weights', type=str, default=PRETRAINED_WEIGHTS_PATH, help='Path to pretrained weights')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=RUN_NAME, help='Subdirectory name for this run checkpoints')


    args = parser.parse_args()
    # Update global vars based on args (or pass args dict around)
    global BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, CITYSCAPES_ROOT, PRETRAINED_WEIGHTS_PATH, CHECKPOINT_DIR, RUN_NAME
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.epochs
    CITYSCAPES_ROOT = args.data_root
    PRETRAINED_WEIGHTS_PATH = args.weights
    CHECKPOINT_DIR = args.checkpoint_dir
    RUN_NAME = args.run_name
    
    return args

# --- Helper Function ---
def save_checkpoint(model, optimizer, epoch, run_ckpt_dir, filename="checkpoint.pth.tar"):
    """Saves model checkpoint."""
    state = {
        'epoch': epoch + 1, # Save the next epoch to start from
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = os.path.join(run_ckpt_dir, filename)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    torch.save(state, filepath)
    # print(f"Checkpoint saved to {filepath}") # Optional

# --- Main Training Function ---
def main(args): # Accept parsed arguments
    print(f"Using device: {DEVICE}")
    
    # Determine checkpoint directory for this run
    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, RUN_NAME)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {run_checkpoint_dir}")

    start_epoch = 0
    current_iteration = 0 

    # Optional: Check if dataset directory exists
    if not os.path.isdir(CITYSCAPES_ROOT):
         print(f"ERROR: Cityscapes root directory not found at {CITYSCAPES_ROOT}")
         return

    # 1. Dataset and DataLoader (Training Only)
    print("Loading training dataset...")
    train_dataset = CityScapes(root_dir=CITYSCAPES_ROOT, split='train', transform_mode='train')
    # Check if dataset loading failed (length is 0)
    if len(train_dataset) == 0:
         print("ERROR: Dataset loaded 0 samples. Check dataset path and implementation.")
         return
         
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True) # Reduced workers
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    # Recalculate max_iterations based on actual loader length
    max_iterations = NUM_EPOCHS * len(train_loader) 

    # 2. Model
    print("Initializing model...")
    model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=(args.resume is None)) # Only load imagenet weights if not resuming

    # Load pre-trained weights only if NOT resuming from a checkpoint
    if args.resume is None:
        if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
            print(f"ERROR: Pretrained weights not found at {PRETRAINED_WEIGHTS_PATH}")
            return
        print(f"Loading ImageNet pre-trained weights from: {PRETRAINED_WEIGHTS_PATH}")
        # The get_deeplab_v2 function handles loading these when pretrain=True
    
    model.to(DEVICE)
    print("Model initialized.")

    # 3. Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(DEVICE)

    # 4. Optimizer
    optimizer = optim.SGD(model.optim_parameters(LEARNING_RATE),
                          lr=LEARNING_RATE, # Base LR is set here but adjusted by scheduler
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)

    # --- Checkpoint Loading Logic ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            
            # Load Model State
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("=> Loaded model state_dict")

            # Load Optimizer State
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> Loaded optimizer state_dict")
                except ValueError as e:
                     print(f"=> Warning: Could not load optimizer state_dict: {e}. Starting optimizer from scratch.")
                     # Optionally reset optimizer: optimizer = optim.SGD(...)
            else:
                 print("=> Warning: Optimizer state_dict not found in checkpoint. Starting optimizer from scratch.")

            # Load Epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] # Start from the next epoch
                print(f"=> Resuming from epoch {start_epoch}")
                current_iteration = start_epoch * len(train_loader) 
            else:
                 print("=> Warning: Epoch not found in checkpoint, starting from epoch 0.")

        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'")
            return # Exit if specified checkpoint doesn't exist
    else:
        print("=> Not resuming, starting from scratch.")
    # --- End Checkpoint Loading ---


    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch+1} for {NUM_EPOCHS} total epochs...")
    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS): 
        model.train() 
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")
        
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True) 

            adjust_learning_rate(optimizer, current_iteration, max_iterations, LEARNING_RATE)

            outputs, _, _ = model(images) 
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_iteration += 1 

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss/(i+1):.4f}", lr=f"{current_lr:.6f}")

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Completed. Average Training Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint (epoch number here is the one just completed)
        # Pass run_checkpoint_dir to save function
        epoch_filename = f"deeplabv2_epoch_{epoch+1}.pth.tar"
        latest_filename = f"deeplabv2_latest.pth.tar"
        save_checkpoint(model, optimizer, epoch, run_checkpoint_dir, filename=epoch_filename)
        save_checkpoint(model, optimizer, epoch, run_checkpoint_dir, filename=latest_filename) 

    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    print(f"Final checkpoint saved in {run_checkpoint_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    run_args = parse_args() # Parse arguments
    main(run_args)          # Pass arguments to main
