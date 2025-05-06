# train/train_deeplabv2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm # For progress bars
import numpy as np # Keep for potential future use

from datasets.cityscapes import CityScapes 
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils.lr_scheduler import adjust_learning_rate 

# --- Configuration ---
CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityscapes/" # Adjust if needed
PRETRAINED_WEIGHTS_PATH = "/content/drive/MyDrive/deeplab_resnet_pretrained_imagenet.pth" # Your path
CHECKPOINT_DIR = "./checkpoints_deeplabv2" # Directory to save model checkpoints
RUN_NAME = "deeplabv2_run1" # Name for this specific training run's checkpoints

# Model & Dataset Config
NUM_CLASSES = 19
IGNORE_INDEX = 255
INPUT_SIZE = (512, 1024) # H, W

# Training Hyperparameters
NUM_EPOCHS = 50 # As per project spec
BATCH_SIZE = 2 # Adjust based on your GPU memory
LEARNING_RATE = 1e-3 # Initial LR for backbone
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Other Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 100 # Print training loss every N batches

# --- Helper Function ---
def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    """Saves model checkpoint (simplified)."""
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    # print(f"Checkpoint saved to {filename}") # Optional print

# --- Main Training Function ---
def main():
    print(f"Using device: {DEVICE}")
    # Create a specific directory for this run's checkpoints
    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, RUN_NAME)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {run_checkpoint_dir}")

    # Optional: Check if dataset directory exists
    if not os.path.isdir(CITYSCAPES_ROOT):
         print(f"ERROR: Cityscapes root directory not found at {CITYSCAPES_ROOT}")
         return

    # 1. Dataset and DataLoader (Training Only)
    print("Loading training dataset...")
    train_dataset = CityScapes(root_dir=CITYSCAPES_ROOT, split='train', transform_mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print("Training dataset loaded.")

    # 2. Model
    print("Initializing model...")
    if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"ERROR: Pretrained weights not found at {PRETRAINED_WEIGHTS_PATH}")
        return
    model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path=PRETRAINED_WEIGHTS_PATH)
    model.to(DEVICE)
    print("Model initialized.")

    # 3. Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(DEVICE)

    # 4. Optimizer
    optimizer = optim.SGD(model.optim_parameters(LEARNING_RATE),
                          lr=LEARNING_RATE, # Base LR is set here but adjusted by scheduler
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)

    # Calculate total iterations for poly LR scheduler
    max_iterations = NUM_EPOCHS * len(train_loader)
    current_iteration = 0

    # --- Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")
        
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True) 

            # Adjust learning rate (using imported function)
            adjust_learning_rate(optimizer, current_iteration, max_iterations, LEARNING_RATE)

            # Forward pass
            outputs, _, _ = model(images) 
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_iteration += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss/(i+1):.4f}", lr=f"{current_lr:.6f}")

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Completed. Average Training Loss: {avg_epoch_loss:.4f}")

        # Save latest checkpoint after each epoch
        epoch_filename = os.path.join(run_checkpoint_dir, f"deeplabv2_epoch_{epoch+1}.pth.tar")
        latest_filename = os.path.join(run_checkpoint_dir, f"deeplabv2_latest.pth.tar")
        save_checkpoint(model, optimizer, epoch, filename=epoch_filename)
        save_checkpoint(model, optimizer, epoch, filename=latest_filename) # Overwrite latest


    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    print(f"Final checkpoint saved in {run_checkpoint_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    # Add imports needed only here, if any
    main()
