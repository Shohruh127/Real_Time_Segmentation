# train/train_deeplabv2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm 
import numpy as np 
import argparse 
from collections import OrderedDict 

# --- Default Configurations (Module Level) ---
DEFAULT_CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityspaces/" 
DEFAULT_PRETRAINED_WEIGHTS_PATH = "/content/drive/MyDrive/deeplab_resnet_pretrained_imagenet.pth" 
DEFAULT_CHECKPOINT_DIR = "./checkpoints_deeplabv2" 
DEFAULT_RUN_NAME = "deeplabv2_run1" 
DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_INPUT_SIZE = (512, 1024) # H, W
DEFAULT_NUM_EPOCHS = 50 
DEFAULT_BATCH_SIZE = 2 
DEFAULT_LEARNING_RATE = 1e-3 
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 100 

# Import project components using absolute paths from project root
from datasets.cityscapes import CityScapes 
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils.lr_scheduler import adjust_learning_rate 


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabV2 Training on Cityscapes")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, 
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, 
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, 
                        help='Total number of epochs to train')
    parser.add_argument('--data_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, 
                        help='Path to Cityscapes root directory')
    parser.add_argument('--weights', type=str, default=DEFAULT_PRETRAINED_WEIGHTS_PATH, 
                        help='Path to pretrained ImageNet weights (used if not resuming)')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, 
                        help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=DEFAULT_RUN_NAME, 
                        help='Subdirectory name for this run checkpoints')
    args = parser.parse_args()
    return args

# --- Helper Function ---
def save_checkpoint(model, optimizer, epoch, run_ckpt_dir, filename="checkpoint.pth.tar"):
    state = {
        'epoch': epoch + 1, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = os.path.join(run_ckpt_dir, filename)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    torch.save(state, filepath)

# --- Main Training Function ---
def main(args): 
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {args}")

    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {run_checkpoint_dir}")

    start_epoch = 0
    current_iteration = 0 

    if not os.path.isdir(args.data_root):
         print(f"ERROR: Cityscapes root directory not found at {args.data_root}")
         return

    print("Loading training dataset...")
    train_dataset = CityScapes(root_dir=args.data_root, split='train', transform_mode='train')
    if len(train_dataset) == 0:
         print("ERROR: Dataset loaded 0 samples. Check dataset path and implementation.")
         return
         
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    max_iterations = args.epochs * len(train_loader) 

    print("Initializing model...")
    # Load ImageNet pre-trained weights only if not resuming from a specific training checkpoint
    load_imagenet_pretrained = (args.resume is None) 
    
    model = get_deeplab_v2(num_classes=DEFAULT_NUM_CLASSES, 
                           pretrain=load_imagenet_pretrained, 
                           pretrain_model_path=args.weights if load_imagenet_pretrained else None)
    
    if load_imagenet_pretrained and not os.path.exists(args.weights):
        print(f"ERROR: Pretrained weights for ImageNet not found at {args.weights}")
        return
    
    model.to(DEVICE)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)

    optimizer = optim.SGD(model.optim_parameters(args.lr), # Pass args.lr here
                          lr=args.lr, 
                          momentum=DEFAULT_MOMENTUM,
                          weight_decay=DEFAULT_WEIGHT_DECAY)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("=> Loaded model state_dict")

            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> Loaded optimizer state_dict")
                except ValueError as e:
                     print(f"=> Warning: Could not load optimizer state_dict: {e}. Starting optimizer from scratch.")
            else:
                 print("=> Warning: Optimizer state_dict not found in checkpoint. Starting optimizer from scratch.")

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] 
                print(f"=> Resuming from epoch {start_epoch + 1}") # Display next epoch to run
                current_iteration = start_epoch * len(train_loader) 
            else:
                 print("=> Warning: Epoch not found in checkpoint, starting from epoch 0.")
        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'")
            return 
    else:
        print("=> Not resuming, starting from scratch.")


    print(f"Starting training from epoch {start_epoch+1} for {args.epochs} total epochs...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs): 
        model.train() 
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch")
        
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True) 

            adjust_learning_rate(optimizer, current_iteration, max_iterations, args.lr)

            outputs, _, _ = model(images) 
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_iteration += 1 

            current_lr_display = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss/(i+1):.4f}", lr=f"{current_lr_display:.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} Completed. Average Training Loss: {avg_epoch_loss:.4f}")

        epoch_filename = f"deeplabv2_epoch_{epoch+1}.pth.tar"
        latest_filename = f"deeplabv2_latest.pth.tar"
        save_checkpoint(model, optimizer, epoch, run_checkpoint_dir, filename=epoch_filename)
        save_checkpoint(model, optimizer, epoch, run_checkpoint_dir, filename=latest_filename) 

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    print(f"Final checkpoint saved in {run_checkpoint_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    run_args = parse_args() 
    main(run_args)
