# train/train_bisenet_city.py
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

# Import project components
from datasets.cityscapes import CityScapes 
from models.bisenet.build_bisenet import BiSeNet # Import BiSeNet
from utils.lr_scheduler import adjust_learning_rate # Using the same poly scheduler

# --- Default Configurations ---
DEFAULT_CITYSCAPES_ROOT = "/content/drive/MyDrive/datasets/Cityscapes/Cityscapes/" 
DEFAULT_CHECKPOINT_DIR = "./checkpoints_bisenet" # Different checkpoint dir
DEFAULT_RUN_NAME = "bisenet_resnet18_city_run1" 
DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_INPUT_SIZE = (512, 1024) # H, W
DEFAULT_NUM_EPOCHS = 50 
DEFAULT_BATCH_SIZE = 8 # BiSeNet ResNet18 is lighter, can try larger BS
DEFAULT_LEARNING_RATE = 2.5e-2 # From BiSeNet paper for Cityscapes
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4 # BiSeNet paper uses this

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 100 

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="BiSeNet Training on Cityscapes")
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
    # BiSeNet loads backbone weights internally, no separate global pretrain path needed here
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, 
                        help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=DEFAULT_RUN_NAME, 
                        help='Subdirectory name for this run checkpoints')
    args = parser.parse_args()
    return args

# --- Helper Function ---
def save_checkpoint(model, optimizer, epoch, run_ckpt_dir, filename="checkpoint.pth.tar"):
    # (Same as in train_deeplabv2.py)
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
         print("ERROR: Dataset loaded 0 samples.")
         return         
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    max_iterations = args.epochs * len(train_loader) 

    print("Initializing BiSeNet model (ResNet-18 backbone)...")
    # Instantiate BiSeNet with ResNet-18 backbone
    # The build_contextpath in BiSeNet's init handles pre-trained ResNet-18
    model = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
    model.to(DEVICE)
    print("Model initialized.")

    # Loss Function for main output and auxiliary outputs
    criterion = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)

    # Optimizer - using the model.optim_parameters() method
    optimizer = optim.SGD(model.optim_parameters(args.lr), 
                          # lr is set per group by optim_parameters
                          momentum=DEFAULT_MOMENTUM,
                          weight_decay=DEFAULT_WEIGHT_DECAY)

    # Resume from checkpoint if specified
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
                     print(f"=> Warning: Could not load optimizer state_dict: {e}.")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] 
                print(f"=> Resuming from epoch {start_epoch + 1}") 
                current_iteration = start_epoch * len(train_loader) 
        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'")
            return 
    else:
        print("=> Not resuming, starting from scratch (ResNet-18 backbone is ImageNet pre-trained).")


    print(f"Starting training from epoch {start_epoch+1} for {args.epochs} total epochs...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs): 
        model.train() 
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch")
        
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True) 

            # Learning rate adjustment
            # Use the base LR for the backbone (group 0) from args.lr
            # The adjust_learning_rate function will use this base for group 0
            # and then multiply for other groups if optimizer is set up that way.
            # Our BiSeNet.optim_parameters already sets the 10x LR for the second group.
            current_base_lr = lr_poly(args.lr, current_iteration, max_iterations) # Calculate base LR for this iteration
            optimizer.param_groups[0]['lr'] = current_base_lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = current_base_lr * 10.0


            # Forward pass - BiSeNet returns 3 outputs in training mode
            outputs, aux_out1, aux_out2 = model(images) 
            
            loss_main = criterion(outputs, labels)
            loss_aux1 = criterion(aux_out1, labels)
            loss_aux2 = criterion(aux_out2, labels)
            
            # Total loss (BiSeNet paper uses alpha=1 for auxiliary losses)
            total_loss = loss_main + loss_aux1 + loss_aux2 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            current_iteration += 1 

            current_lr_display = optimizer.param_groups[0]['lr'] # Display backbone LR
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", avg_loss=f"{epoch_loss/(i+1):.4f}", lr=f"{current_lr_display:.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} Completed. Average Training Loss: {avg_epoch_loss:.4f}")

        epoch_filename = f"bisenet_resnet18_epoch_{epoch+1}.pth.tar"
        latest_filename = f"bisenet_resnet18_latest.pth.tar"
        save_checkpoint(model, optimizer, epoch, run_checkpoint_dir, filename=epoch_filename)
        save_checkpoint(model, optimizer, epoch, run_checkpoint_dir, filename=latest_filename) 

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    print(f"Final checkpoint saved in {run_checkpoint_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args() 
    # Update global defaults with cmd_args before calling main for cleaner access, 
    # or pass cmd_args dict and use args.value everywhere in main.
    # For simplicity here, main will directly use cmd_args.
    main(cmd_args)
