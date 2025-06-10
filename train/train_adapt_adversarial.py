# train/train_adapt_adversarial.py
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

# Import project components (ensure paths are correct relative to project root)
from datasets.gta5 import GTA5
from datasets.cityscapes import CityScapes 
from models.bisenet.build_bisenet import BiSeNet # This is our Generator
from models.discriminator import Discriminator # The new Discriminator model
from utils.lr_scheduler import lr_poly 

# --- Default Configurations ---
DEFAULT_GTA5_ROOT = "/kaggle/input/datasetcityscapes/GTA5/GTA5/" 
DEFAULT_CITYSCAPES_ROOT = "/kaggle/input/datasetcityscapes/Cityscapes/Cityscapes/Cityspaces/" 
DEFAULT_CHECKPOINT_DIR = "/kaggle/working/checkpoints_bisenet_adversarial" 
DEFAULT_RUN_NAME = "bisenet_gta5_adversarial_run1" 
DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_GTA5_TRAIN_SIZE = (720, 1280) 
DEFAULT_CITYSCAPES_TRAIN_SIZE = (512, 1024) 
DEFAULT_NUM_EPOCHS = 50 
DEFAULT_BATCH_SIZE = 4 
DEFAULT_LR_G = 2.5e-2 # Learning rate for Generator (BiSeNet)
DEFAULT_LR_D = 1e-4  # Learning rate for Discriminator
DEFAULT_LAMBDA_ADV = 0.001 # Weight for the adversarial loss term
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
AUX_LOSS_WEIGHT = 1.0 # Weight for BiSeNet's auxiliary segmentation losses

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Domain Adaptation with BiSeNet")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for both GTA5 and Cityscapes loaders')
    parser.add_argument('--lr_g', type=float, default=DEFAULT_LR_G, help='Initial learning rate for Generator (BiSeNet)')
    parser.add_argument('--lr_d', type=float, default=DEFAULT_LR_D, help='Initial learning rate for Discriminator')
    parser.add_argument('--lambda_adv', type=float, default=DEFAULT_LAMBDA_ADV, help='Weight for adversarial loss')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Total epochs')
    parser.add_argument('--gta5_root', type=str, default=DEFAULT_GTA5_ROOT, help='Path to GTA5 root')
    parser.add_argument('--cityscapes_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, help='Path to Cityscapes root for UNLABELED target data')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=DEFAULT_RUN_NAME, help='Run name for checkpoints')
    
    # Use best augmentation setting from Step 3b
    parser.add_argument('--aug_hflip', action='store_true', help='Enable Random Horizontal Flip')
    parser.add_argument('--aug_colorjitter', action='store_true', help='Enable Color Jitter')
    parser.add_argument('--aug_gblur', action='store_true', help='Enable Gaussian Blur')
    
    args = parser.parse_args()
    return args

# --- Helper Function: save_checkpoint ---
def save_checkpoint(model, optimizer, epoch, run_ckpt_dir, filename="checkpoint.pth.tar"):
    # Simplified save function without best_miou tracking
    state = {
        'epoch': epoch + 1, 
        'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
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

    # --- Models ---
    print("Initializing Generator (BiSeNet) and Discriminator models...")
    generator = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
    discriminator = Discriminator(num_classes=DEFAULT_NUM_CLASSES)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator.to(DEVICE)
    discriminator.to(DEVICE)
    print("Models initialized.")

    # --- DataLoaders ---
    # Source (GTA5) Loader with labels and best augmentations from Step 3b
    print("Loading datasets...")
    gta_train_dataset = GTA5(root_dir=args.gta5_root, target_size=DEFAULT_GTA5_TRAIN_SIZE,
                             use_horizontal_flip=args.aug_hflip, use_color_jitter=args.aug_colorjitter,
                             use_gaussian_blur=args.aug_gblur)
    source_loader = DataLoader(gta_train_dataset, batch_size=args.batch_size, shuffle=True, 
                               num_workers=2, pin_memory=True, drop_last=True)
    
    # Target (Cityscapes) Loader for UNLABELED images
    cityscapes_target_dataset = CityScapes(root_dir=args.cityscapes_root, split='train',
                                           target_size=DEFAULT_CITYSCAPES_TRAIN_SIZE)
    target_loader = DataLoader(cityscapes_target_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"Loaded {len(gta_train_dataset)} GTA5 source images and {len(cityscapes_target_dataset)} Cityscapes target images.")
    
    # --- Optimizers ---
    optimizer_G = optim.SGD(generator.module.optim_parameters(args.lr_g) if isinstance(generator, nn.DataParallel) else generator.optim_parameters(args.lr_g), 
                          momentum=DEFAULT_MOMENTUM, weight_decay=DEFAULT_WEIGHT_DECAY)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.99))

    # --- Losses ---
    criterion_seg = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)
    criterion_adv = nn.BCEWithLogitsLoss().to(DEVICE)

    source_label_val = 0
    target_label_val = 1
    
    # --- Resume from Checkpoint ---
    # (Resume logic can be added here if needed, simplified for clarity)

    # --- Training Loop ---
    print(f"Starting Adversarial Training for {args.epochs} epochs...")
    start_time = time.time()
    
    len_source_loader = len(source_loader)
    len_target_loader = len(target_loader)
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        # We'll iterate up to the length of the larger loader
        loop_len = max(len_source_loader, len_target_loader)
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        progress_bar = tqdm(range(loop_len), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i in progress_bar:
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            
            # Get target batch (from Cityscapes)
            try:
                target_images, _ = next(target_iter) 
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            target_images = target_images.to(DEVICE)

            # --- D loss on Target (should be predicted as "target" -> 1) ---
            target_preds, _, _ = generator(target_images)
            D_out_target = discriminator(target_preds.detach()) # Detach from generator graph
            D_target_labels = torch.full(D_out_target.size(), target_label_val, dtype=torch.float, device=DEVICE)
            loss_D_target = criterion_adv(D_out_target, D_target_labels)
            loss_D_target.backward()

            # Get source batch (from GTA5)
            try:
                source_images, source_seg_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_images, source_seg_labels = next(source_iter)
            source_images = source_images.to(DEVICE)

            # --- D loss on Source (should be predicted as "source" -> 0) ---
            source_preds, _, _ = generator(source_images)
            D_out_source = discriminator(source_preds.detach())
            D_source_labels = torch.full(D_out_source.size(), source_label_val, dtype=torch.float, device=DEVICE)
            loss_D_source = criterion_adv(D_out_source, D_source_labels)
            loss_D_source.backward()
            
            # Update the discriminator
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            # 1. Adversarial Loss (Generator tries to fool Discriminator)
            # We use the source predictions but want the discriminator to think they are target (label 1)
            D_out_gen = discriminator(source_preds) # Use the same source_preds, but DO NOT detach
            loss_G_adv = criterion_adv(D_out_gen, D_target_labels) # Compare with target label (1)

            # 2. Segmentation Loss on Source
            # BiSeNet returns 3 outputs, calculate loss for all
            loss_G_seg_main = criterion_seg(source_preds, source_seg_labels.to(DEVICE))
            # Need to get aux outputs from the forward pass again, as they were detached
            _, aux_out1, aux_out2 = generator(source_images) 
            loss_G_seg_aux1 = criterion_seg(aux_out1, source_seg_labels.to(DEVICE))
            loss_G_seg_aux2 = criterion_seg(aux_out2, source_seg_labels.to(DEVICE))
            loss_G_seg = loss_G_seg_main + AUX_LOSS_WEIGHT * (loss_G_seg_aux1 + loss_G_seg_aux2)

            # Total Generator loss and update
            loss_G = loss_G_seg + args.lambda_adv * loss_G_adv
            loss_G.backward()
            optimizer_G.step()
            
            progress_bar.set_postfix(
                D_Loss=f"{loss_D_source.item() + loss_D_target.item():.4f}", 
                G_Loss_Seg=f"{loss_G_seg.item():.4f}", 
                G_Loss_Adv=f"{loss_G_adv.item():.4f}"
            )

        # --- Checkpoint Saving ---
        # No per-epoch validation, just save at the end of each epoch
        epoch_filename = f"{args.run_name}_epoch_{epoch+1}.pth.tar"
        latest_filename = f"{args.run_name}_latest.pth.tar"
        save_checkpoint(generator, optimizer_G, epoch, run_checkpoint_dir, filename=epoch_filename)
        save_checkpoint(generator, optimizer_G, epoch, run_checkpoint_dir, filename=latest_filename)
        print(f"\nEpoch {epoch+1} training completed. Checkpoint saved.")

    total_training_time = time.time() - start_time
    print(f"\nAdversarial training finished in {total_training_time / 3600:.2f} hours.")
    print(f"Final checkpoint saved in {run_checkpoint_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args() 
    main(cmd_args)
