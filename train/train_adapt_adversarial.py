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
from models.discriminator import Discriminator      # The new Discriminator model
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
DEFAULT_CITYSCAPES_EVAL_SIZE = (512, 1024) 
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
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for both loaders')
    parser.add_argument('--lr_g', type=float, default=DEFAULT_LR_G, help='Initial LR for Generator')
    parser.add_argument('--lr_d', type=float, default=DEFAULT_LR_D, help='Initial LR for Discriminator')
    parser.add_argument('--lambda_adv', type=float, default=DEFAULT_LAMBDA_ADV, help='Weight for adversarial loss')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Total epochs')
    parser.add_argument('--gta5_root', type=str, default=DEFAULT_GTA5_ROOT, help='Path to GTA5 root')
    parser.add_argument('--cityscapes_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, help='Path to Cityscapes root')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=DEFAULT_RUN_NAME, help='Run name for checkpoints')
    
    # Augmentation Flags (use best setting from Step 3b)
    parser.add_argument('--aug_hflip', action='store_true', help='Enable Random Horizontal Flip')
    parser.add_argument('--aug_colorjitter', action='store_true', help='Enable Color Jitter')
    parser.add_argument('--aug_gblur', action='store_true', help='Enable Gaussian Blur')
    
    args = parser.parse_args()
    return args

# --- Helper Function: save_checkpoint ---
def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, run_ckpt_dir, filename="checkpoint.pth.tar"):
    # This saves state for both models and optimizers to allow resuming adversarial training
    state = {
        'epoch': epoch + 1, 
        'generator_state_dict': generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict(),
        'discriminator_state_dict': discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
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
    
    # --- Models ---
    print("Initializing Generator (BiSeNet) and Discriminator models...")
    generator = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
    discriminator = Discriminator(num_classes=DEFAULT_NUM_CLASSES)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator.to(DEVICE)
    discriminator.to(DEVICE)
    print("Models initialized.")

    # --- DataLoaders ---
    print("Loading datasets...")
    gta_train_dataset = GTA5(root_dir=args.gta5_root, target_size=DEFAULT_GTA5_TRAIN_SIZE,
                             use_horizontal_flip=args.aug_hflip, use_color_jitter=args.aug_colorjitter,
                             use_gaussian_blur=args.aug_gblur)
    source_loader = DataLoader(gta_train_dataset, batch_size=args.batch_size, shuffle=True, 
                               num_workers=2, pin_memory=True, drop_last=True)
    
    cityscapes_target_dataset = CityScapes(root_dir=args.cityscapes_root, split='train',
                                           target_size=DEFAULT_CITYSCAPES_TRAIN_SIZE)
    target_loader = DataLoader(cityscapes_target_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"Loaded {len(gta_train_dataset)} GTA5 source images and {len(cityscapes_target_dataset)} Cityscapes target images.")
    
    # --- Optimizers ---
    optimizer_G = optim.SGD(
        generator.module.optim_parameters(args.lr_g) if isinstance(generator, nn.DataParallel) else generator.optim_parameters(args.lr_g), 
        momentum=DEFAULT_MOMENTUM, weight_decay=DEFAULT_WEIGHT_DECAY
    )
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.99))

    # --- Losses ---
    criterion_seg = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)
    criterion_adv = nn.BCEWithLogitsLoss().to(DEVICE)

    source_label_val = 0
    target_label_val = 1
    
    # --- Resume from Checkpoint ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
            
            gen_state_dict = checkpoint['generator_state_dict']
            new_gen_state_dict = OrderedDict()
            for k, v in gen_state_dict.items(): name = k[7:] if k.startswith('module.') else k; new_gen_state_dict[name] = v
            
            # Load into base model if DataParallel is used
            if isinstance(generator, nn.DataParallel): generator.module.load_state_dict(new_gen_state_dict)
            else: generator.load_state_dict(new_gen_state_dict)
            print("=> Loaded Generator state_dict")

            disc_state_dict = checkpoint['discriminator_state_dict']
            new_disc_state_dict = OrderedDict()
            for k, v in disc_state_dict.items(): name = k[7:] if k.startswith('module.') else k; new_disc_state_dict[name] = v

            if isinstance(discriminator, nn.DataParallel): discriminator.module.load_state_dict(new_disc_state_dict)
            else: discriminator.load_state_dict(new_disc_state_dict)
            print("=> Loaded Discriminator state_dict")

            if 'optimizer_G_state_dict' in checkpoint: optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict']); print("=> Loaded Optimizer_G state_dict")
            if 'optimizer_D_state_dict' in checkpoint: optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict']); print("=> Loaded Optimizer_D state_dict")
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] 
            
            print(f"=> Resuming from epoch {start_epoch + 1}")
        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'"); return 
    else:
        print("=> Not resuming, starting from scratch.")

    # --- Training Loop ---
    print(f"Starting Adversarial Training for {args.epochs} epochs...")
    start_time = time.time()
    
    max_iterations = args.epochs * len(source_loader)
    current_iteration = start_epoch * len(source_loader)
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        target_iter = iter(target_loader)
        progress_bar = tqdm(enumerate(source_loader), total=len(source_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (source_images, source_seg_labels) in progress_bar:
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            
            # 1. D loss on real (target) data
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            target_images = target_images.to(DEVICE)

            with torch.no_grad():
                target_preds, _, _ = generator(target_images)
            
            D_out_target = discriminator(target_preds.detach())
            loss_D_target = criterion_adv(D_out_target, torch.full_like(D_out_target, target_label_val))
            loss_D_target.backward()

            # 2. D loss on fake (source) data
            source_images = source_images.to(DEVICE)
            with torch.no_grad():
                source_preds, _, _ = generator(source_images)
                
            D_out_source = discriminator(source_preds.detach())
            loss_D_source = criterion_adv(D_out_source, torch.full_like(D_out_source, source_label_val))
            loss_D_source.backward()
            
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            
            # Re-evaluate BiSeNet's output for generator training to keep gradients
            final_pred_G, aux1_pred_G, aux2_pred_G = generator(source_images)

            # 1. Segmentation Loss on Source
            loss_G_seg_main = criterion_seg(final_pred_G, source_seg_labels.to(DEVICE))
            loss_G_seg_aux1 = criterion_seg(aux1_pred_G, source_seg_labels.to(DEVICE))
            loss_G_seg_aux2 = criterion_seg(aux2_pred_G, source_seg_labels.to(DEVICE))
            loss_G_seg = loss_G_seg_main + AUX_LOSS_WEIGHT * (loss_G_seg_aux1 + loss_G_seg_aux2)
            
            # 2. Adversarial Loss (to fool discriminator)
            D_out_adv = discriminator(final_pred_G)
            loss_G_adv = criterion_adv(D_out_adv, torch.full_like(D_out_adv, target_label_val))
            
            # Total Generator loss and update
            loss_G = loss_G_seg + args.lambda_adv * loss_G_adv
            loss_G.backward()
            optimizer_G.step()
            
            # --- Update LR and Progress Bar ---
            # Adjust learning rate for the GENERATOR's optimizer
            lr_group0 = lr_poly(args.lr_g, current_iteration, max_iterations) 
            optimizer_G.param_groups[0]['lr'] = lr_group0 # CORRECTED: Use optimizer_G
                if len(optimizer_G.param_groups) > 1:
                    lr_group1 = lr_poly(args.lr_g * 10.0, current_iteration, max_iterations) 
                    optimizer_G.param_groups[1]['lr'] = lr_group1 # CORRECTED: Use optimizer_G
            
            current_iteration += 1
            progress_bar.set_postfix(
                D_Loss=f"{loss_D_source.item() + loss_D_target.item():.4f}", 
                G_Loss_Seg=f"{loss_G_seg.item():.4f}", 
                G_Loss_Adv=f"{loss_G_adv.item():.4f}"
            )

        # --- Checkpoint Saving (No per-epoch validation) ---
        epoch_filename = f"{args.run_name}_epoch_{epoch+1}.pth.tar"
        latest_filename = f"{args.run_name}_latest.pth.tar"
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, -1, run_checkpoint_dir, filename=epoch_filename)
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, -1, run_checkpoint_dir, filename=latest_filename)
        print(f"\nEpoch {epoch+1} training completed. Checkpoint saved.")

    total_training_time = time.time() - start_time
    print(f"\nAdversarial training finished in {total_training_time / 3600:.2f} hours.")
    print(f"Final checkpoint saved in {run_checkpoint_dir}")

# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args() 
    main(cmd_args)
