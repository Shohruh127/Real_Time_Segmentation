# train/train_gta5_aug.py
import torch
# ... (all other necessary imports from train_gta5.py) ...
from datasets.gta5 import GTA5 # Ensure this is the updated GTA5 class
# ...

# --- Default Configurations (Update RUN_NAME based on args later) ---
# ... (most defaults can be the same as train_gta5.py) ...
DEFAULT_RUN_NAME = "bisenet_gta5_augmented_run" # Generic, will be refined

# --- Argument Parser (Add augmentation flags) ---
def parse_args():
    parser = argparse.ArgumentParser(description="BiSeNet Training on GTA5 with Augmentations")
    # ... (copy all arguments from train_gta5.py's parse_args) ...
    parser.add_argument('--gta5_root', type=str, default=DEFAULT_GTA5_ROOT, help='Path to GTA5 root')
    parser.add_argument('--cityscapes_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, help='Path to Cityscapes root for validation')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory for checkpoints')
    
    # --- New Augmentation Arguments ---
    parser.add_argument('--aug_hflip', action='store_true', help='Enable Random Horizontal Flip')
    parser.add_argument('--aug_colorjitter', action='store_true', help='Enable Color Jitter')
    parser.add_argument('--aug_gblur', action='store_true', help='Enable Gaussian Blur')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability for applying selected augmentations')
    
    parser.add_argument('--run_name', type=str, default=DEFAULT_RUN_NAME, 
                        help='Subdirectory name for this run checkpoints')
    args = parser.parse_args()

    # Dynamically set run_name based on applied augmentations if not explicitly set
    if args.run_name == DEFAULT_RUN_NAME: # Only override if default is used
        active_augs = []
        if args.aug_hflip: active_augs.append("hflip")
        if args.aug_colorjitter: active_augs.append("cjitter")
        if args.aug_gblur: active_augs.append("gblur")
        if not active_augs:
            args.run_name = "bisenet_gta5_no_aug_run" # Or could be from train_gta5.py
        else:
            args.run_name = f"bisenet_gta5_aug_{'_'.join(active_augs)}_run1"
            
    return args

# --- Main Training and Validation Function ---
def main(args): 
    # ... (Same setup as train_gta5.py: device, checkpoint_dir, start_epoch, etc.) ...
    global BEST_CITYSCAPES_MIOU
    BEST_CITYSCAPES_MIOU = 0.0 # Reset for each run

    print(f"Using device: {DEVICE}")
    print(f"Configuration: {args}") # Will show augmentation flags

    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name) # Uses dynamic run_name
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {run_checkpoint_dir}")


    # --- Datasets and DataLoaders ---
    print("Loading GTA5 training dataset with augmentations...")
    gta_train_dataset = GTA5(
        root_dir=args.gta5_root,
        target_size=DEFAULT_GTA5_TRAIN_SIZE,
        use_horizontal_flip=args.aug_hflip,
        use_color_jitter=args.aug_colorjitter,
        use_gaussian_blur=args.aug_gblur,
        aug_probability=args.aug_prob
    )
    # ... (rest of DataLoader setup for gta_train_loader and city_val_loader as in train_gta5.py) ...
    if len(gta_train_dataset) == 0: print("ERROR: GTA5 Training dataset loaded 0 samples."); return         
    gta_train_loader = DataLoader(gta_train_dataset, batch_size=args.gta5_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"GTA5 training dataset loaded with {len(gta_train_dataset)} samples.")

    # Cityscapes for Validation (no augmentations here)
    if not os.path.isdir(args.cityscapes_root):
         print(f"ERROR: Cityscapes root for validation not found at {args.cityscapes_root}"); return
    print("Loading Cityscapes validation dataset...")
    city_val_dataset = CityScapes(root_dir=args.cityscapes_root, split='val', 
                                  transform_mode='val', target_size=DEFAULT_CITYSCAPES_EVAL_SIZE)
    if len(city_val_dataset) == 0: print("ERROR: Cityscapes Validation dataset loaded 0 samples."); return
    city_val_loader = DataLoader(city_val_dataset, batch_size=args.city_val_bs, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Cityscapes validation dataset loaded with {len(city_val_dataset)} samples.")
    
    # ... (Rest of the main() function: max_iterations, model, loss, optimizer, resume logic, training loop, validation loop, checkpoint saving) ...
    # The training loop and validation loop logic can remain IDENTICAL to train_gta5.py.
    # The only change is that gta_train_dataset now applies augmentations.
    # Make sure checkpoint filenames inside save_checkpoint also use args.run_name implicitly or explicitly.
    # Example: epoch_filename = f"{args.run_name}_epoch_{epoch+1}.pth.tar"
    # The save_checkpoint function should take run_checkpoint_dir which already includes run_name.

    max_iterations = args.epochs * len(gta_train_loader) 

    print("Initializing BiSeNet model (ResNet-18 backbone)...")
    model = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
    model.to(DEVICE)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)
    optimizer = optim.SGD(model.optim_parameters(args.lr), momentum=DEFAULT_MOMENTUM, weight_decay=DEFAULT_WEIGHT_DECAY)

    if args.resume:
        # ... (Standard resume logic - make sure it loads BEST_CITYSCAPES_MIOU if saved) ...
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False) # Added weights_only=False
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            current_iteration = start_epoch * len(gta_train_loader)
            if 'best_cityscapes_miou' in checkpoint: BEST_CITYSCAPES_MIOU = checkpoint['best_cityscapes_miou']
            print(f"=> Resuming from epoch {start_epoch + 1}. Best Cityscapes mIoU: {BEST_CITYSCAPES_MIOU:.2f}%")
        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'"); return 
    else:
        print("=> Not resuming, starting from scratch (ResNet-18 backbone is ImageNet pre-trained).")
    
    print(f"Starting training on GTA5 from epoch {start_epoch+1} for {args.epochs} total epochs...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs): 
        model.train() 
        epoch_train_loss = 0.0
        train_progress_bar = tqdm(gta_train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [GTA5 Train]", unit="batch")
        
        for i, (images, labels) in enumerate(train_progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True) 

            lr_group0 = lr_poly(args.lr, current_iteration, max_iterations) 
            optimizer.param_groups[0]['lr'] = lr_group0
            if len(optimizer.param_groups) > 1:
                lr_group1 = lr_poly(args.lr * 10.0, current_iteration, max_iterations) 
                optimizer.param_groups[1]['lr'] = lr_group1

            outputs, aux_out1, aux_out2 = model(images) 
            loss_main = criterion(outputs, labels)
            loss_aux1 = criterion(aux_out1, labels)
            loss_aux2 = criterion(aux_out2, labels)
            total_loss = loss_main + AUX_LOSS_WEIGHT * loss_aux1 + AUX_LOSS_WEIGHT * loss_aux2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()
            current_iteration += 1 
            current_lr_display = optimizer.param_groups[0]['lr']
            train_progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", avg_loss=f"{epoch_train_loss/(i+1):.4f}", lr=f"{current_lr_display:.6f}")

        avg_epoch_train_loss = epoch_train_loss / len(gta_train_loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} GTA5 Training Completed. Avg Loss: {avg_epoch_train_loss:.4f}")

        # --- Validation Phase on Cityscapes ---
        model.eval()
        print(f"Starting validation on Cityscapes for epoch {epoch+1}...")
        val_conf_mat = ConfusionMatrix(num_classes=DEFAULT_NUM_CLASSES, ignore_label=DEFAULT_IGNORE_INDEX)
        val_progress_bar = tqdm(city_val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Cityscapes Val]", unit="batch")

        with torch.no_grad():
            for images_val, labels_val in val_progress_bar:
                images_val = images_val.to(DEVICE)
                outputs_val = model(images_val) 
                preds_val = torch.argmax(outputs_val, dim=1)
                val_conf_mat.update(preds_val.cpu(), labels_val.cpu())
        
        current_cityscapes_miou, _ = val_conf_mat.compute_iou()
        print(f"Epoch {epoch+1} Cityscapes Validation mIoU: {current_cityscapes_miou:.2f}%")

        is_best = current_cityscapes_miou > BEST_CITYSCAPES_MIOU
        if is_best:
            BEST_CITYSCAPES_MIOU = current_cityscapes_miou
            save_checkpoint(model, optimizer, epoch, BEST_CITYSCAPES_MIOU, run_checkpoint_dir, 
                            filename=f"{args.run_name}_best_on_city.pth.tar") # Use dynamic run_name
            print(f"  ** New best Cityscapes mIoU: {BEST_CITYSCAPES_MIOU:.2f}%. Best checkpoint saved.")
        
        save_checkpoint(model, optimizer, epoch, current_cityscapes_miou, run_checkpoint_dir, 
                        filename=f"{args.run_name}_epoch_{epoch+1}.pth.tar")
        save_checkpoint(model, optimizer, epoch, current_cityscapes_miou, run_checkpoint_dir, 
                        filename=f"{args.run_name}_latest.pth.tar")

    total_training_time = time.time() - start_time
    print(f"\nGTA5 Training with Augmentations finished in {total_training_time / 3600:.2f} hours.")
    print(f"Best mIoU on Cityscapes validation: {BEST_CITYSCAPES_MIOU:.2f}%")
    print(f"Checkpoints saved in {run_checkpoint_dir}")


# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args() 
    main(cmd_args)
