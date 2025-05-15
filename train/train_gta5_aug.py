# train/train_gta5_aug.py
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
from datasets.gta5 import GTA5 # Updated GTA5 class
from datasets.cityscapes import CityScapes 
from models.bisenet.build_bisenet import BiSeNet 
from utils.lr_scheduler import lr_poly 
from utils.metrics import ConfusionMatrix

# --- Default Configurations ---
DEFAULT_GTA5_ROOT = "/kaggle/input/datasetcityscapes/GTA5/GTA5/" 
DEFAULT_CITYSCAPES_ROOT = "/kaggle/input/datasetcityscapes/Cityscapes/Cityscapes/Cityspaces/" 
DEFAULT_CHECKPOINT_DIR = "/kaggle/working/checkpoints_bisenet_gta5_aug" 
DEFAULT_RUN_NAME = "bisenet_gta5_aug_default" 
DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_GTA5_TRAIN_SIZE = (720, 1280) 
DEFAULT_CITYSCAPES_EVAL_SIZE = (512, 1024) 
DEFAULT_NUM_EPOCHS = 50 
DEFAULT_BATCH_SIZE_GTA5 = 4 
DEFAULT_BATCH_SIZE_CITY_VAL = 8 
DEFAULT_LEARNING_RATE = 2.5e-2 
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
AUX_LOSS_WEIGHT = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 100 
BEST_CITYSCAPES_MIOU = 0.0 

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="BiSeNet Training on GTA5 with Augmentations, Validating on Cityscapes")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--gta5_bs', type=int, default=DEFAULT_BATCH_SIZE_GTA5, help='GTA5 training batch size')
    parser.add_argument('--city_val_bs', type=int, default=DEFAULT_BATCH_SIZE_CITY_VAL, help='Cityscapes validation batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Initial learning rate for backbone')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Total epochs')
    parser.add_argument('--gta5_root', type=str, default=DEFAULT_GTA5_ROOT, help='Path to GTA5 root')
    parser.add_argument('--cityscapes_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, help='Path to Cityscapes root for validation')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory for checkpoints')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for checkpoints (overrides dynamic naming if set)')

    # Augmentation Flags
    parser.add_argument('--aug_hflip', action='store_true', help='Enable Random Horizontal Flip')
    parser.add_argument('--aug_colorjitter', action='store_true', help='Enable Color Jitter')
    parser.add_argument('--aug_gblur', action='store_true', help='Enable Gaussian Blur')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability for applying enabled augmentations')
    
    args = parser.parse_args()

    if args.run_name is None:
        active_augs_names = []
        if args.aug_hflip: active_augs_names.append("hflip")
        if args.aug_colorjitter: active_augs_names.append("cjitter")
        if args.aug_gblur: active_augs_names.append("gblur")
        
        if not active_augs_names:
            args.run_name = "bisenet_gta5_no_aug_runX" 
        else:
            args.run_name = f"bisenet_gta5_aug_{'_'.join(active_augs_names)}_p{str(args.aug_prob).replace('.', '')}"
    return args

# --- Helper Function: save_checkpoint ---
def save_checkpoint(model, optimizer, epoch, best_miou, run_ckpt_dir, filename="checkpoint.pth.tar"):
    state = {
        'epoch': epoch + 1, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_cityscapes_miou': best_miou,
    }
    filepath = os.path.join(run_ckpt_dir, filename)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    torch.save(state, filepath)

# --- Main Training and Validation Function ---
def main(args): 
    global BEST_CITYSCAPES_MIOU 
    BEST_CITYSCAPES_MIOU = 0.0 

    print(f"Using device: {DEVICE}")
    print(f"Configuration: {args}")

    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {run_checkpoint_dir}")

    start_epoch = 0
    current_iteration = 0 

    # --- Datasets and DataLoaders ---
    if not os.path.isdir(args.gta5_root):
         print(f"ERROR: GTA5 root directory not found at {args.gta5_root}"); return
    print("Loading GTA5 training dataset with specified augmentations...")
    gta_train_dataset = GTA5(
        root_dir=args.gta5_root, 
        target_size=DEFAULT_GTA5_TRAIN_SIZE,
        use_horizontal_flip=args.aug_hflip,
        use_color_jitter=args.aug_colorjitter,
        use_gaussian_blur=args.aug_gblur,
        aug_probability=args.aug_prob
    )
    if len(gta_train_dataset) == 0: print("ERROR: GTA5 Training dataset loaded 0 samples."); return         
    gta_train_loader = DataLoader(gta_train_dataset, batch_size=args.gta5_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"GTA5 training dataset loaded with {len(gta_train_dataset)} samples.")

    if not os.path.isdir(args.cityscapes_root):
         print(f"ERROR: Cityscapes root for validation not found at {args.cityscapes_root}"); return
    print("Loading Cityscapes validation dataset...")
    city_val_dataset = CityScapes(root_dir=args.cityscapes_root, split='val', 
                                  transform_mode='val', target_size=DEFAULT_CITYSCAPES_EVAL_SIZE)
    if len(city_val_dataset) == 0: print("ERROR: Cityscapes Validation dataset loaded 0 samples."); return
    city_val_loader = DataLoader(city_val_dataset, batch_size=args.city_val_bs, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Cityscapes validation dataset loaded with {len(city_val_dataset)} samples.")
    
    max_iterations = args.epochs * len(gta_train_loader) 

    print("Initializing BiSeNet model (ResNet-18 backbone)...")
    model = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
    model.to(DEVICE)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)
    optimizer = optim.SGD(model.optim_parameters(args.lr), momentum=DEFAULT_MOMENTUM, weight_decay=DEFAULT_WEIGHT_DECAY)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
            
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items(): name = k[7:] if k.startswith('module.') else k; new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if 'optimizer' in checkpoint:
                try: optimizer.load_state_dict(checkpoint['optimizer'])
                except ValueError as e: print(f"=> Warning: Could not load optimizer state: {e}.")
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch']
            if 'best_cityscapes_miou' in checkpoint: BEST_CITYSCAPES_MIOU = checkpoint['best_cityscapes_miou']
            current_iteration = start_epoch * len(gta_train_loader)
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

            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            epoch_train_loss += total_loss.item(); current_iteration += 1 
            current_lr_display = optimizer.param_groups[0]['lr']
            train_progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", avg_loss=f"{epoch_train_loss/(i+1):.4f}", lr=f"{current_lr_display:.6f}")

        avg_epoch_train_loss = epoch_train_loss / len(gta_train_loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} GTA5 Training Completed. Avg Loss: {avg_epoch_train_loss:.4f}")

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
                            filename=f"{args.run_name}_best_on_city.pth.tar") 
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
if __name__ == "__main__": # Make sure this block is present
    cmd_args = parse_args() 
    main(cmd_args)
