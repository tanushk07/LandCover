"""
Enhanced training script with mixed precision for transformer models.
Supports different model architectures via command line arguments.
Usage: python train_model.py --model unet
       python train_model.py --model deeplabv3
       python train_model.py --model linknet
"""

import os
import shutil
import torch
import splitfolders
import argparse
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from utils.constants import Constants
from utils.logger import custom_logger
from utils.model_config import get_model_config
from utils.patching import patching, discard_useless_patches
from utils.preprocess import get_training_augmentation, get_preprocessing
from utils.dataset import SegmentationDataset
from utils.transformer_models import get_transformer_model
from torch.cuda.amp import autocast, GradScaler


if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Memory Allocated (MB):", torch.cuda.memory_allocated(0)/1024**2)
    print("Max Memory Allocated (MB):", torch.cuda.max_memory_allocated(0)/1024**2)
    print("Memory Cached (MB):", torch.cuda.memory_reserved(0)/1024**2)

def transformer_preprocessing(x, **kwargs):
    return x.astype('float32') / 255.0

def train_transformer_epoch(model, train_loader, loss_fn, optimizer, device, scaler):
    """Custom training epoch for transformer models using mixed precision."""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        masks = masks.argmax(dim=1).long()
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct_pixels += (pred == masks).sum().item()
        total_pixels += masks.numel()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    pixel_acc = correct_pixels / total_pixels
    
    return {'loss': avg_loss, 'pixel_accuracy': pixel_acc}


def validate_transformer_epoch(model, valid_loader, loss_fn, device):
    """Custom validation epoch for transformer models using mixed precision."""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.argmax(dim=1).long()
            
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(valid_loader)
    pixel_acc = correct_pixels / total_pixels
    
    return {'loss': avg_loss, 'pixel_accuracy': pixel_acc}


def train_model(model_name):
    """
    Train a specific model architecture with mixed precision for transformers.
    """
    
    ROOT, slice_config = get_model_config(__file__, Constants, model_name)

    # Extract config variables
    batch_size = slice_config['vars']['batch_size']
    patch_size = slice_config['vars']['patch_size']
    discard_rate = slice_config['vars']['discard_rate']
    model_arch = slice_config['vars']['model_arch']
    encoder = slice_config['vars']['encoder']
    encoder_weights = slice_config['vars']['encoder_weights']
    activation = slice_config['vars']['activation']
    optimizer_choice = slice_config['vars']['optimizer_choice']
    init_lr = slice_config['vars']['init_lr']
    lr_reduce_factor = slice_config['vars']['reduce_lr_by_factor']
    lr_reduce_patience = slice_config['vars']['patience_epochs_before_reducing_lr']
    lr_reduce_threshold = slice_config['vars']['lr_reduce_threshold']
    minimum_lr = slice_config['vars']['minimum_lr']
    epochs = slice_config['vars']['epochs']
    all_classes = slice_config['vars']['all_classes']
    classes = slice_config['vars']['train_classes']
    device = torch.device(slice_config['vars']['device'] if torch.cuda.is_available() and slice_config['vars']['device'].startswith('cuda') else 'cpu')

    transformer_models = ['SegFormer', 'ViTSeg', 'HybridCNNTransformer', 'MaskDINO', 'SCTNet', 'EnhancedDeepLabV3Plus']
    
    preprocessing_fn = transformer_preprocessing if model_arch in transformer_models else smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    log_dir = ROOT / slice_config['dirs']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / slice_config['vars']['train_log_name']).as_posix()
    logger = custom_logger(f"Land Cover Segmentation {model_arch} Train", log_path, slice_config['vars']['log_level'])

    # Directories
    train_dir = (ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir']).as_posix()
    img_dir = os.path.join(train_dir, slice_config['dirs']['image_dir'])
    mask_dir = os.path.join(train_dir, slice_config['dirs']['mask_dir'])
    model_dir = os.path.join(ROOT, slice_config['dirs']['model_dir'])
    os.makedirs(model_dir, exist_ok=True)

    # Create patches
    patches_dir = os.path.join(train_dir, f"patches_{patch_size}_{model_name}")
    patches_img_dir = os.path.join(patches_dir, "images")
    patches_mask_dir = os.path.join(patches_dir, "masks")
    os.makedirs(patches_img_dir, exist_ok=True)
    os.makedirs(patches_mask_dir, exist_ok=True)

    patching(img_dir, patches_img_dir, slice_config['vars']['file_type'], patch_size)
    patching(mask_dir, patches_mask_dir, slice_config['vars']['file_type'], patch_size)
    discard_useless_patches(patches_img_dir, patches_mask_dir, discard_rate)

    output_folder = os.path.join(patches_dir, "train_val_test")
    os.makedirs(output_folder, exist_ok=True)
    splitfolders.ratio(patches_dir, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None, move=False)

    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")
    x_train_dir, y_train_dir = os.path.join(train_dir, "images"), os.path.join(train_dir, "masks")
    x_val_dir, y_val_dir = os.path.join(val_dir, "images"), os.path.join(val_dir, "masks")

    # Build model
    if model_arch in transformer_models:
        model = get_transformer_model(model_arch=model_arch, num_classes=len(classes), encoder=encoder, encoder_weights=encoder_weights)
    else:
        smp_model = getattr(smp, model_arch)
        model = smp_model(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), activation=activation)
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters - Total: {total_params:,}")

    # Datasets and loaders
    train_dataset = SegmentationDataset(x_train_dir, y_train_dir, all_classes=all_classes, classes=classes,
                                        augmentation=get_training_augmentation(),
                                        preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = SegmentationDataset(x_val_dir, y_val_dir, all_classes=all_classes, classes=classes,
                                      preprocessing=get_preprocessing(preprocessing_fn))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    # Loss, optimizer, scheduler
    loss = torch.nn.CrossEntropyLoss() if model_arch in transformer_models else smp.utils.losses.DiceLoss()
    optimizer = getattr(torch.optim, optimizer_choice)([dict(params=model.parameters(), lr=init_lr)])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor,
                                                           patience=lr_reduce_patience, threshold=lr_reduce_threshold,
                                                           min_lr=minimum_lr)
    
    scaler = GradScaler() if model_arch in transformer_models else None

    best_model_path = None
    max_score = 0
    best_epoch = 0
    default_metrics = [
    smp.utils.metrics.IoU(threshold=0.5),    # mean IoU
    smp.utils.metrics.Fscore(),              # Dice score
    ]
    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        if model_arch in transformer_models:
            train_logs = train_transformer_epoch(model, train_loader, loss, optimizer, device, scaler)
            valid_logs = validate_transformer_epoch(model, valid_loader, loss, device)
            current_score = valid_logs['pixel_accuracy']
            score_name = 'pixel_accuracy'
        else:
            
            train_epoch = smp.utils.train.TrainEpoch(model, loss=loss,  metrics=default_metrics,optimizer=optimizer, device=device, verbose=True)
            valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=default_metrics, device=device, verbose=True)
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            current_score = valid_logs['iou_score']
            score_name = 'iou_score'

        # Save best model
        if current_score > max_score:
            max_score = current_score
            best_epoch = epoch + 1
            model_filename = f'landcover_{model_arch.lower()}_{encoder}_{optimizer_choice}_epoch{epoch+1}_patch{patch_size}_batch{batch_size}.pth'
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(model_dir, model_filename)
            torch.save(model, best_model_path)
            print(f'✅ New best model saved! {score_name}: {max_score:.4f}')

        # Step scheduler
        if model_arch in transformer_models:
            scheduler.step(valid_logs['loss'])
        else:
            scheduler.step(valid_logs['dice_loss'])

        # Print progress
        print(f"📈 Train Loss: {train_logs.get('loss', train_logs.get('dice_loss')):.4f}, Val Loss: {valid_logs.get('loss', valid_logs.get('dice_loss')):.4f}")
        if model_arch in transformer_models:
            print(f"📈 Train Pixel Acc: {train_logs['pixel_accuracy']:.4f}, Val Pixel Acc: {valid_logs['pixel_accuracy']:.4f}")
        else:
            print(f"📈 Train IoU: {train_logs['iou_score']:.4f}, Val IoU: {valid_logs['iou_score']:.4f}")

    shutil.rmtree(patches_dir)
    print("🧹 Cleaned up temporary patch files")
    return max_score, best_epoch


if __name__ == "__main__":
    from pathlib import Path
    config_dir = Path(__file__).parent.parent / "config" / "models"
    available_configs = [f.stem.replace('_config', '') for f in config_dir.glob('*_config.yaml')]
    
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    parser.add_argument('--model', type=str, required=True, choices=available_configs,
                        help=f'Model architecture to train. Available: {", ".join(available_configs)}')
    args = parser.parse_args()
    
    print(f"🚀 Starting training for {args.model.upper()} architecture...")
    max_score, best_epoch = train_model(args.model)
    print(f"\n✅ Training completed successfully!")
    print(f"📊 Final Results - Model: {args.model.upper()}, Best Score: {max_score:.4f}, Best Epoch: {best_epoch}")
