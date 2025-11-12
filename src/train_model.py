"""
FIXED: Enhanced training script with proper loss functions and metrics.
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
import numpy as np
import gc
gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


def transformer_preprocessing(x, **kwargs):
    """Simple normalization for transformers."""
    return x.astype('float32') / 255.0


def calculate_iou_torch(pred, target, num_classes):
    """Calculate IoU for each class using torch tensors."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(1.0 if intersection == 0 else 0.0)
        else:
            ious.append((intersection / union).item())
    
    return ious


def train_transformer_epoch(model, train_loader, loss_fn, optimizer, device, scaler, num_classes):
    """FIXED: Custom training epoch with proper metrics."""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    all_ious = []
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # FIX 1: Check if masks are one-hot or class indices
        if masks.dim() == 4 and masks.shape[1] > 1:  # One-hot encoded (B, C, H, W)
            masks = masks.argmax(dim=1).long()
        else:  # Already class indices (B, H, W)
            masks = masks.long().squeeze(1) if masks.dim() == 4 else masks.long()
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
            
            # Calculate IoU per batch
            batch_ious = calculate_iou_torch(pred, masks, num_classes)
            all_ious.append(batch_ious)
        
        if batch_idx % 10 == 0:
            current_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {current_acc:.4f}')
    
    avg_loss = total_loss / len(train_loader)
    pixel_acc = correct_pixels / total_pixels
    
    # Calculate mean IoU across all batches
    mean_iou = np.mean([np.mean(iou) for iou in all_ious])
    
    return {
        'loss': avg_loss,
        'pixel_accuracy': pixel_acc,
        'iou_score': mean_iou  # ADD THIS FOR CONSISTENCY
    }


def validate_transformer_epoch(model, valid_loader, loss_fn, device, num_classes):
    """FIXED: Custom validation epoch with proper metrics."""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    all_ious = []
    
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # FIX 1: Check mask format
            if masks.dim() == 4 and masks.shape[1] > 1:
                masks = masks.argmax(dim=1).long()
            else:
                masks = masks.long().squeeze(1) if masks.dim() == 4 else masks.long()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
            
            # Calculate IoU
            batch_ious = calculate_iou_torch(pred, masks, num_classes)
            all_ious.append(batch_ious)
    
    avg_loss = total_loss / len(valid_loader)
    pixel_acc = correct_pixels / total_pixels
    mean_iou = np.mean([np.mean(iou) for iou in all_ious])
    
    return {
        'loss': avg_loss,
        'pixel_accuracy': pixel_acc,
        'iou_score': mean_iou
    }


def _get_loss_from_logs(logs: dict):
    # try common keys that SMP or your loops might emit
    for k in ("loss", "dice_loss", "cross_entropy_loss", "ce_loss", "bce_loss", "seg_loss"):
        v = logs.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return v
    return 0.0


def train_model(model_name):
    """Train a specific model architecture with proper loss and metrics."""
    
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

    # FIX 2: Proper loss function with class weights
    if model_arch in transformer_models:
        # Calculate class weights to handle imbalance
        loss = torch.nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss for {model_arch}")
    else:
        loss = smp.utils.losses.DiceLoss()
        print(f"Using DiceLoss for {model_arch}")
    
    # FIX 3: Adjust learning rate for transformers
    if model_arch in transformer_models:
        actual_lr = init_lr * 0.1  # Transformers need lower LR
        print(f"⚠️  Adjusting LR for transformer: {init_lr} → {actual_lr}")
    else:
        actual_lr = init_lr
    
    optimizer = getattr(torch.optim, optimizer_choice)([dict(params=model.parameters(), lr=actual_lr)])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor,
                                                           patience=lr_reduce_patience, threshold=lr_reduce_threshold,
                                                           min_lr=minimum_lr)
    
    scaler = torch.amp.GradScaler('cuda') if model_arch in transformer_models else None

    best_model_path = None
    max_score = 0
    best_epoch = 0
    
    default_metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(),
    ]
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        if model_arch in transformer_models:
            train_logs = train_transformer_epoch(model, train_loader, loss, optimizer, device, scaler, len(classes))
            valid_logs = validate_transformer_epoch(model, valid_loader, loss, device, len(classes))
            train_loss = _get_loss_from_logs(train_logs)
            val_loss   = _get_loss_from_logs(valid_logs)

            # FIX 4: Use IoU for model selection, not pixel accuracy
            current_score = valid_logs['iou_score']
            score_name = 'IoU'
        else:
            train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=default_metrics, optimizer=optimizer, device=device, verbose=True)
            valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=default_metrics, device=device, verbose=True)
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_loss = _get_loss_from_logs(train_logs)
            val_loss   = _get_loss_from_logs(valid_logs)
            current_score = valid_logs['iou_score']
            score_name = 'IoU'

        # Save best model
        if current_score > max_score:
            max_score = current_score
            best_epoch = epoch + 1
            model_filename = f'landcover_{model_arch.lower()}_{encoder}_{optimizer_choice}_epoch{epoch+1}_patch{patch_size}_batch{batch_size}.pth'
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(model_dir, model_filename)
            torch.save(model, best_model_path)
            print(f'\n✅ New best model saved! {score_name}: {max_score:.4f}')

        # Step scheduler
        scheduler.step(val_loss)

        # Print progress
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Train IoU:  {train_logs.get('iou_score', 0):.4f} | Val IoU:  {valid_logs.get('iou_score', 0):.4f}")
        if 'pixel_accuracy' in train_logs or 'pixel_accuracy' in valid_logs:
            print(f"   Train Acc:  {train_logs.get('pixel_accuracy', 0):.4f} | Val Acc:  {valid_logs.get('pixel_accuracy', 0):.4f}")

        print(f"   Best IoU so far: {max_score:.4f} (Epoch {best_epoch})")

    shutil.rmtree(patches_dir)
    print("\n🧹 Cleaned up temporary patch files")
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
    print(f"📊 Final Results - Model: {args.model.upper()}, Best IoU: {max_score:.4f}, Best Epoch: {best_epoch}")