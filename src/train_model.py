"""
Enhanced training script that supports different model architectures via command line arguments.
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

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Memory Allocated (MB):", torch.cuda.memory_allocated(0)/1024**2)
    print("Max Memory Allocated (MB):", torch.cuda.max_memory_allocated(0)/1024**2)
    print("Memory Cached (MB):", torch.cuda.memory_reserved(0)/1024**2)


def train_transformer_epoch(model, train_loader, loss_fn, optimizer, device):
    """Custom training epoch for transformer models."""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Convert masks to class indices for CrossEntropyLoss
        masks = masks.argmax(dim=1).long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate pixel accuracy
        pred = outputs.argmax(dim=1)
        correct_pixels += (pred == masks).sum().item()
        total_pixels += masks.numel()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    pixel_acc = correct_pixels / total_pixels
    
    return {'loss': avg_loss, 'pixel_accuracy': pixel_acc}


def validate_transformer_epoch(model, valid_loader, loss_fn, device):
    """Custom validation epoch for transformer models."""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Convert masks to class indices
            masks = masks.argmax(dim=1).long()
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate pixel accuracy
            pred = outputs.argmax(dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(valid_loader)
    pixel_acc = correct_pixels / total_pixels
    
    return {'loss': avg_loss, 'pixel_accuracy': pixel_acc}


def train_model(model_name):
    """
    Train a specific model architecture.
    
    Args:
        model_name (str): Name of the model configuration to use
    """
    
    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_model_config(__file__, Constants, model_name)

    # get the required variable values from config
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']
    discard_rate = slice_config['vars']['discard_rate']
    batch_size = slice_config['vars']['batch_size']
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
    device = slice_config['vars']['device']

    # Get preprocessing function based on model type
    transformer_models = ['SegFormer', 'ViTSeg', 'HybridCNNTransformer', 'MaskDINO', 'SCTNet', 'EnhancedDeepLabV3Plus']
    
    if model_arch in transformer_models:
        # For transformers, we'll use a simple identity function and handle normalization in the model
        def transformer_preprocessing(x, **kwargs):
            # Just normalize to [0, 1] - further normalization done in model
            return x.astype('float32') / 255.0
        preprocessing_fn = transformer_preprocessing
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # get the log file dir from config
    log_dir = ROOT / slice_config['dirs']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / slice_config['vars']['train_log_name']
    log_path = log_path.as_posix()
    
    # initialize the logger
    logger = custom_logger(f"Land Cover Segmentation {model_arch} Train", log_path, log_level)

    # get directory paths
    train_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir']
    train_dir = train_dir.as_posix()

    img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['image_dir']
    img_dir = img_dir.as_posix()

    mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['mask_dir']
    mask_dir = mask_dir.as_posix()

    model_dir = ROOT / slice_config['dirs']['model_dir']
    model_dir = model_dir.as_posix()

    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    print(f"\n🚀 Starting training for {model_arch} architecture...")
    logger.info(f"Starting training for {model_arch} architecture...")

    # create the directory to save the patches of images and masks
    patches_dir = os.path.join(train_dir, f"patches_{patch_size}_{model_name}")
    patches_img_dir = os.path.join(patches_dir, "images")
    os.makedirs(patches_img_dir, exist_ok=True)
    patches_mask_dir = os.path.join(patches_dir, "masks")
    os.makedirs(patches_mask_dir, exist_ok=True)

    try:
        print("\nDividing images into patches...")
        patching(img_dir, patches_img_dir, file_type, patch_size)
        print("\nDivided images into patches successfully!")
        logger.info("Divided images into patches successfully!")
    except Exception as e:
        logger.error("Failed to divide images into patches!")
        raise e

    try:
        print("\nDividing masks into patches...")
        patching(mask_dir, patches_mask_dir, file_type, patch_size)
        print("\nDivided masks into patches successfully!")
        logger.info("Divided masks into patches successfully!")
    except Exception as e:
        logger.error("Failed to divide masks into patches!")
        raise e

    try:
        print(f"\nDiscarding useless patches where background covers more than {discard_rate*100}% of the area...")
        discard_useless_patches(patches_img_dir, patches_mask_dir, discard_rate)
        print("\nDiscarded unused patches successfully!")
        logger.info("Discarded unused patches successfully!")
    except Exception as e:
        logger.error("Failed to discard unused patches!")
        raise e

    output_folder = os.path.join(patches_dir, "train_val_test")
    os.makedirs(output_folder, exist_ok=True)

    try:
        print("\nSplitting training and validation data...")
        splitfolders.ratio(patches_dir, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None, move=False)
        print("\nTraining and validation data split successfully!")
        logger.info("Training and validation data split successfully!")
    except Exception as e:
        logger.error("Failed to split training and validation data!")
        raise e

    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")
    x_train_dir = os.path.join(train_dir, "images")
    y_train_dir = os.path.join(train_dir, "masks")
    x_val_dir = os.path.join(val_dir, "images")
    y_val_dir = os.path.join(val_dir, "masks")

    try:
        # Check if it's a transformer model
        
        if model_arch in transformer_models:
            print(f"\nBuilding {model_arch} transformer model with {encoder} encoder...")
            model = get_transformer_model(
                model_arch=model_arch,
                num_classes=len(classes),
                encoder=encoder,
                encoder_weights=encoder_weights
            )
            print(f"\nBuilt the {model_arch} transformer model successfully!")
            logger.info(f"Built the {model_arch} transformer model successfully!")
        else:
            # Traditional SMP models
            print(f"\nBuilding {model_arch} model with {encoder} encoder...")
            smp_model = getattr(smp, model_arch)
            model = smp_model(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=len(classes),
                activation=activation,
            )
            print(f"\nBuilt the {model_arch} model successfully!")
            logger.info(f"Built the {model_arch} model successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        logger.info(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to build the {model_arch} model!")
        raise e

    try:
        train_dataset = SegmentationDataset(
            x_train_dir,
            y_train_dir,
            all_classes=all_classes,
            classes=classes,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        val_dataset = SegmentationDataset(
            x_val_dir,
            y_val_dir,
            all_classes=all_classes,
            classes=classes,
            preprocessing=get_preprocessing(preprocessing_fn),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print(f"\nDataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    except Exception as e:
        logger.error("Failed to initialize training and validation datasets and dataloaders!")
        raise e

    try:
        # Choose loss function based on model type
        if model_arch in transformer_models:
            loss = torch.nn.CrossEntropyLoss()
            print("\nUsing CrossEntropyLoss for transformer model!")
        else:
            loss = smp.utils.losses.DiceLoss()
        
        metrics = [smp.utils.metrics.IoU(threshold=0.5)] if model_arch not in transformer_models else []
        print("\nInitialized the loss and evaluation metrics!")
        logger.info("Initialized the loss and evaluation metrics!")
    except Exception as e:
        logger.error("Failed to initialize loss and evaluation metrics")
        raise e

    try:
        torch_optimizer = getattr(torch.optim, optimizer_choice)
        optimizer = torch_optimizer([dict(params=model.parameters(), lr=init_lr)])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=lr_reduce_factor, 
            patience=lr_reduce_patience, 
            threshold=lr_reduce_threshold, 
            min_lr=minimum_lr
        )
        print(f"\nInitialized {optimizer_choice} optimizer with learning rate {init_lr}!")
        logger.info(f"Initialized {optimizer_choice} optimizer with learning rate {init_lr}!")
    except Exception as e:
        logger.error("Failed to initialize the optimizer!")
        raise e

    try:
        # Create epoch runners based on model type
        if model_arch in transformer_models:
            # Custom training loop for transformers
            print("\nUsing custom training loop for transformer models!")
            logger.info("Using custom training loop for transformer models!")
            train_epoch = None
            valid_epoch = None
        else:
            # SMP epoch runners for traditional models
            train_epoch = smp.utils.train.TrainEpoch(
                model,
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                device=device,
                verbose=True,
            )
            valid_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=metrics,
                device=device,
                verbose=True,
            )
            print("\nInitialized SMP epoch runners!")
            logger.info("Initialized SMP epoch runners!")
    except Exception as e:
        logger.error("Failed to initialize epoch runners!")
        raise e

    try:
        print(f"\nStarting {model_arch} model training for {epochs} epochs...")
        logger.info(f"Starting {model_arch} model training for {epochs} epochs...")
        
        max_score = 0
        best_epoch = 0
        
        for i in range(0, epochs):
            print(f'\n📊 Epoch: {i+1}/{epochs}')
            
            if model_arch in transformer_models:
                # Custom training loop for transformers
                train_logs = train_transformer_epoch(model, train_loader, loss, optimizer, device)
                valid_logs = validate_transformer_epoch(model, valid_loader, loss, device)
                
                # Use pixel accuracy as the main metric for transformers
                current_score = valid_logs['pixel_accuracy']
                score_name = 'pixel_accuracy'
                loss_key = 'loss'
            else:
                # SMP training loop for traditional models
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)
                
                current_score = valid_logs['iou_score']
                score_name = 'iou_score'
                loss_key = 'dice_loss'

            # Save best model
            if max_score < current_score:
                max_score = current_score
                best_epoch = i + 1
                model_filename = f'landcover_{model_arch.lower()}_{encoder}_{optimizer_choice}_epochs{i+1}_patch{patch_size}_batch{batch_size}.pth'
                torch.save(model, f'{model_dir}/{model_filename}')
                print(f'✅ New best model saved! {score_name}: {max_score:.4f}')
                logger.info(f'New best model saved! {score_name}: {max_score:.4f}')

            # Learning rate scheduling
            if model_arch in transformer_models:
                scheduler.step(valid_logs['loss'])
            else:
                scheduler.step(valid_logs['dice_loss'])
            
            # Print progress
            if model_arch in transformer_models:
                print(f"📈 Train Loss: {train_logs['loss']:.4f}, Val Loss: {valid_logs['loss']:.4f}")
                print(f"📈 Train Pixel Acc: {train_logs['pixel_accuracy']:.4f}, Val Pixel Acc: {valid_logs['pixel_accuracy']:.4f}")
            else:
                print(f"📈 Train Loss: {train_logs['dice_loss']:.4f}, Val Loss: {valid_logs['dice_loss']:.4f}")
                print(f"📈 Train IoU: {train_logs['iou_score']:.4f}, Val IoU: {valid_logs['iou_score']:.4f}")
            
        print(f"\n🎉 {model_arch} model training finished!")
        print(f"🏆 Best {score_name}: {max_score:.4f} achieved at epoch {best_epoch}")
        logger.info(f"{model_arch} model training finished!")
        logger.info(f"Best {score_name}: {max_score:.4f} achieved at epoch {best_epoch}")
        
    except Exception as e:
        logger.error(f"Failed to train the {model_arch} model!")
        raise e

    # Clean up patches directory
    shutil.rmtree(patches_dir)
    print("🧹 Cleaned up temporary patch files")

    return max_score, best_epoch


if __name__ == "__main__":
    # Get available model configs dynamically
    from pathlib import Path
    config_dir = Path(__file__).parent.parent / "config" / "models"
    available_configs = [f.stem.replace('_config', '') for f in config_dir.glob('*_config.yaml')]
    
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=available_configs,
                       help=f'Model architecture to train. Available: {", ".join(available_configs)}')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting training for {args.model.upper()} architecture...")
    
    try:
        max_score, best_epoch = train_model(args.model)
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   - Model: {args.model.upper()}")
        print(f"   - Best IoU Score: {max_score:.4f}")
        print(f"   - Best Epoch: {best_epoch}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise e
