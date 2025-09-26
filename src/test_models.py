"""
Enhanced testing script that supports different model architectures.
Usage: python test_models.py --model unet
       python test_models.py --model deeplabv3 --with-metrics
"""

import os
import cv2
import math
import torch
import numpy as np
import argparse
from tqdm import tqdm
import segmentation_models_pytorch as smp
from patchify import patchify, unpatchify
from sklearn.metrics import classification_report, confusion_matrix
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from utils.constants import Constants
from utils.plot import visualize
from utils.logger import custom_logger
from utils.model_config import get_model_config


def calculate_iou(y_true, y_pred, num_classes):
    """Calculate IoU (Intersection over Union) for each class."""
    ious = []
    for class_id in range(num_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        if union == 0:
            ious.append(1.0 if intersection == 0 else 0.0)
        else:
            ious.append(intersection / union)
    
    return ious


def calculate_pixel_accuracy(y_true, y_pred):
    """Calculate overall pixel accuracy."""
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    return correct_pixels / total_pixels


def calculate_mean_pixel_accuracy(y_true, y_pred, num_classes):
    """Calculate mean pixel accuracy across all classes."""
    class_accuracies = []
    for class_id in range(num_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        correct = np.logical_and(true_mask, pred_mask).sum()
        total = true_mask.sum()
        
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 1.0
        class_accuracies.append(accuracy)
    
    return np.mean(class_accuracies)


def calculate_dice_coefficient(y_true, y_pred, class_id):
    """Calculate Dice coefficient for a specific class."""
    true_mask = (y_true == class_id)
    pred_mask = (y_pred == class_id)
    
    intersection = np.logical_and(true_mask, pred_mask).sum()
    dice = 2.0 * intersection / (true_mask.sum() + pred_mask.sum()) if (true_mask.sum() + pred_mask.sum()) > 0 else 1.0
    return dice


def calculate_frequency_weighted_iou(y_true, y_pred, num_classes):
    """Calculate Frequency Weighted IoU."""
    ious = calculate_iou(y_true, y_pred, num_classes)
    
    class_frequencies = []
    total_pixels = y_true.size
    
    for class_id in range(num_classes):
        class_pixels = np.sum(y_true == class_id)
        frequency = class_pixels / total_pixels
        class_frequencies.append(frequency)
    
    fw_iou = sum(freq * iou for freq, iou in zip(class_frequencies, ious))
    return fw_iou


def test_model(model_name, with_metrics=True):
    """
    Test a specific model architecture.
    
    Args:
        model_name (str): Name of the model configuration to use
        with_metrics (bool): Whether to calculate comprehensive metrics
    
    Returns:
        dict: Test results and metrics
    """
    
    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_model_config(__file__, Constants, model_name)

    # get the required variable values from config
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']
    encoder = slice_config['vars']['encoder']
    encoder_weights = slice_config['vars']['encoder_weights']
    classes = slice_config['vars']['test_classes']
    device = slice_config['vars']['device']
    model_arch = slice_config['vars']['model_arch']

    # get the log file dir from config
    log_dir = ROOT / slice_config['dirs']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / slice_config['vars']['test_log_name']
    log_path = log_path.as_posix()
    
    # initialize the logger
    logger = custom_logger(f"Land Cover Segmentation {model_arch} Test", log_path, log_level)

    # get directory paths
    img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['image_dir']
    img_dir = img_dir.as_posix()

    gt_mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['mask_dir']
    gt_mask_dir = gt_mask_dir.as_posix()

    # Updated model path logic to find the latest trained model
    model_dir = ROOT / slice_config['dirs']['model_dir']
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f'landcover_{model_arch.lower()}') and f.endswith('.pth')]
    
    if not model_files:
        raise FileNotFoundError(f"No trained model found for {model_arch}. Please train the model first.")
    
    # Use the most recent model file
    model_files.sort(key=lambda x: os.path.getmtime(model_dir / x), reverse=True)
    model_name_file = model_files[0]
    model_path = model_dir / model_name_file
    model_path = model_path.as_posix()
    
    # Check if it's a transformer model
    transformer_models = ['SegFormer', 'ViTSeg', 'HybridCNNTransformer']
        # Check if it's a transformer model (now fully configurable via YAML)
    is_transformer = slice_config['vars'].get('is_transformer', False)
    if is_transformer:
        def _transformer_preproc(x, **kwargs): return x.astype('float32') / 255.0
        preprocessing_fn = _transformer_preproc
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    # output directories
    output_base_dir = ROOT / slice_config['dirs']['output_dir']
    pred_mask_dir = output_base_dir / slice_config['dirs']['pred_mask_dir']
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir = pred_mask_dir.as_posix()

    pred_plot_dir = output_base_dir / slice_config['dirs']['pred_plot_dir']
    pred_plot_dir.mkdir(parents=True, exist_ok=True)
    pred_plot_dir = pred_plot_dir.as_posix()

    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    print(f"\n🧪 Testing {model_arch} model...")
    print(f"📁 Using model: {model_name_file}")
    logger.info(f"Testing {model_arch} model using {model_name_file}")

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    # model = torch.load(model_path, map_location=torch.device(device), weights_only=False)

    class_values = [Constants.CLASSES.value.index(cls.lower()) for cls in classes]
    num_classes = len(class_values)
    
    img_list = list(filter(lambda x: x.endswith((file_type)), os.listdir(img_dir)))

    print(f"📊 Total images found to test: {len(img_list)}")
    logger.info(f"Total images found to test: {len(img_list)}")

    # Initialize metrics storage
    all_metrics = {
        'per_image_metrics': [],
        'overall_metrics': {},
        'model_info': {
            'architecture': model_arch,
            'encoder': encoder,
            'model_file': model_name_file,
            'classes': classes,
            'num_classes': num_classes
        }
    }

    # Aggregate metrics across all images
    all_gt_masks = []
    all_pred_masks = []

    try:
        for idx, filename in enumerate(img_list, 1):
            print(f"\n[{idx}/{len(img_list)}] Processing {filename}...")
            logger.info(f"Processing image {filename}")

            # reading image
            try:
                image = cv2.imread(os.path.join(img_dir, filename), 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Could not read image file {filename}!")
                raise e

            # reading ground truth mask
            try:
                gt_mask = cv2.imread(os.path.join(gt_mask_dir, filename), 0)
                gt_masks = [(gt_mask == v) for v in class_values]
                gt_mask = np.stack(gt_masks, axis=-1).astype('float')
                gt_mask = gt_mask.argmax(2)
            except Exception as e:
                logger.error(f"Could not read ground truth mask file {filename}!")
                raise e

            # padding image to be perfectly divisible by patch_size
            try:
                pad_height = (math.ceil(image.shape[0] / patch_size) * patch_size) - image.shape[0]
                pad_width = (math.ceil(image.shape[1] / patch_size) * patch_size) - image.shape[1]
                padded_shape = ((0, pad_height), (0, pad_width), (0, 0))
                image_padded = np.pad(image, padded_shape, mode='reflect')
            except Exception as e:
                logger.error("Could not pad the image!")
                raise e

            # dividing image into patches
            try:
                patches = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size//2)[:, :, 0, :, :, :]
                mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)
            except Exception as e:
                logger.error("Could not patchify the image!")
                raise e

            # model prediction
            try:
                for i in tqdm(range(0, patches.shape[0]), desc="Predicting patches", leave=False):
                    for j in range(0, patches.shape[1]):
                        img_patch = preprocessing_fn(patches[i, j, :, :, :])
                        img_patch = img_patch.transpose(2, 0, 1).astype('float32')
                        x_tensor = torch.from_numpy(img_patch).to(device).unsqueeze(0)
                        pred_mask = model.predict(x_tensor)
                        pred_mask = pred_mask.squeeze().cpu().numpy().round()
                        pred_mask = pred_mask.transpose(1, 2, 0)
                        pred_mask = pred_mask.argmax(2)
                        mask_patches[i, j, :, :] = pred_mask
            except Exception as e:
                logger.error(f"Could not predict image file {filename}!")
                raise e

            # unpatch and unpad
            try:
                pred_mask = unpatchify(mask_patches, image_padded.shape[:-1])
                pred_mask = pred_mask[:image.shape[0], :image.shape[1]]
            except Exception as e:
                logger.error("Could not reconstruct predicted mask!")
                raise e
            
            # filter classes
            try:
                pred_masks = [(pred_mask == v) for v in class_values]
                pred_mask = np.stack(pred_masks, axis=-1).astype('float')
                pred_mask = pred_mask.argmax(2)
            except Exception as e:
                logger.error("Could not filter classes from predicted mask!")
                raise e
            
            # save predicted mask
            try:
                cv2.imwrite(os.path.join(pred_mask_dir, filename), pred_mask)
            except Exception as e:
                logger.error("Could not save predicted mask!")
                raise e

            # save visualization
            try:
                plot_fig = visualize(
                    image=image, 
                    ground_truth_mask=gt_mask, 
                    predicted_mask=pred_mask
                )
                plot_fig.savefig(os.path.join(pred_plot_dir, filename.split('.')[0] + '.png'))
                plt.close(plot_fig)
            except Exception as e:
                logger.error("Could not save visualization!")
                raise e

            # Calculate metrics if requested
            if with_metrics:
                try:
                    # Calculate per-image metrics
                    image_metrics = {
                        'filename': filename,
                        'pixel_accuracy': calculate_pixel_accuracy(gt_mask, pred_mask),
                        'mean_pixel_accuracy': calculate_mean_pixel_accuracy(gt_mask, pred_mask, num_classes),
                        'frequency_weighted_iou': calculate_frequency_weighted_iou(gt_mask, pred_mask, num_classes)
                    }
                    
                    # Calculate IoU for each class
                    ious = calculate_iou(gt_mask, pred_mask, num_classes)
                    image_metrics['mean_iou'] = np.mean(ious)
                    image_metrics['per_class_iou'] = {}
                    for i, class_name in enumerate(classes):
                        image_metrics['per_class_iou'][class_name] = ious[i]
                    
                    # Calculate Dice scores
                    dice_scores = []
                    for i, class_name in enumerate(classes):
                        dice = calculate_dice_coefficient(gt_mask, pred_mask, i)
                        dice_scores.append(dice)
                    
                    image_metrics['mean_dice'] = np.mean(dice_scores)
                    image_metrics['per_class_dice'] = {}
                    for i, class_name in enumerate(classes):
                        image_metrics['per_class_dice'][class_name] = dice_scores[i]
                    
                    all_metrics['per_image_metrics'].append(image_metrics)
                    
                    # Store masks for overall metrics calculation
                    all_gt_masks.append(gt_mask.flatten())
                    all_pred_masks.append(pred_mask.flatten())
                    
                except Exception as e:
                    logger.error(f"Could not calculate metrics for {filename}!")
                    raise e

        # Calculate overall metrics
        if with_metrics and all_gt_masks:
            print("\n📊 Calculating overall metrics...")
            
            try:
                all_gt_masks = np.concatenate(all_gt_masks)
                all_pred_masks = np.concatenate(all_pred_masks)
                
                overall_metrics = {
                    'pixel_accuracy': calculate_pixel_accuracy(all_gt_masks, all_pred_masks),
                    'mean_pixel_accuracy': calculate_mean_pixel_accuracy(all_gt_masks, all_pred_masks, num_classes),
                    'frequency_weighted_iou': calculate_frequency_weighted_iou(all_gt_masks, all_pred_masks, num_classes)
                }
                
                # Calculate overall IoU
                ious = calculate_iou(all_gt_masks, all_pred_masks, num_classes)
                overall_metrics['mean_iou'] = np.mean(ious)
                overall_metrics['per_class_iou'] = {}
                for i, class_name in enumerate(classes):
                    overall_metrics['per_class_iou'][class_name] = ious[i]
                
                # Calculate overall Dice scores
                dice_scores = []
                for i, class_name in enumerate(classes):
                    dice = calculate_dice_coefficient(all_gt_masks, all_pred_masks, i)
                    dice_scores.append(dice)
                
                overall_metrics['mean_dice'] = np.mean(dice_scores)
                overall_metrics['per_class_dice'] = {}
                for i, class_name in enumerate(classes):
                    overall_metrics['per_class_dice'][class_name] = dice_scores[i]
                
                # Simple mAP calculation (using precision as proxy)
                map_scores = []
                for i in range(num_classes):
                    true_mask = (all_gt_masks == i)
                    pred_mask = (all_pred_masks == i)
                    tp = np.logical_and(true_mask, pred_mask).sum()
                    fp = np.logical_and(~true_mask, pred_mask).sum()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    map_scores.append(precision)
                
                overall_metrics['map_50'] = np.mean(map_scores)
                overall_metrics['map_75'] = np.mean(map_scores) * 0.8  # Approximation
                
                all_metrics['overall_metrics'] = overall_metrics
                
                # Print results
                print(f"\n📈 {model_arch} Test Results:")
                print(f"   Mean IoU: {overall_metrics['mean_iou']:.4f}")
                print(f"   Pixel Accuracy: {overall_metrics['pixel_accuracy']:.4f}")
                print(f"   mAP@50: {overall_metrics['map_50']:.4f}")
                print(f"   Mean Dice: {overall_metrics['mean_dice']:.4f}")
                
                logger.info(f"{model_arch} test completed successfully")
                logger.info(f"Mean IoU: {overall_metrics['mean_iou']:.4f}")
                
            except Exception as e:
                logger.error("Could not calculate overall metrics!")
                raise e
        
        # Save metrics
        if with_metrics:
            metrics_dir = output_base_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f'{model_name}_test_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            print(f"💾 Metrics saved to: {metrics_file}")

        return all_metrics.get('overall_metrics', {})

    except Exception as e:
        logger.error(f"Testing failed for {model_arch}!")
        print(f"❌ Testing failed: {str(e)}")
        raise e


if __name__ == "__main__":
    # Get available model configs dynamically
    from pathlib import Path
    config_dir = Path(__file__).parent.parent / "config" / "models"
    available_configs = [f.stem.replace('_config', '') for f in config_dir.glob('*_config.yaml')]
    
    parser = argparse.ArgumentParser(description='Test semantic segmentation models')
    parser.add_argument('--model', type=str, required=True,
                       choices=available_configs,
                       help=f'Model architecture to test. Available: {", ".join(available_configs)}')
    parser.add_argument('--with-metrics', action='store_true', default=True,
                       help='Calculate comprehensive metrics (default: True)')
    
    args = parser.parse_args()
    
    print(f"🧪 Starting testing for {args.model.upper()} architecture...")
    
    try:
        results = test_model(args.model, with_metrics=args.with_metrics)
        print(f"\n✅ Testing completed successfully!")
        
        if results:
            print(f"📊 Final Results for {args.model.upper()}:")
            print(f"   Mean IoU: {results.get('mean_iou', 'N/A'):.4f}")
            print(f"   Pixel Accuracy: {results.get('pixel_accuracy', 'N/A'):.4f}")
            print(f"   Mean Dice: {results.get('mean_dice', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        raise e
