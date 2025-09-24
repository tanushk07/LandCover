#!/usr/bin/env python3

"""
Quick demo setup script for model architecture comparison.
This script prepares a small subset of data for fast experimentation.
"""

import os
import shutil
import yaml
from pathlib import Path

def setup_quick_demo():
    """Set up a quick demo with limited data and fast training configs."""
    
    print("🚀 Setting up quick demo for model architecture comparison...")
    
    # Define paths
    archive_dir = Path("archive")
    data_dir = Path("data")
    config_dir = Path("config/models")
    
    # Create data directories
    train_img_dir = data_dir / "train" / "images"
    train_mask_dir = data_dir / "train" / "masks"
    test_img_dir = data_dir / "test" / "images" 
    test_mask_dir = data_dir / "test" / "masks"
    
    # Clean and create directories
    for dir_path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Select subset of images for demo
    # Training: 4 images
    train_images = [
        "N-33-60-D-c-4-2.tif",
        "N-34-97-D-c-2-4.tif", 
        "M-33-20-D-c-4-2.tif",
        "M-34-65-D-c-4-2.tif"
    ]
    
    # Testing: 2 images (including the existing test images)
    test_images = [
        "N-33-60-D-c-4-2.tif",  # Same as existing
        "N-34-97-D-c-2-4.tif"   # Same as existing
    ]
    
    print(f"📁 Copying {len(train_images)} training images...")
    # Copy training data
    for img_name in train_images:
        src_img = archive_dir / "images" / img_name
        src_mask = archive_dir / "masks" / img_name
        dst_img = train_img_dir / img_name
        dst_mask = train_mask_dir / img_name
        
        if src_img.exists() and src_mask.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            print(f"  ✅ Copied {img_name}")
        else:
            print(f"  ❌ Missing {img_name}")
    
    print(f"📁 Copying {len(test_images)} test images...")
    # Copy test data  
    for img_name in test_images:
        src_img = archive_dir / "images" / img_name
        src_mask = archive_dir / "masks" / img_name
        dst_img = test_img_dir / img_name
        dst_mask = test_mask_dir / img_name
        
        if src_img.exists() and src_mask.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            print(f"  ✅ Copied {img_name}")
        else:
            print(f"  ❌ Missing {img_name}")
    
    # Create fast demo configs with reduced epochs
    print("⚙️  Creating fast demo configurations...")
    
    models_to_demo = ['unet', 'deeplabv3', 'linknet', 'fpn']
    
    for model_name in models_to_demo:
        original_config = config_dir / f"{model_name}_config.yaml"
        demo_config = config_dir / f"{model_name}_demo_config.yaml"
        
        if original_config.exists():
            # Load original config
            with open(original_config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Modify for fast demo
            config['vars']['epochs'] = 3  # Very few epochs for demo
            config['vars']['patch_size'] = 256  # Smaller patches = faster
            config['vars']['batch_size'] = 4   # Smaller batch = less memory
            config['dirs']['output_dir'] = f"output/{model_name}_demo"
            config['vars']['train_log_name'] = f"{model_name}_demo_train.log"
            config['vars']['test_log_name'] = f"{model_name}_demo_test.log"
            
            # Save demo config
            with open(demo_config, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"  ✅ Created {model_name}_demo_config.yaml")
        else:
            print(f"  ❌ Missing {original_config}")
    
    print("\n🎯 Demo setup complete!")
    print(f"📊 Training data: {len(train_images)} images")
    print(f"📊 Test data: {len(test_images)} images") 
    print(f"🏗️  Demo configs: {len(models_to_demo)} models")
    print(f"⚡ Fast settings: 3 epochs, 256px patches, batch size 4")
    
    return models_to_demo

def cleanup_demo():
    """Clean up demo configurations after completion."""
    config_dir = Path("config/models")
    demo_configs = list(config_dir.glob("*_demo_config.yaml"))
    
    for config_file in demo_configs:
        config_file.unlink()
        print(f"🧹 Removed {config_file.name}")

if __name__ == "__main__":
    models = setup_quick_demo()
    
    print(f"\n🚀 Ready to run quick demo comparison!")
    print(f"💻 Next steps:")
    print(f"   cd src")
    print(f"   python compare_models.py --models {' '.join([m+'_demo' for m in models])}")
    print(f"\n⏱️  Expected time: ~10-15 minutes for all {len(models)} models")
    print(f"🎯 This will train and test: {', '.join([m.upper() for m in models])}")
