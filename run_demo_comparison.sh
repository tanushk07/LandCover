#!/bin/bash

# Demo script for model architecture comparison
# This script demonstrates how to compare different architectures

echo "🛣 Land Cover Segmentation - Model Architecture Comparison Demo"
echo "=============================================================="

# Activate environment
echo "🔧 Activating environment..."
source land_cover_env/bin/activate

# Navigate to source directory
cd src

echo ""
echo "📚 This demo will show you how to:"
echo "  1. Train individual models"
echo "  2. Test individual models"
echo "  3. Compare multiple architectures"
echo ""

# Check if user wants to proceed
read -p "🤔 Do you want to run a quick comparison demo? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Demo cancelled. Check MODEL_COMPARISON_GUIDE.md for detailed instructions."
    exit 0
fi

echo ""
echo "🚀 Starting demo with 3 different architectures..."
echo ""

# Option 1: Quick test with existing model (if available)
if [ -f "../models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth" ]; then
    echo "📋 Option 1: Testing existing U-Net model..."
    python test_models.py --model unet --with-metrics
    echo ""
fi

# Option 2: Train and compare lightweight models (faster demo)
echo "📋 Option 2: Quick comparison of 2 architectures (U-Net vs LinkNet)..."
echo "⚡ This will be faster for demonstration purposes"
echo ""

read -p "🔄 Proceed with quick training demo? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Temporarily modify configs for faster training (fewer epochs)
    echo "⚙️  Setting up quick demo configs..."
    
    # Create temporary demo configs with fewer epochs
    cp ../config/models/unet_config.yaml ../config/models/unet_demo_config.yaml
    cp ../config/models/linknet_config.yaml ../config/models/linknet_demo_config.yaml
    
    # Modify epochs to 5 for demo (using sed)
    sed -i 's/epochs: 20/epochs: 5/' ../config/models/unet_demo_config.yaml
    sed -i 's/epochs: 20/epochs: 5/' ../config/models/linknet_demo_config.yaml
    
    echo "🏃‍♂️ Training U-Net (5 epochs for demo)..."
    python train_model.py --model unet_demo
    
    echo "🏃‍♂️ Training LinkNet (5 epochs for demo)..."
    python train_model.py --model linknet_demo
    
    echo "🧪 Testing both models..."
    python test_models.py --model unet_demo --with-metrics
    python test_models.py --model linknet_demo --with-metrics
    
    # Clean up demo configs
    rm ../config/models/unet_demo_config.yaml
    rm ../config/models/linknet_demo_config.yaml
    
    echo "✅ Demo completed! Check output directories for results."
else
    echo "📖 Demo skipped. Here's how to use the full framework:"
fi

echo ""
echo "🎯 Full Framework Usage:"
echo ""
echo "1️⃣  Train a single model:"
echo "   python train_model.py --model unet"
echo "   python train_model.py --model deeplabv3"
echo ""
echo "2️⃣  Test a trained model:"
echo "   python test_models.py --model unet --with-metrics"
echo ""
echo "3️⃣  Compare multiple models:"
echo "   python compare_models.py --models unet deeplabv3 linknet"
echo ""
echo "4️⃣  Compare all available models:"
echo "   python compare_models.py --models all"
echo ""
echo "5️⃣  Test existing models only (no training):"
echo "   python compare_models.py --models all --test-only"
echo ""
echo "📚 Available architectures:"
echo "   unet, deeplabv3, deeplabv3plus, linknet, fpn, pspnet, unetplusplus"
echo ""
echo "📖 For detailed instructions, see: MODEL_COMPARISON_GUIDE.md"
echo ""
echo "🎉 Happy experimenting with different architectures!"
