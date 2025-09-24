#!/bin/bash

# Advanced Segmentation Models Comparison Script
# Tests Mask DINO, SCTNet, Enhanced DeepLabV3+, and SegFormer

echo "🤖 Advanced Segmentation Models Comparison"
echo "=========================================="
echo ""
echo "🎯 Testing advanced models:"
echo "   • Mask DINO: DETR-based unified segmentation"
echo "   • SCTNet: Self-Correcting Transformer Network"
echo "   • Enhanced DeepLabV3+: Improved ASPP architecture"
echo "   • SegFormer: Hierarchical transformer"
echo ""
echo "⏱️  Expected time: ~20-25 minutes"
echo ""

# Check if user wants to proceed
read -p "🚀 Ready to start advanced models comparison? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Demo cancelled."
    exit 0
fi

# Activate environment
echo "🔧 Activating environment..."
source land_cover_env/bin/activate

# Setup demo data if needed
echo "📁 Setting up demo data..."
python setup_quick_demo.py

# Define models to test
MODELS=("maskdino_demo" "sctnet_demo" "enhanced_deeplab_demo" "segformer_demo")
MODEL_NAMES=("Mask DINO" "SCTNet" "Enhanced DeepLabV3+" "SegFormer")

echo ""
echo "🏃‍♂️ Starting training phase..."
echo "================================"

# Train each model
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    name="${MODEL_NAMES[$i]}"
    
    echo ""
    echo "🎯 Training $name ($model)..."
    echo "Time: $(date '+%H:%M:%S')"
    
    cd src
    if python train_model.py --model $model; then
        echo "✅ $name training completed successfully!"
    else
        echo "❌ $name training failed!"
        cd ..
        continue
    fi
    cd ..
done

echo ""
echo "🧪 Starting testing phase..."
echo "=============================="

# Test each model
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    name="${MODEL_NAMES[$i]}"
    
    echo ""
    echo "🔍 Testing $name ($model)..."
    echo "Time: $(date '+%H:%M:%S')"
    
    cd src
    if python test_models.py --model $model; then
        echo "✅ $name testing completed successfully!"
    else
        echo "❌ $name testing failed!"
        cd ..
        continue
    fi
    cd ..
done

echo ""
echo "📊 Running comprehensive comparison..."
echo "====================================="

cd src
if python compare_models.py --models maskdino_demo sctnet_demo enhanced_deeplab_demo segformer_demo; then
    echo "✅ Comparison completed successfully!"
else
    echo "❌ Comparison failed!"
fi
cd ..

echo ""
echo "🎉 Advanced models comparison completed!"
echo "======================================="
echo ""
echo "📁 Results available in:"
echo "   • output/comparison_results/"
echo "   • Individual model outputs in output/[model]_demo/"
echo ""
echo "📈 Key files:"
echo "   • comparison_metrics.csv - Quantitative results"
echo "   • comparison_plots/ - Visual comparisons"
echo "   • model_comparison_report.json - Detailed analysis"
echo ""
echo "💡 Tips:"
echo "   • Check the CSV file for IoU, Dice, and accuracy scores"
echo "   • Visual plots show prediction quality differences"
echo "   • JSON report contains training times and model sizes"
