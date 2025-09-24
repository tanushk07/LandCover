#!/bin/bash

# Transformer vs CNN Model Comparison Script
# This script compares transformer-based models with traditional CNN models

echo "🤖 Land Cover Segmentation - Transformer vs CNN Comparison"
echo "========================================================="
echo ""
echo "🎯 This demo will compare:"
echo "   • CNN Models: U-Net, DeepLabV3, LinkNet"
echo "   • Transformer Models: SegFormer"
echo "   • Using the same dataset (4 train + 2 test images)"
echo ""
echo "⏱️  Expected time: ~15-20 minutes"
echo ""

# Check if user wants to proceed
read -p "🚀 Ready to start transformer comparison? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Demo cancelled."
    exit 0
fi

# Activate environment
echo "🔧 Activating environment..."
source land_cover_env/bin/activate

# Setup demo data if not already done
if [ ! -d "data/train/images" ] || [ ! "$(ls -A data/train/images)" ]; then
    echo ""
    echo "📁 Setting up demo data..."
    python setup_quick_demo.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to setup demo data. Exiting."
        exit 1
    fi
fi

# Navigate to source directory
cd src

echo ""
echo "🏗️  Starting Transformer vs CNN comparison..."
echo ""

# Compare CNN models with transformer models
python compare_models.py --models unet_demo linknet_demo segformer_demo

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Transformer vs CNN comparison completed successfully!"
    echo ""
    echo "📊 Results are available in:"
    echo "   • output/model_comparison/ - Comparison reports and plots"
    echo "   • output/unet_demo/ - U-Net (CNN) results"
    echo "   • output/linknet_demo/ - LinkNet (CNN) results"
    echo "   • output/segformer_demo/ - SegFormer (Transformer) results"
    echo ""
    echo "📈 Key insights to look for:"
    echo "   • Parameter count differences (Transformers typically have more)"
    echo "   • Training time differences"
    echo "   • Segmentation accuracy (IoU/Pixel Accuracy)"
    echo "   • Visual quality of predictions"
    echo ""
    
    # Show quick results if CSV exists
    LATEST_CSV=$(ls -t ../output/model_comparison/model_comparison_summary_*.csv 2>/dev/null | head -1)
    if [ -f "$LATEST_CSV" ]; then
        echo "🏆 Quick Results Preview:"
        echo "========================"
        head -5 "$LATEST_CSV" | column -t -s,
        echo ""
        echo "📝 Key Observations:"
        echo "   • SegFormer: Modern transformer-based architecture"
        echo "   • U-Net: Classic CNN with skip connections"
        echo "   • LinkNet: Efficient CNN for real-time applications"
        echo ""
    fi
    
    echo "🔍 For detailed analysis:"
    echo "   • Check performance_comparison_*.png for visual metrics"
    echo "   • Compare prediction quality in prediction_plots/ folders"
    echo "   • Review training logs for convergence patterns"
    
else
    echo ""
    echo "❌ Comparison failed. You can try individual training:"
    echo "   python train_model.py --model segformer_demo"
    echo "   python test_models.py --model segformer_demo"
fi

# Return to root directory
cd ..

echo ""
echo "🧠 Transformer vs CNN Analysis Summary:"
echo "======================================="
echo "📊 Transformers typically offer:"
echo "   ✅ Better global context understanding"
echo "   ✅ Attention-based feature learning"
echo "   ✅ Potential for better performance on complex scenes"
echo "   ❌ Higher computational requirements"
echo "   ❌ More memory usage"
echo "   ❌ Longer training times"
echo ""
echo "🔧 CNNs typically offer:"
echo "   ✅ Faster training and inference"
echo "   ✅ Lower memory requirements"
echo "   ✅ Well-established architectures"
echo "   ✅ Good inductive biases for spatial data"
echo "   ❌ Limited global context modeling"
echo ""
echo "💡 Choose based on your use case:"
echo "   • Real-time applications → CNNs (LinkNet, U-Net)"
echo "   • Highest accuracy → Transformers (SegFormer)"
echo "   • Balanced performance → Hybrid approaches"
echo ""
echo "✨ Comparison completed! Check the output directories for detailed results."
