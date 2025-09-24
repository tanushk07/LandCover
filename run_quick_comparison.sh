#!/bin/bash

# Quick Model Architecture Comparison Script
# This script runs a fast comparison with limited data for demonstration

echo "🛣 Land Cover Segmentation - Quick Model Comparison"
echo "=================================================="
echo ""
echo "🎯 This demo will:"
echo "   • Use 4 training images + 2 test images"
echo "   • Train 4 different architectures (3 epochs each)"
echo "   • Compare results automatically"
echo "   • Generate comparison reports and plots"
echo ""
echo "⏱️  Expected time: ~10-15 minutes"
echo ""

# Check if user wants to proceed
read -p "🚀 Ready to start quick comparison? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Demo cancelled."
    exit 0
fi

# Activate environment
echo "🔧 Activating environment..."
source land_cover_env/bin/activate

# Setup demo data and configs
echo ""
echo "📁 Setting up demo data (4 train + 2 test images)..."
python setup_quick_demo.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to setup demo data. Exiting."
    exit 1
fi

# Navigate to source directory
cd src

echo ""
echo "🏗️  Starting model comparison..."
echo "   Models: U-Net, DeepLabV3, LinkNet, FPN"
echo "   Settings: 3 epochs, 256px patches, batch size 4"
echo ""

# Run the comparison
python compare_models.py --models unet_demo deeplabv3_demo linknet_demo fpn_demo

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Quick comparison completed successfully!"
    echo ""
    echo "📊 Results are available in:"
    echo "   • output/model_comparison/ - Comparison reports and plots"
    echo "   • output/unet_demo/ - U-Net specific results"
    echo "   • output/deeplabv3_demo/ - DeepLabV3 results"
    echo "   • output/linknet_demo/ - LinkNet results"
    echo "   • output/fpn_demo/ - FPN results"
    echo ""
    echo "📈 Key files to check:"
    echo "   • model_comparison_summary_*.csv - Performance ranking"
    echo "   • performance_comparison_*.png - Visual comparison"
    echo "   • comparison_results_*.json - Detailed metrics"
    echo ""
    
    # Show quick results if CSV exists
    LATEST_CSV=$(ls -t ../output/model_comparison/model_comparison_summary_*.csv 2>/dev/null | head -1)
    if [ -f "$LATEST_CSV" ]; then
        echo "🏆 Quick Results Preview:"
        echo "========================"
        head -5 "$LATEST_CSV" | column -t -s,
        echo ""
    fi
    
    echo "🔍 For detailed analysis:"
    echo "   • Open the generated PNG plots"
    echo "   • Check the CSV files for numerical results"
    echo "   • View prediction_plots/ for visual outputs"
    
else
    echo ""
    echo "❌ Comparison failed. Check the error messages above."
    echo "💡 You can try running individual models:"
    echo "   python train_model.py --model unet_demo"
    echo "   python test_models.py --model unet_demo"
fi

# Return to root directory
cd ..

# Optional cleanup
echo ""
read -p "🧹 Clean up demo configuration files? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -c "
from setup_quick_demo import cleanup_demo
cleanup_demo()
print('✅ Demo configs cleaned up!')
"
fi

echo ""
echo "✨ Demo completed! Check the output directories for detailed results."
echo ""
echo "🎯 To run with more data or different settings:"
echo "   • Modify the image lists in setup_quick_demo.py"
echo "   • Adjust epochs/batch_size in the demo configs"
echo "   • Use: python compare_models.py --models all (for full comparison)"
echo ""
echo "📚 See MODEL_COMPARISON_GUIDE.md for advanced usage."
