#!/bin/bash

# Activation script for Land Cover Semantic Segmentation PyTorch environment
# Usage: source activate_env.sh

echo "🛣 Activating Land Cover Semantic Segmentation PyTorch environment..."

# Activate the virtual environment
source land_cover_env/bin/activate

# Display environment information
echo "✅ Environment activated successfully!"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Current directory: $(pwd)"

echo ""
echo "📚 Available commands:"
echo "  Original Scripts:"
echo "    Training:    cd src && python train.py"
echo "    Testing:     cd src && python test.py"
echo "    Inference:   cd src && python inference.py"
echo ""
echo "  🆕 Model Comparison Framework:"
echo "    Single Model:     cd src && python train_model.py --model unet"
echo "    Test Model:       cd src && python test_models.py --model unet"
echo "    Compare Models:   cd src && python compare_models.py --models unet deeplabv3 linknet"
echo "    Compare All:      cd src && python compare_models.py --models all"
echo ""
echo "📝 Configuration file: config/config.yaml"
echo "📁 Models directory: models/"
echo "📊 Output directory: output/"
echo ""
echo "To deactivate the environment, run: deactivate"
