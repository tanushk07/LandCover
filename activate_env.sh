#!/bin/bash

# Activation script for Land Cover Semantic Segmentation PyTorch environment
# Usage: source activate_env.sh

echo "🛣 Activating Land Cover Semantic Segmentation PyTorch environment (segment_env)..."

# Activate the virtual environment
if [ -d "segment_env" ]; then
    source segment_env/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️ Warning: Could not find segment_env or venv directory. Please make sure your environment is created."
fi

# Display environment information
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Environment activated successfully!"
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    echo "Current directory: $(pwd)"
    
    echo ""
    echo "📚 Available commands (run from project root):"
    echo "  Training:"
    echo "    python src/train_model.py --model unet"
    echo "    python src/train_model.py --model deeplabv3"
    echo ""
    echo "  Testing & Metrics:"
    echo "    python src/test_models.py --model unet"
    echo ""
    echo "  Model Comparison:"
    echo "    python src/compare_models.py --models unet deeplabv3 linknet"
    echo "    python src/compare_models.py --models all"
    echo ""
    echo "  Inference:"
    echo "    python src/inference.py"
    echo ""
    echo "📝 Configuration files: config/models/"
    echo "📁 Models directory: models/"
    echo "📊 Output directory: output/"
    echo ""
    echo "To deactivate the environment, run: deactivate"
else
    echo "❌ Error: Failed to activate environment."
fi
