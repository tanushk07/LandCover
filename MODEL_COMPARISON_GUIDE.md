# 🛣 Model Architecture Comparison Guide

This guide explains how to use the enhanced Land Cover Semantic Segmentation framework to compare different model architectures like U-Net, DeepLab, LinkNet, and more.

## 🎯 Available Model Architectures

The framework supports the following architectures from `segmentation_models_pytorch`:

| Architecture | Description | Best For |
|--------------|-------------|----------|
| **Unet** | Classic U-Net with skip connections | General-purpose segmentation |
| **DeepLabV3** | Atrous convolution with ASPP | Large objects, varied scales |
| **DeepLabV3Plus** | DeepLabV3 + encoder-decoder | Fine detail preservation |
| **LinkNet** | Efficient with fewer parameters | Real-time applications |
| **FPN** | Feature Pyramid Network | Multi-scale feature extraction |
| **PSPNet** | Pyramid Scene Parsing | Scene understanding |
| **UnetPlusPlus** | U-Net with nested skip connections | Fine-grained segmentation |

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the environment
source activate_env.sh

# Navigate to source directory
cd src
```

### 2. Single Model Training
```bash
# Train a specific model
python train_model.py --model unet
python train_model.py --model deeplabv3
python train_model.py --model linknet
```

### 3. Single Model Testing
```bash
# Test a specific model
python test_models.py --model unet
python test_models.py --model deeplabv3 --with-metrics
```

### 4. Automated Model Comparison
```bash
# Compare specific models
python compare_models.py --models unet deeplabv3 linknet

# Compare all available models
python compare_models.py --models all

# Test only (skip training)
python compare_models.py --models all --test-only
```

## 📁 Configuration System

### Model-Specific Configurations

Each model has its own configuration file in `config/models/`:

```
config/models/
├── unet_config.yaml
├── deeplabv3_config.yaml
├── deeplabv3plus_config.yaml
├── linknet_config.yaml
├── fpn_config.yaml
├── pspnet_config.yaml
└── unetplusplus_config.yaml
```

### Customizing Model Parameters

You can modify any model's configuration by editing its YAML file:

```yaml
vars:
  model_arch: 'Unet'              # Architecture type
  encoder: 'efficientnet-b0'      # Encoder backbone
  encoder_weights: 'imagenet'     # Pre-trained weights
  batch_size: 16                  # Batch size
  epochs: 20                      # Training epochs
  init_lr: 0.0003                 # Learning rate
  # ... other parameters
```

### Available Encoders

You can experiment with different encoders for any architecture:

| Encoder Family | Examples |
|----------------|----------|
| **EfficientNet** | efficientnet-b0, efficientnet-b1, efficientnet-b2, etc. |
| **ResNet** | resnet18, resnet34, resnet50, resnet101, resnet152 |
| **ResNeXt** | resnext50_32x4d, resnext101_32x8d |
| **DenseNet** | densenet121, densenet169, densenet201 |
| **VGG** | vgg11, vgg13, vgg16, vgg19 |
| **Inception** | inceptionv3, inceptionv4, inception_resnet_v2 |

## 🧪 Detailed Workflow

### Step 1: Data Preparation

Ensure your data is structured as:
```
data/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Step 2: Individual Model Training

```bash
# Train U-Net
python train_model.py --model unet

# Train DeepLabV3+
python train_model.py --model deeplabv3plus

# Train LinkNet
python train_model.py --model linknet
```

Each training session will:
- Create model-specific patches
- Train with validation monitoring
- Save the best model based on IoU score
- Log detailed training progress
- Clean up temporary files

### Step 3: Model Testing

```bash
# Test with comprehensive metrics
python test_models.py --model unet --with-metrics
```

Testing will generate:
- Predicted masks
- Visualization plots
- Comprehensive metrics (IoU, mAP, Dice, etc.)
- Per-class performance analysis

### Step 4: Automated Comparison

```bash
# Compare multiple architectures
python compare_models.py --models unet deeplabv3plus linknet fpn
```

This will:
1. **Train** all specified models (unless `--test-only`)
2. **Test** all models with comprehensive metrics
3. **Generate** comparison reports and visualizations
4. **Rank** models by performance

## 📊 Understanding Results

### Training Outputs

Each model training produces:
```
models/
└── landcover_unet_efficientnet-b0_adam_epochs15_patch512_batch16.pth

logs/
├── unet_train.log
├── deeplabv3_train.log
└── linknet_train.log
```

### Testing Outputs

Each model test produces:
```
output/
├── unet_experiment/
│   ├── predicted_masks/
│   ├── prediction_plots/
│   └── metrics/
├── deeplabv3_experiment/
│   └── ...
└── linknet_experiment/
    └── ...
```

### Comparison Results

Model comparison generates:
```
output/model_comparison/
├── comparison_results_2024-01-15_14-30-25.json
├── model_comparison_summary_2024-01-15_14-30-25.csv
├── detailed_results_2024-01-15_14-30-25.csv
├── performance_comparison_2024-01-15_14-30-25.png
├── train_vs_test_2024-01-15_14-30-25.png
└── performance_vs_time_2024-01-15_14-30-25.png
```

## 🎨 Customization Options

### 1. Adding New Architectures

To add a new architecture:

1. Create a new config file: `config/models/mymodel_config.yaml`
2. Ensure the architecture is available in `segmentation_models_pytorch`
3. Add the model name to the choices in argument parsers

### 2. Custom Encoders

Change the encoder in any config file:
```yaml
vars:
  encoder: 'resnet50'           # Change from efficientnet-b0
  encoder_weights: 'imagenet'   # Keep pre-trained weights
```

### 3. Hyperparameter Tuning

Modify training parameters:
```yaml
vars:
  batch_size: 32              # Increase batch size
  init_lr: 0.001              # Higher learning rate
  epochs: 50                  # More epochs
  patch_size: 256             # Smaller patches
```

### 4. Custom Classes

Modify the class configuration:
```yaml
vars:
  all_classes: ['background', 'building', 'woodland', 'water', 'road']
  train_classes: ['background', 'building', 'woodland', 'water', 'road']  # Use all classes
  test_classes: ['background', 'building', 'water']                      # Test subset
```

## 📈 Performance Metrics

### Metrics Calculated

| Metric | Description | Range |
|--------|-------------|-------|
| **Mean IoU** | Average Intersection over Union | 0-1 (higher better) |
| **Pixel Accuracy** | Correct pixels / Total pixels | 0-1 (higher better) |
| **mAP@50** | Mean Average Precision at 50% IoU | 0-1 (higher better) |
| **mAP@75** | Mean Average Precision at 75% IoU | 0-1 (higher better) |
| **Dice Score** | 2×TP/(2×TP+FP+FN) | 0-1 (higher better) |
| **Frequency Weighted IoU** | IoU weighted by class frequency | 0-1 (higher better) |

### Per-Class Analysis

Each metric is calculated both:
- **Overall**: Across all classes combined
- **Per-Class**: Individual performance for each land cover class

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   # Reduce batch size in config
   batch_size: 8  # Instead of 16
   ```

2. **Model Not Found**
   ```bash
   # Check if model was trained
   ls models/
   
   # Train the model first
   python train_model.py --model unet
   ```

3. **Config File Issues**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/models/unet_config.yaml'))"
   ```

### Performance Tips

1. **Faster Training**: Use smaller patch sizes or fewer epochs
2. **Better Accuracy**: Use larger encoders (efficientnet-b4, resnet101)
3. **Memory Optimization**: Reduce batch size and use gradient accumulation
4. **Faster Inference**: Use LinkNet or smaller encoders

## 🎯 Example Workflows

### Research Comparison
```bash
# Compare all major architectures
python compare_models.py --models unet deeplabv3plus linknet fpn pspnet

# Focus on encoder comparison
# Edit configs to use different encoders, then:
python compare_models.py --models unet unet unet  # With different encoder configs
```

### Production Deployment
```bash
# Find fastest model
python compare_models.py --models linknet fpn

# Test efficiency vs accuracy trade-off
python compare_models.py --models unet linknet --test-only
```

### Ablation Studies
```bash
# Compare patch sizes (edit configs)
python train_model.py --model unet  # patch_size: 256
python train_model.py --model unet  # patch_size: 512

# Compare loss functions (modify source code)
# Compare augmentation strategies (edit preprocess.py)
```

This framework gives you complete flexibility to experiment with different architectures, encoders, and hyperparameters while maintaining consistent evaluation and comparison across all experiments.
