#!/usr/bin/env python3
"""
Quick test to verify that all advanced models can be instantiated correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from utils.transformer_models import (
    SegFormerModel, MaskDINOModel, SCTNetModel, 
    EnhancedDeepLabV3Plus, get_transformer_model
)

def test_model_compilation():
    """Test that all models can be instantiated and forward pass works."""
    
    num_classes = 4
    device = torch.device('cpu')
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256)
    
    models_to_test = [
        ("SegFormer", "segformer-b0"),
        ("MaskDINO", "detr"),
        ("SCTNet", "selfcorrecting"),
        ("EnhancedDeepLabV3Plus", "aspp")
    ]
    
    print("🧪 Testing Advanced Segmentation Models Compilation")
    print("=" * 55)
    
    for model_arch, encoder in models_to_test:
        try:
            print(f"\n🔍 Testing {model_arch}...")
            
            # Instantiate model
            model = get_transformer_model(
                model_arch=model_arch,
                num_classes=num_classes,
                encoder=encoder,
                encoder_weights="imagenet"
            )
            
            model.eval()
            print(f"   ✅ Model instantiated successfully")
            
            # Test forward pass
            with torch.no_grad():
                output = model(test_input)
            
            expected_shape = (1, num_classes, 256, 256)
            if output.shape == expected_shape:
                print(f"   ✅ Forward pass successful - Output shape: {output.shape}")
            else:
                print(f"   ⚠️  Forward pass shape mismatch - Expected: {expected_shape}, Got: {output.shape}")
            
            # Test predict method
            pred_output = model.predict(test_input)
            if pred_output.shape == expected_shape:
                print(f"   ✅ Predict method successful")
            else:
                print(f"   ⚠️  Predict method shape mismatch")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   📊 Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
        except Exception as e:
            print(f"   ❌ Error with {model_arch}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Model compilation test completed!")

if __name__ == "__main__":
    test_model_compilation()
