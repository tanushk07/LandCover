import torch
from utils.transformer_models import get_transformer_model
from utils.constants import Constants
from utils.model_config import get_model_config

def debug_model_output(model_name="SegFormer"):
    # Load config like your training script
    ROOT, slice_config = get_model_config(__file__, Constants, model_name)

    encoder = slice_config['vars']['encoder']
    encoder_weights = slice_config['vars']['encoder_weights']
    classes = slice_config['vars']['train_classes']
    num_classes = len(classes)

    print(f"\n🚀 Loading {model_name} | encoder={encoder} | num_classes={num_classes}")

    # Build model (same as in train_model.py)
    model = get_transformer_model(
        model_arch=model_name,
        num_classes=num_classes,
        encoder=encoder,
        encoder_weights=encoder_weights
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Dummy input to check output shape
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nRaw model output type: {type(output)}")

    # Unwrap dictionaries or tuples
    if isinstance(output, dict):
        for k, v in output.items():
            print(f"  Key: {k} | Shape: {v.shape if isinstance(v, torch.Tensor) else type(v)}")
        if "pred_masks" in output:
            output = output["pred_masks"]
        elif "out" in output:
            output = output["out"]
        else:
            output = list(output.values())[0]

    elif isinstance(output, (tuple, list)):
        print(f"Tuple/List output length: {len(output)}")
        for i, v in enumerate(output):
            if isinstance(v, torch.Tensor):
                print(f"  Element[{i}] Shape: {v.shape}")
        output = output[0]

    print(f"\nFinal tensor used for prediction: {output.shape}")
    print(f"Min value: {output.min().item():.6f}, Max value: {output.max().item():.6f}")

    # Sanity check: Should have 5 channels
    if output.ndim == 4:
        B, C, H, W = output.shape
        print(f"✅ Output is 4D: (B={B}, C={C}, H={H}, W={W})")
        if C != num_classes:
            print(f"⚠️ Mismatch! Output has {C} channels but dataset has {num_classes} classes.")
        else:
            print("🎯 Output channels match dataset classes.")
    else:
        print(f"⚠️ Unexpected output shape: {output.shape}")

if __name__ == "__main__":
    for arch in ["MaskDINO", "SegFormer", "SCTNet"]:
        debug_model_output(arch)
