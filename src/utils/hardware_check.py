import torch

def print_hardware_status(step_name=""):
    print("\n" + "=" * 60)
    print(f"🖥️  HARDWARE STATUS CHECK: {step_name}")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU is AVAILABLE! Detected {gpu_count} GPU(s).")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  - Memory Reserved:  {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("❌ GPU is NOT available. Running on CPU!")
    print("=" * 60 + "\n")
