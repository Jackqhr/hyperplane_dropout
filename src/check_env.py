import torch

print("="*30)
print(f"✅ PyTorch 版本: {torch.__version__}")
print(f"✅ Torchvision 版本: {torch.__version__}")

if torch.cuda.is_available():
    print(f"🚀 CUDA 可用! 当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 CUDA 版本: {torch.version.cuda}")
else:
    print("⚠️ CUDA 不可用，将使用 CPU 进行训练 (速度较慢，但完全可行)")
print("="*30)