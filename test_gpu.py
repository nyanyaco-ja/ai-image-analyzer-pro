import torch
import sys

print("=" * 60)
print("GPU検出テスト")
print("=" * 60)

print(f"\nPyTorchバージョン: {torch.__version__}")
print(f"CUDA利用可能: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDAバージョン: {torch.version.cuda}")
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
    print(f"GPU数: {torch.cuda.device_count()}")

    # 簡単なGPUテスト
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)

    print("\nGPU計算テスト中...")
    for i in range(100):
        z = torch.matmul(x, y)

    print("✓ GPU計算成功")
    print(f"\n使用デバイス: {device}")
    print(f"VRAM使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

else:
    print("\n❌ CUDAが利用できません")
    print("\n考えられる原因:")
    print("1. CUDA対応のPyTorchがインストールされていない")
    print("2. NVIDIAドライバーが古い")
    print("3. CPU版PyTorchがインストールされている")
    print("\nCPU版PyTorchかどうか確認:")
    print(f"  torch.version.cuda = {torch.version.cuda}")

    print("\n解決方法:")
    print("  CUDA版PyTorchを再インストール:")
    print("  pip uninstall torch torchvision")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 60)
