"""
CLIP機能のテストスクリプト
"""
import sys
import os

# 必要なモジュールのインポートテスト
print("=" * 80)
print("CLIP機能テスト")
print("=" * 80)

print("\n1. モジュールのインポートテスト...")
try:
    from transformers import CLIPProcessor, CLIPModel
    print("  ✅ transformers ライブラリが正常にインポートされました")
    CLIP_AVAILABLE = True
except ImportError as e:
    print(f"  ❌ transformers ライブラリのインポートに失敗: {e}")
    print("  → pip install transformers を実行してください")
    CLIP_AVAILABLE = False
    sys.exit(1)

try:
    import torch
    print(f"  ✅ PyTorch が正常にインポートされました (version: {torch.__version__})")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA利用可能: {torch.cuda.get_device_name(0)}")
        DEVICE = torch.device('cuda')
    else:
        print(f"  ⚠️  CUDA利用不可: CPU処理を使用します")
        DEVICE = torch.device('cpu')
except ImportError as e:
    print(f"  ❌ PyTorch のインポートに失敗: {e}")
    sys.exit(1)

print("\n2. CLIPモデルのロードテスト...")
try:
    print("  CLIPモデルをダウンロード中... (初回のみ時間がかかります)")
    # safetensors形式でロード（PyTorch 2.6未満でも動作）
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # GPUに転送
    if DEVICE.type == 'cuda':
        model = model.to(DEVICE)

    model.eval()
    print("  ✅ CLIPモデルのロードに成功しました")
    print(f"     デバイス: {DEVICE}")
    print(f"     モデルサイズ: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M パラメータ")
except Exception as e:
    print(f"  ❌ CLIPモデルのロードに失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. ダミー画像での動作テスト...")
try:
    from PIL import Image
    import numpy as np

    # 2つの類似したダミー画像を作成（200x200、RGB）
    print("  ダミー画像を作成中...")
    img1_array = np.random.randint(100, 150, (200, 200, 3), dtype=np.uint8)
    # ノイズを追加（uint8の範囲内でクリップ）
    noise = np.random.randint(-10, 11, (200, 200, 3), dtype=np.int16)
    img2_array = np.clip(img1_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img1 = Image.fromarray(img1_array)
    img2 = Image.fromarray(img2_array)

    # 画像を前処理
    print("  画像を前処理中...")
    inputs1 = processor(images=img1, return_tensors="pt")
    inputs2 = processor(images=img2, return_tensors="pt")

    # GPUに転送
    if DEVICE.type == 'cuda':
        inputs1 = {k: v.to(DEVICE) for k, v in inputs1.items()}
        inputs2 = {k: v.to(DEVICE) for k, v in inputs2.items()}

    # Embedding抽出
    print("  Embeddingを抽出中...")
    with torch.no_grad():
        image_features1 = model.get_image_features(**inputs1)
        image_features2 = model.get_image_features(**inputs2)

    # L2正規化
    image_features1 = image_features1 / image_features1.norm(p=2, dim=-1, keepdim=True)
    image_features2 = image_features2 / image_features2.norm(p=2, dim=-1, keepdim=True)

    # コサイン類似度を計算
    cosine_similarity = (image_features1 @ image_features2.T).item()

    print(f"  ✅ CLIP類似度計算に成功しました")
    print(f"     類似度: {cosine_similarity:.4f}")
    print(f"     (ランダム画像なので類似度は高めに出る傾向があります)")

except Exception as e:
    print(f"  ❌ 動作テストに失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. advanced_image_analyzer.py との統合テスト...")
try:
    from advanced_image_analyzer import calculate_clip_similarity, CLIP_AVAILABLE as ANALYZER_CLIP_AVAILABLE

    if not ANALYZER_CLIP_AVAILABLE:
        print("  ❌ advanced_image_analyzer.py でCLIPが利用不可と判定されています")
        sys.exit(1)

    # RGB numpy配列での動作テスト
    print("  calculate_clip_similarity() 関数をテスト中...")
    result = calculate_clip_similarity(img1_array, img2_array)

    if result is not None:
        print(f"  ✅ calculate_clip_similarity() が正常に動作しました")
        print(f"     類似度: {result:.4f}")
    else:
        print(f"  ❌ calculate_clip_similarity() がNoneを返しました")
        sys.exit(1)

except Exception as e:
    print(f"  ❌ 統合テストに失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ すべてのテストが正常に完了しました！")
print("=" * 80)
print("\nCLIP機能は正常に動作しています。")
print("GUI (modern_gui.py) から使用できます。")
print("=" * 80)
