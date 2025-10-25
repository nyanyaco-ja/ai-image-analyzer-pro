"""
文書画像検出ロジックの簡易テスト（依存なし）
"""
import numpy as np

def is_document_image(img_rgb):
    """
    画像が文書/テキスト主体の画像かどうかを判定
    """
    try:
        # 1. 白背景率の計算（文書は白背景が多い）
        white_pixels = np.sum(np.all(img_rgb >= 240, axis=2))
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        white_ratio = white_pixels / total_pixels

        # 2. 色分散の計算（文書は色のバリエーションが少ない）
        color_std = np.std(img_rgb)

        # 3. グレースケール率（文書は白黒が多い）
        rgb_diff = np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)
        gray_pixels = np.sum(rgb_diff < 30)
        gray_ratio = gray_pixels / total_pixels

        # 判定基準
        is_document = (white_ratio > 0.60 and color_std < 50) or \
                     (gray_ratio > 0.80 and white_ratio > 0.40)

        print(f"  白背景率: {white_ratio*100:.1f}%")
        print(f"  色分散: {color_std:.1f}")
        print(f"  グレー率: {gray_ratio*100:.1f}%")
        print(f"  判定: {'文書画像 📄' if is_document else '自然画像 🖼️'}")

        return is_document

    except Exception as e:
        print(f"エラー: {e}")
        return False

print("=" * 80)
print("文書画像検出ロジックテスト")
print("=" * 80)

# テスト1: 白背景の医療カルテ風
print("\n【テスト1】白背景の医療カルテ風画像")
doc_img = np.ones((800, 600, 3), dtype=np.uint8) * 250
doc_img[100:150, 50:500] = 30
doc_img[200:230, 50:400] = 30
result = is_document_image(doc_img)
print(f"✅ 期待: 文書画像 → {'正解' if result else '不正解'}\n")

# テスト2: カラフルな自然画像
print("【テスト2】カラフルな自然画像")
natural_img = np.zeros((800, 600, 3), dtype=np.uint8)
natural_img[0:300, :, 0] = 120   # 空（青）R
natural_img[0:300, :, 1] = 170   # G
natural_img[0:300, :, 2] = 230   # B
natural_img[300:, :, 0] = 70     # 草（緑）R
natural_img[300:, :, 1] = 180    # G
natural_img[300:, :, 2] = 70     # B
result = is_document_image(natural_img)
print(f"✅ 期待: 自然画像 → {'正解' if not result else '不正解'}\n")

# テスト3: スキャン文書
print("【テスト3】スキャン文書風画像")
scan_img = np.ones((800, 600, 3), dtype=np.uint8) * 245
for i in range(10):
    y = 100 + i * 60
    scan_img[y:y+20, 50:550] = 20
result = is_document_image(scan_img)
print(f"✅ 期待: 文書画像 → {'正解' if result else '不正解'}\n")

# テスト4: 通常の写真（中間的な色分散）
print("【テスト4】通常の写真")
photo_img = np.random.randint(50, 200, (800, 600, 3), dtype=np.uint8)
result = is_document_image(photo_img)
print(f"✅ 期待: 自然画像 → {'正解' if not result else '不正解'}\n")

print("=" * 80)
print("テスト完了")
print("=" * 80)
