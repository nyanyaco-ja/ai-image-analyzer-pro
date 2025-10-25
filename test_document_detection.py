"""
文書画像検出機能のテストスクリプト
"""
import numpy as np
from PIL import Image
from advanced_image_analyzer import is_document_image

print("=" * 80)
print("文書画像検出機能テスト")
print("=" * 80)

# テスト1: 白背景の文書画像（医療カルテ風）
print("\n【テスト1】白背景の医療カルテ風画像")
doc_img = np.ones((800, 600, 3), dtype=np.uint8) * 250  # ほぼ白背景
# テキスト領域を黒で追加（カルテの文字を模擬）
doc_img[100:150, 50:500] = 30   # ヘッダー
doc_img[200:230, 50:400] = 30   # テキスト行1
doc_img[250:280, 50:450] = 30   # テキスト行2
doc_img[300:330, 50:350] = 30   # テキスト行3
result = is_document_image(doc_img)
print(f"判定結果: {'文書画像' if result else '自然画像'}")
print(f"期待値: 文書画像 → {'✅ 正解' if result else '❌ 不正解'}")

# テスト2: 自然画像（カラフルな風景）
print("\n【テスト2】カラフルな自然画像（風景）")
natural_img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
# 空（青）
natural_img[0:300, :, 0] = np.random.randint(100, 150, (300, 600))   # R
natural_img[0:300, :, 1] = np.random.randint(150, 200, (300, 600))   # G
natural_img[0:300, :, 2] = np.random.randint(200, 255, (300, 600))   # B
# 草（緑）
natural_img[300:, :, 0] = np.random.randint(50, 100, (500, 600))     # R
natural_img[300:, :, 1] = np.random.randint(150, 200, (500, 600))    # G
natural_img[300:, :, 2] = np.random.randint(50, 100, (500, 600))     # B
result = is_document_image(natural_img)
print(f"判定結果: {'文書画像' if result else '自然画像'}")
print(f"期待値: 自然画像 → {'✅ 正解' if not result else '❌ 不正解'}")

# テスト3: グレースケールの医療レントゲン風
print("\n【テスト3】グレースケールの医療レントゲン風画像")
xray_img = np.random.randint(100, 200, (800, 600), dtype=np.uint8)
xray_img = np.stack([xray_img, xray_img, xray_img], axis=2)  # RGBに変換
result = is_document_image(xray_img)
print(f"判定結果: {'文書画像' if result else '自然画像'}")
print(f"期待値: 自然画像（レントゲンは医療画像だが文書ではない） → {'✅ 正解' if not result else '⚠️  誤検出（許容範囲）'}")

# テスト4: 高い白背景率 + 低い色分散（スキャン文書）
print("\n【テスト4】スキャン文書風画像")
scan_img = np.ones((800, 600, 3), dtype=np.uint8) * 245
# テキスト領域（黒）
for i in range(10):
    y = 100 + i * 60
    scan_img[y:y+20, 50:550] = 20
result = is_document_image(scan_img)
print(f"判定結果: {'文書画像' if result else '自然画像'}")
print(f"期待値: 文書画像 → {'✅ 正解' if result else '❌ 不正解'}")

# テスト5: 写真（顔画像風）
print("\n【テスト5】写真（人物画像風）")
photo_img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
# 肌色（顔領域）
photo_img[200:600, 200:400, 0] = np.random.randint(200, 230, (400, 200))  # R
photo_img[200:600, 200:400, 1] = np.random.randint(170, 200, (400, 200))  # G
photo_img[200:600, 200:400, 2] = np.random.randint(150, 180, (400, 200))  # B
# 背景（ランダム色）
result = is_document_image(photo_img)
print(f"判定結果: {'文書画像' if result else '自然画像'}")
print(f"期待値: 自然画像 → {'✅ 正解' if not result else '❌ 不正解'}")

print("\n" + "=" * 80)
print("テスト完了")
print("=" * 80)
