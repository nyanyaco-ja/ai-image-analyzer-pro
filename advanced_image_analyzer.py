import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
try:
    from skimage.metrics import structural_similarity as compare_ssim
    from pytorch_msssim import ms_ssim as pytorch_ms_ssim
    MS_SSIM_AVAILABLE = True
except ImportError:
    MS_SSIM_AVAILABLE = False
import matplotlib
matplotlib.use('Agg')  # GUI非表示のバックエンドを使用
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy import stats
from skimage import feature
import json
from datetime import datetime

# LPIPS用インポート
try:
    import torch
    import torch.nn.functional as F
    import lpips
    import kornia
    import kornia.filters as KF
    import kornia.color as KC
    LPIPS_AVAILABLE = True
    KORNIA_AVAILABLE = True

    # GPU利用可否をチェック
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_AVAILABLE = True
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        DEVICE = torch.device('cpu')
        GPU_AVAILABLE = False
        GPU_NAME = None
except ImportError:
    LPIPS_AVAILABLE = False
    KORNIA_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = None
    GPU_NAME = None
    torch = None

# CLIP用インポート
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
    CLIP_MODEL = None
    CLIP_PROCESSOR = None
except ImportError:
    CLIP_AVAILABLE = False
    CLIP_MODEL = None
    CLIP_PROCESSOR = None

# CPU/GPUモニタリング用
try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

def get_system_usage():
    """CPU/GPU使用率を取得"""
    usage_info = {}

    if MONITORING_AVAILABLE:
        # CPU使用率
        usage_info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        usage_info['cpu_count'] = psutil.cpu_count()

        # メモリ使用率
        memory = psutil.virtual_memory()
        usage_info['ram_percent'] = memory.percent
        usage_info['ram_used_gb'] = memory.used / (1024**3)
        usage_info['ram_total_gb'] = memory.total / (1024**3)

        # GPU使用率
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                usage_info['gpu_percent'] = gpu.load * 100
                usage_info['gpu_memory_percent'] = gpu.memoryUtil * 100
                usage_info['gpu_memory_used_mb'] = gpu.memoryUsed
                usage_info['gpu_memory_total_mb'] = gpu.memoryTotal
                usage_info['gpu_temp'] = gpu.temperature
            else:
                usage_info['gpu_percent'] = None
        except:
            usage_info['gpu_percent'] = None

    return usage_info

def print_usage_status(stage_name):
    """処理段階ごとの使用率を表示"""
    if not MONITORING_AVAILABLE:
        return

    usage = get_system_usage()

    print(f"\n[{stage_name}] システム使用状況:")
    print(f"  CPU: {usage.get('cpu_percent', 0):.1f}% ({usage.get('cpu_count', 0)}コア)")
    print(f"  RAM: {usage.get('ram_percent', 0):.1f}% ({usage.get('ram_used_gb', 0):.1f}/{usage.get('ram_total_gb', 0):.1f} GB)")

    if usage.get('gpu_percent') is not None:
        print(f"  GPU: {usage.get('gpu_percent', 0):.1f}% 使用中")
        print(f"  VRAM: {usage.get('gpu_memory_percent', 0):.1f}% ({usage.get('gpu_memory_used_mb', 0):.0f}/{usage.get('gpu_memory_total_mb', 0):.0f} MB)")
        if usage.get('gpu_temp'):
            print(f"  GPU温度: {usage.get('gpu_temp')}°C")
    else:
        print(f"  GPU: 未使用（CPU処理中）")

def calculate_sharpness(image_gray):
    """シャープネス（鮮鋭度）計算 - ラプラシアン分散法"""
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    return laplacian.var()

def calculate_contrast(image_gray):
    """コントラスト計算 - RMS対比"""
    return image_gray.std()

def calculate_entropy(image_gray):
    """エントロピー計算 - 情報量の指標"""
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_lpips(img1_rgb, img2_rgb):
    """
    LPIPS（知覚的類似度）計算

    Returns:
        float: LPIPS距離（0に近いほど知覚的に類似）
    """
    if not LPIPS_AVAILABLE:
        return None, None

    try:
        # 警告を抑制
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

        # LPIPSモデルをロード（AlexNetベース）
        loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

        # 評価モードに設定（推論用）
        loss_fn.eval()

        # 画像をPyTorchテンソルに変換 [0-255] -> [-1, 1]
        def to_tensor(img):
            # RGB -> PyTorchの順序 (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            # [0, 255] -> [-1, 1]
            img_tensor = (img_tensor / 127.5) - 1.0
            # バッチ次元を追加してGPU/CPUに転送
            return img_tensor.unsqueeze(0).to(DEVICE)

        img1_tensor = to_tensor(img1_rgb)
        img2_tensor = to_tensor(img2_rgb)

        # GPU使用率取得（GPUの場合）
        gpu_usage = None
        if GPU_AVAILABLE:
            torch.cuda.synchronize()
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100

        # LPIPS距離を計算
        with torch.no_grad():
            distance = loss_fn(img1_tensor, img2_tensor)

        return float(distance.item()), gpu_usage

    except Exception as e:
        print(f"LPIPS計算エラー: {e}")
        return None, None

def is_document_image(img_rgb):
    """
    画像が文書/テキスト主体の画像かどうかを判定

    医療カルテ、レシート、スキャン文書などはCLIPが苦手とするため検出する

    Args:
        img_rgb: RGB画像 (numpy array)

    Returns:
        bool: 文書画像と判定された場合True
    """
    try:
        # 1. 明るい背景率の計算（文書は明るい背景が多い）
        # RGB平均が200以上のピクセルを「明るい背景」とみなす（医療カルテ対応）
        bright_pixels = np.sum(np.mean(img_rgb, axis=2) >= 200)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        bright_ratio = bright_pixels / total_pixels

        # 2. 非常に明るいピクセル（白に近い）
        white_pixels = np.sum(np.all(img_rgb >= 230, axis=2))
        white_ratio = white_pixels / total_pixels

        # 3. 色分散の計算（文書は色のバリエーションが少ない）
        color_std = np.std(img_rgb)

        # 4. グレースケール率（文書は白黒が多い）
        # RGBの差が小さいピクセルを「グレースケール」とみなす
        rgb_diff = np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)
        gray_pixels = np.sum(rgb_diff < 40)
        gray_ratio = gray_pixels / total_pixels

        # 5. LAB色空間でのL値（明度）が高い
        lab_l_mean = np.mean(img_rgb)  # 簡易的な明度指標

        # 判定基準（医療カルテに最適化）:
        # - 明るい背景率 > 50% AND (色分散 < 60 OR グレー率 > 70%) → 文書
        # - 非常に明るい背景率 > 30% AND グレー率 > 60% → 文書
        # - 平均輝度 > 200 AND 色分散 < 70 → 文書（医療カルテ特化）
        is_document = (bright_ratio > 0.50 and (color_std < 60 or gray_ratio > 0.70)) or \
                     (white_ratio > 0.30 and gray_ratio > 0.60) or \
                     (lab_l_mean > 200 and color_std < 70)

        # デバッグ用：判定情報を常に表示
        print(f"    文書判定 - 明背景: {bright_ratio*100:.1f}%, 白背景: {white_ratio*100:.1f}%, 色分散: {color_std:.1f}, グレー率: {gray_ratio*100:.1f}%, 平均輝度: {lab_l_mean:.1f} → {'📄文書' if is_document else '🖼️自然画像'}")

        return is_document

    except Exception as e:
        print(f"文書判定エラー: {e}")
        return False

def calculate_clip_similarity(img1_rgb, img2_rgb):
    """
    CLIP Embeddings を使用した意味的類似度計算

    Returns:
        float: コサイン類似度（1.0に近いほど意味的に類似、-1.0〜1.0の範囲）
    """
    global CLIP_MODEL, CLIP_PROCESSOR

    if not CLIP_AVAILABLE:
        return None

    try:
        # 初回のみモデルとプロセッサを読み込み
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            print("CLIP モデルを読み込み中...")
            # safetensors形式でロード（PyTorch 2.6未満でも動作）
            CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
            CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # GPUが利用可能ならGPUに転送
            if GPU_AVAILABLE and DEVICE:
                CLIP_MODEL = CLIP_MODEL.to(DEVICE)

            # 評価モードに設定
            CLIP_MODEL.eval()
            print(f"CLIP モデル読み込み完了 (デバイス: {DEVICE if DEVICE else 'CPU'})")

        # RGB画像をPIL Imageに変換（CLIPProcessorが期待する形式）
        from PIL import Image as PILImage
        img1_pil = PILImage.fromarray(img1_rgb.astype('uint8'))
        img2_pil = PILImage.fromarray(img2_rgb.astype('uint8'))

        # 画像を前処理
        inputs1 = CLIP_PROCESSOR(images=img1_pil, return_tensors="pt")
        inputs2 = CLIP_PROCESSOR(images=img2_pil, return_tensors="pt")

        # GPUが利用可能ならGPUに転送
        if GPU_AVAILABLE and DEVICE:
            inputs1 = {k: v.to(DEVICE) for k, v in inputs1.items()}
            inputs2 = {k: v.to(DEVICE) for k, v in inputs2.items()}

        # Embedding抽出
        with torch.no_grad():
            image_features1 = CLIP_MODEL.get_image_features(**inputs1)
            image_features2 = CLIP_MODEL.get_image_features(**inputs2)

        # L2正規化
        image_features1 = image_features1 / image_features1.norm(p=2, dim=-1, keepdim=True)
        image_features2 = image_features2 / image_features2.norm(p=2, dim=-1, keepdim=True)

        # コサイン類似度を計算（内積）
        cosine_similarity = (image_features1 @ image_features2.T).item()

        return cosine_similarity

    except Exception as e:
        print(f"CLIP類似度計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_ssim_gpu(img1_rgb, img2_rgb):
    """
    GPU対応SSIM計算（PyTorch使用）

    Returns:
        float: SSIM値（1.0に近いほど類似）
    """
    if not LPIPS_AVAILABLE or torch is None:
        # PyTorchがない場合はCPU版にフォールバック
        return ssim(img1_rgb, img2_rgb, channel_axis=2)

    try:
        # pytorch-msssimが利用可能ならそれを使用
        if MS_SSIM_AVAILABLE:
            from pytorch_msssim import ssim as pytorch_ssim

            def to_tensor(img):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                img_tensor = img_tensor / 255.0
                return img_tensor.unsqueeze(0).to(DEVICE)

            img1_tensor = to_tensor(img1_rgb)
            img2_tensor = to_tensor(img2_rgb)

            with torch.no_grad():
                ssim_val = pytorch_ssim(img1_tensor, img2_tensor, data_range=1.0)
            return float(ssim_val.item())

        # pytorch-msssimがない場合は独自GPU実装
        else:
            def to_tensor(img):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                img_tensor = img_tensor / 255.0
                return img_tensor.unsqueeze(0).to(DEVICE)

            img1_tensor = to_tensor(img1_rgb)
            img2_tensor = to_tensor(img2_rgb)

            # 簡易SSIM計算（GPU上で平均と分散を計算）
            with torch.no_grad():
                mu1 = F.avg_pool2d(img1_tensor, 11, 1, 5)
                mu2 = F.avg_pool2d(img2_tensor, 11, 1, 5)

                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2

                sigma1_sq = F.avg_pool2d(img1_tensor ** 2, 11, 1, 5) - mu1_sq
                sigma2_sq = F.avg_pool2d(img2_tensor ** 2, 11, 1, 5) - mu2_sq
                sigma12 = F.avg_pool2d(img1_tensor * img2_tensor, 11, 1, 5) - mu1_mu2

                C1 = 0.01 ** 2
                C2 = 0.03 ** 2

                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                          ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

                ssim_val = torch.mean(ssim_map)

            return float(ssim_val.item())

    except Exception as e:
        print(f"GPU SSIM計算エラー（CPU版にフォールバック）: {e}")
        return ssim(img1_rgb, img2_rgb, channel_axis=2)

def calculate_psnr_gpu(img1_rgb, img2_rgb):
    """
    GPU対応PSNR計算（PyTorch使用）

    Returns:
        float: PSNR値（高いほど類似）
    """
    if not LPIPS_AVAILABLE or torch is None:
        # PyTorchがない場合はCPU版にフォールバック
        return psnr(img1_rgb, img2_rgb)

    try:
        # 画像をPyTorchテンソルに変換
        def to_tensor(img):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_tensor = img_tensor / 255.0
            return img_tensor.unsqueeze(0).to(DEVICE)

        img1_tensor = to_tensor(img1_rgb)
        img2_tensor = to_tensor(img2_rgb)

        # MSE計算
        with torch.no_grad():
            mse = F.mse_loss(img1_tensor, img2_tensor)

            if mse == 0:
                return float('inf')

            # PSNR計算
            psnr_val = 10 * torch.log10(1.0 / mse)

        return float(psnr_val.item())

    except Exception as e:
        print(f"GPU PSNR計算エラー（CPU版にフォールバック）: {e}")
        return psnr(img1_rgb, img2_rgb)

def calculate_sharpness_gpu(img_gray):
    """
    GPU対応シャープネス計算（Kornia Laplacian）

    Returns:
        float: シャープネス値（高いほど鮮明）
    """
    if not KORNIA_AVAILABLE or torch is None:
        # Korniaがない場合はCPU版にフォールバック
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        return laplacian.var()

    try:
        # 画像をPyTorchテンソルに変換
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        # Korniaのラプラシアン
        with torch.no_grad():
            laplacian = KF.laplacian(img_tensor, kernel_size=3)
            variance = torch.var(laplacian)

        return float(variance.item()) * 255 * 255  # スケール調整

    except Exception as e:
        print(f"GPU シャープネス計算エラー（CPU版にフォールバック）: {e}")
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        return laplacian.var()

def estimate_noise_gpu(img_gray):
    """
    GPU対応ノイズ推定（高周波成分のエネルギー）

    Returns:
        float: ノイズレベル推定値
    """
    if not LPIPS_AVAILABLE or torch is None:
        # CPU版にフォールバック
        H = cv2.dct(np.float32(img_gray) / 255.0)
        noise_level = np.sum(np.abs(H[int(H.shape[0]*0.9):, int(H.shape[1]*0.9):]))
        return noise_level

    try:
        # 画像をPyTorchテンソルに変換
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        # 高周波フィルタ（簡易版）
        high_pass_kernel = torch.tensor([[-1, -1, -1],
                                         [-1,  8, -1],
                                         [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            high_freq = F.conv2d(img_tensor, high_pass_kernel, padding=1)
            noise_level = torch.mean(torch.abs(high_freq))

        return float(noise_level.item()) * 100  # スケール調整

    except Exception as e:
        print(f"GPU ノイズ推定エラー（CPU版にフォールバック）: {e}")
        H = cv2.dct(np.float32(img_gray) / 255.0)
        noise_level = np.sum(np.abs(H[int(H.shape[0]*0.9):, int(H.shape[1]*0.9):]))
        return noise_level

def detect_edges_gpu(img_gray):
    """
    GPU対応エッジ検出（Kornia Sobel）

    Returns:
        float: エッジ密度
    """
    if not KORNIA_AVAILABLE or torch is None:
        # CPU版にフォールバック
        edges = cv2.Canny(img_gray, 100, 200)
        return np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255) * 100

    try:
        # 画像をPyTorchテンソルに変換
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        # Korniaの Sobel
        with torch.no_grad():
            edges = KF.sobel(img_tensor)
            edge_density = torch.sum(edges > 0.1) / edges.numel() * 100

        return float(edge_density.item())

    except Exception as e:
        print(f"GPU エッジ検出エラー（CPU版にフォールバック）: {e}")
        edges = cv2.Canny(img_gray, 100, 200)
        return np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255) * 100

def calculate_color_difference_gpu(img1_rgb, img2_rgb):
    """
    GPU対応LAB色空間での色差計算（Kornia）

    Returns:
        float: 平均色差（ΔE）
    """
    if not KORNIA_AVAILABLE or torch is None:
        # CPU版にフォールバック
        img1_lab = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2LAB)
        img2_lab = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2LAB)
        delta_e = np.sqrt(np.sum((img1_lab.astype(float) - img2_lab.astype(float)) ** 2, axis=2))
        return np.mean(delta_e)

    try:
        # 画像をPyTorchテンソルに変換
        def to_tensor(img):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            return img_tensor.unsqueeze(0).to(DEVICE) / 255.0

        img1_tensor = to_tensor(img1_rgb)
        img2_tensor = to_tensor(img2_rgb)

        # KorniaのLAB変換
        with torch.no_grad():
            img1_lab = KC.rgb_to_lab(img1_tensor)
            img2_lab = KC.rgb_to_lab(img2_tensor)
            delta_e = torch.sqrt(torch.sum((img1_lab - img2_lab) ** 2, dim=1))
            mean_delta_e = torch.mean(delta_e)

        return float(mean_delta_e.item())

    except Exception as e:
        print(f"GPU 色差計算エラー（CPU版にフォールバック）: {e}")
        img1_lab = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2LAB)
        img2_lab = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2LAB)
        delta_e = np.sqrt(np.sum((img1_lab.astype(float) - img2_lab.astype(float)) ** 2, axis=2))
        return np.mean(delta_e)

def calculate_ms_ssim(img1_rgb, img2_rgb):
    """
    MS-SSIM（Multi-Scale SSIM）計算

    Returns:
        float: MS-SSIM値（1.0に近いほど類似）
    """
    if not MS_SSIM_AVAILABLE:
        return None

    try:
        # 画像をPyTorchテンソルに変換 [0-255] -> [0, 1]
        def to_tensor(img):
            # RGB -> PyTorchの順序 (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            # [0, 255] -> [0, 1]
            img_tensor = img_tensor / 255.0
            # バッチ次元を追加してGPU/CPUに転送
            return img_tensor.unsqueeze(0).to(DEVICE)

        img1_tensor = to_tensor(img1_rgb)
        img2_tensor = to_tensor(img2_rgb)

        # MS-SSIM計算（data_range=1.0で正規化済み画像用）
        with torch.no_grad():
            ms_ssim_val = pytorch_ms_ssim(img1_tensor, img2_tensor, data_range=1.0)

        return float(ms_ssim_val.item())

    except Exception as e:
        print(f"MS-SSIM計算エラー: {e}")
        return None

def analyze_local_quality(img1, img2, patch_size=64):
    """局所的な品質分析 - パッチ単位でSSIMを計算"""
    h, w = img1.shape[:2]
    ssim_map = []

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch1 = img1[y:y+patch_size, x:x+patch_size]
            patch2 = img2[y:y+patch_size, x:x+patch_size]

            if patch1.shape[:2] == (patch_size, patch_size):
                local_ssim = ssim(patch1, patch2, channel_axis=2)
                ssim_map.append(local_ssim)

    return np.array(ssim_map)

def detect_artifacts(image_gray):
    """アーティファクト検出 - ブロックノイズやリンギング"""
    # ブロックノイズ検出（8x8ブロック境界の不連続性）
    block_noise = 0
    h, w = image_gray.shape

    for y in range(8, h, 8):
        diff = np.abs(image_gray[y-1, :].astype(float) - image_gray[y, :].astype(float))
        block_noise += np.mean(diff)

    for x in range(8, w, 8):
        diff = np.abs(image_gray[:, x-1].astype(float) - image_gray[:, x].astype(float))
        block_noise += np.mean(diff)

    # リンギング検出（エッジ周辺の振動）
    edges = cv2.Canny(image_gray, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edge_region = cv2.dilate(edges, kernel, iterations=2)

    edge_pixels = image_gray[edge_region > 0]
    ringing = np.std(edge_pixels) if len(edge_pixels) > 0 else 0

    return block_noise, ringing

def analyze_color_distribution(img_rgb):
    """色分布の詳細分析（RGB, HSV, LAB）"""
    # 各チャンネルの統計
    stats_data = {}

    # RGB分析
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = img_rgb[:, :, i]
        stats_data[channel] = {
            'mean': float(np.mean(channel_data)),
            'std': float(np.std(channel_data)),
            'min': int(np.min(channel_data)),
            'max': int(np.max(channel_data)),
            'median': float(np.median(channel_data))
        }

    # HSV分析
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    stats_data['Saturation'] = {
        'mean': float(np.mean(saturation)),
        'std': float(np.std(saturation))
    }

    # LAB色空間分析（知覚的な色差評価用）
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    stats_data['LAB'] = {
        'L_mean': float(np.mean(lab[:, :, 0])),  # 明度
        'L_std': float(np.std(lab[:, :, 0])),
        'a_mean': float(np.mean(lab[:, :, 1])),  # 赤-緑
        'a_std': float(np.std(lab[:, :, 1])),
        'b_mean': float(np.mean(lab[:, :, 2])),  # 黄-青
        'b_std': float(np.std(lab[:, :, 2]))
    }

    return stats_data

def analyze_frequency_domain(img_gray):
    """周波数領域分析 - FFT"""
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # 高周波成分の割合
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(center_h, center_w) // 3

    y, x = np.ogrid[:h, :w]
    mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2

    low_freq_energy = np.sum(magnitude[mask])
    high_freq_energy = np.sum(magnitude[~mask])
    total_energy = low_freq_energy + high_freq_energy

    return {
        'low_freq_ratio': low_freq_energy / total_energy,
        'high_freq_ratio': high_freq_energy / total_energy,
        'total_energy': float(total_energy)
    }

def analyze_texture(img_gray):
    """テクスチャ分析（GPU対応 - Kornia Sobel）"""
    if not KORNIA_AVAILABLE or torch is None:
        # CPU版にフォールバック
        small = cv2.resize(img_gray, (img_gray.shape[1]//4, img_gray.shape[0]//4))
        sobel_x = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
        texture_complexity = np.sqrt(sobel_x**2 + sobel_y**2).mean()
        return {'texture_complexity': float(texture_complexity)}

    try:
        # GPU版
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
        # ダウンサンプリング
        img_small = F.interpolate(img_tensor, scale_factor=0.25, mode='bilinear', align_corners=False)

        with torch.no_grad():
            sobel_magnitude = KF.sobel(img_small)
            texture_complexity = torch.mean(sobel_magnitude)

        return {'texture_complexity': float(texture_complexity.item() * 255)}
    except Exception as e:
        print(f"GPU テクスチャ分析エラー（CPU版にフォールバック）: {e}")
        small = cv2.resize(img_gray, (img_gray.shape[1]//4, img_gray.shape[0]//4))
        sobel_x = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
        texture_complexity = np.sqrt(sobel_x**2 + sobel_y**2).mean()
        return {'texture_complexity': float(texture_complexity)}

def create_detailed_visualizations(img1_rgb, img2_rgb, img1_gray, img2_gray, output_dir):
    """詳細な可視化画像を生成"""
    fig = plt.figure(figsize=(20, 12))

    # 1. 元画像
    plt.subplot(3, 4, 1)
    plt.imshow(img1_rgb)
    plt.title('元画像 (Ground Truth)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(img2_rgb)
    plt.title('AI処理結果', fontsize=12, fontweight='bold')
    plt.axis('off')

    # 2. ヒストグラム
    plt.subplot(3, 4, 3)
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img1_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, linewidth=1.5)
    plt.title('ヒストグラム - 元画像', fontsize=11)
    plt.xlim([0, 256])
    plt.xlabel('輝度値', fontsize=9)
    plt.ylabel('ピクセル数', fontsize=9)

    plt.subplot(3, 4, 4)
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img2_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, linewidth=1.5)
    plt.title('ヒストグラム - AI処理結果', fontsize=11)
    plt.xlim([0, 256])
    plt.xlabel('輝度値', fontsize=9)
    plt.ylabel('ピクセル数', fontsize=9)

    # 3. エッジ検出
    edges1 = cv2.Canny(img1_gray, 100, 200)
    edges2 = cv2.Canny(img2_gray, 100, 200)

    plt.subplot(3, 4, 5)
    plt.imshow(edges1, cmap='gray')
    plt.title('エッジ検出 - 元画像', fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(edges2, cmap='gray')
    plt.title('エッジ検出 - AI処理結果', fontsize=11)
    plt.axis('off')

    # 4. 差分
    diff = cv2.absdiff(img1_rgb, img2_rgb)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    plt.subplot(3, 4, 7)
    plt.imshow(diff)
    plt.title('絶対差分', fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(diff_gray, cmap='hot')
    plt.title('差分ヒートマップ', fontsize=11)
    plt.axis('off')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    # 5. FFT（周波数領域）
    f1 = np.fft.fft2(img1_gray)
    f2 = np.fft.fft2(img2_gray)

    magnitude1 = np.log(np.abs(np.fft.fftshift(f1)) + 1)
    magnitude2 = np.log(np.abs(np.fft.fftshift(f2)) + 1)

    plt.subplot(3, 4, 9)
    plt.imshow(magnitude1, cmap='gray')
    plt.title('周波数スペクトル - 元画像', fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(magnitude2, cmap='gray')
    plt.title('周波数スペクトル - AI処理結果', fontsize=11)
    plt.axis('off')

    # 6. シャープネス可視化（ラプラシアン）
    lap1 = cv2.Laplacian(img1_gray, cv2.CV_64F)
    lap2 = cv2.Laplacian(img2_gray, cv2.CV_64F)

    plt.subplot(3, 4, 11)
    im1 = plt.imshow(np.abs(lap1), cmap='viridis')
    plt.title('シャープネスマップ - 元画像', fontsize=11)
    plt.axis('off')
    cb1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=8)

    plt.subplot(3, 4, 12)
    im2 = plt.imshow(np.abs(lap2), cmap='viridis')
    plt.title('シャープネスマップ - AI処理結果', fontsize=11)
    plt.axis('off')
    cb2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detailed_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_report(results, img1_name, img2_name, output_dir):
    """比較レポート画像を生成"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('画像比較分析レポート', fontsize=20, fontweight='bold', y=0.98)

    # スコア表示（両画像比較）
    ax1 = plt.subplot(2, 3, 1)
    breakdown = results['total_score']['breakdown']

    categories = ['シャープネス', 'コントラスト', 'エントロピー', 'ノイズ対策', 'エッジ保持', '歪み抑制', 'テクスチャ']
    img1_values = [breakdown['img1']['sharpness'], breakdown['img1']['contrast'],
                   breakdown['img1']['entropy'], breakdown['img1']['noise'],
                   breakdown['img1']['edge'], breakdown['img1']['artifact'],
                   breakdown['img1']['texture']]
    img2_values = [breakdown['img2']['sharpness'], breakdown['img2']['contrast'],
                   breakdown['img2']['entropy'], breakdown['img2']['noise'],
                   breakdown['img2']['edge'], breakdown['img2']['artifact'],
                   breakdown['img2']['texture']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.barh(x - width/2, img1_values, width, label='元画像', color='#3498db')
    bars2 = ax1.barh(x + width/2, img2_values, width, label='AI処理結果', color='#e74c3c')

    ax1.set_yticks(x)
    ax1.set_yticklabels(categories)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('スコア', fontsize=11, fontweight='bold')
    ax1.set_title('項目別スコア比較', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # 総合スコア
    ax2 = plt.subplot(2, 3, 2)
    total_score = results['total_score']['img2']
    img1_score = results['total_score']['img1']

    ax2.barh(['元画像 (基準)', 'AI処理結果'], [img1_score, total_score],
             color=['#3498db', '#e74c3c' if total_score < 70 else '#f39c12' if total_score < 90 else '#2ecc71'])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('総合スコア', fontsize=11, fontweight='bold')
    ax2.set_title('総合評価', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3)

    for i, (score, name) in enumerate(zip([img1_score, total_score], ['元画像', 'AI処理結果'])):
        ax2.text(score + 2, i, f'{score:.1f}', va='center', fontsize=12, fontweight='bold')

    # 主要指標
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')

    delta_e_value = results['color_distribution'].get('delta_e', 0)

    # SSIM/PSNR/delta_eは元画像の有無で形式が異なる
    ssim_data = results['ssim']
    if isinstance(ssim_data, dict):
        ssim_display = f"元画像: {ssim_data['img1_vs_original']:.4f}\n  AI処理結果: {ssim_data['img2_vs_original']:.4f}"
    else:
        ssim_display = f"{ssim_data:.4f}"

    psnr_data = results['psnr']
    if isinstance(psnr_data, dict):
        psnr_display = f"元画像: {psnr_data['img1_vs_original']:.2f} dB\n  AI処理結果: {psnr_data['img2_vs_original']:.2f} dB"
    else:
        psnr_display = f"{psnr_data:.2f} dB"

    if isinstance(delta_e_value, dict):
        delta_e_display = f"元画像: {delta_e_value['img1_vs_original']:.2f}\n  AI処理結果: {delta_e_value['img2_vs_original']:.2f}"
    else:
        delta_e_display = f"{delta_e_value:.2f}"

    info_text = f"""
【主要指標】

SSIM: {ssim_display}
  (1.0 = 完全一致)

PSNR: {psnr_display}
  (30dB以上で視覚的に同等)

シャープネス:
  元画像: {results['sharpness']['img1']:.2f}
  AI処理結果: {results['sharpness']['img2']:.2f}
  差: {results['sharpness']['difference_pct']:+.1f}%

色差 (ΔE): {delta_e_display}
  (< 5: 許容, > 10: 明確な違い)
    """

    ax3.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax3.set_title('詳細データ', fontsize=13, fontweight='bold', pad=10)

    # エッジ比較
    ax4 = plt.subplot(2, 3, 4)
    edge_data = [results['edges']['img1_density'], results['edges']['img2_density']]
    ax4.bar(['元画像', 'AI処理結果'], edge_data, color=['#3498db', '#9b59b6'])
    ax4.set_ylabel('エッジ密度 (%)', fontsize=11, fontweight='bold')
    ax4.set_title('エッジ保持率', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3)

    for i, val in enumerate(edge_data):
        ax4.text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ノイズ・アーティファクト
    ax5 = plt.subplot(2, 3, 5)
    noise_data = [results['noise']['img1'], results['noise']['img2']]
    artifact1 = results['artifacts']['img1_block_noise'] + results['artifacts']['img1_ringing']
    artifact2 = results['artifacts']['img2_block_noise'] + results['artifacts']['img2_ringing']

    x = np.arange(2)
    width = 0.35

    ax5.bar(x - width/2, noise_data, width, label='ノイズ', color='#e67e22')
    ax5.bar(x + width/2, [artifact1, artifact2], width, label='アーティファクト', color='#c0392b')

    ax5.set_ylabel('値 (低い方が良い)', fontsize=11, fontweight='bold')
    ax5.set_title('ノイズとアーティファクト', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['元画像', 'AI処理結果'])
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)

    # 周波数分析
    ax6 = plt.subplot(2, 3, 6)
    freq1 = [results['frequency_analysis']['img1']['low_freq_ratio'] * 100,
             results['frequency_analysis']['img1']['high_freq_ratio'] * 100]
    freq2 = [results['frequency_analysis']['img2']['low_freq_ratio'] * 100,
             results['frequency_analysis']['img2']['high_freq_ratio'] * 100]

    x = np.arange(2)
    width = 0.35

    ax6.bar(x - width/2, freq1, width, label='元画像', color='#3498db')
    ax6.bar(x + width/2, freq2, width, label='AI処理結果', color='#9b59b6')

    ax6.set_ylabel('比率 (%)', fontsize=11, fontweight='bold')
    ax6.set_title('周波数成分分布', fontsize=13, fontweight='bold', pad=10)
    ax6.set_xticks(x)
    ax6.set_xticklabels(['低周波', '高周波'])
    ax6.legend(fontsize=10)
    ax6.set_ylim(0, 100)
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    report_path = os.path.join(output_dir, 'comparison_report.png')
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()

    return report_path

def imread_unicode(filename):
    """日本語パスに対応した画像読み込み（透明背景対応）"""
    try:
        from PIL import Image
        pil_image = Image.open(filename)

        # 透明背景（RGBA）の場合、白背景で合成
        if pil_image.mode == 'RGBA':
            print(f"  透明背景を検出: {filename}")
            print(f"  白背景で合成します")
            # 白背景を作成
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            # アルファチャンネルを使って合成
            background.paste(pil_image, mask=pil_image.split()[3])  # 3番目はアルファチャンネル
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # numpy配列に変換
        img_array = np.array(pil_image)
        # RGB -> BGR (OpenCV形式)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return None

def analyze_images(img1_path, img2_path, output_dir='analysis_results', original_path=None, evaluation_mode='image'):
    """
    元画像とAI処理結果を詳細に比較分析する（精度評価）

    Parameters:
    img1_path: 元画像のパス（Ground Truth / 基準画像）
    img2_path: AI処理結果のパス（超解像画像など）
    output_dir: 結果保存ディレクトリ
    original_path: 使用しない（後方互換性のため残す）
    evaluation_mode: 評価モード ('image', 'document', 'developer', 'academic')

    Returns:
    results: 分析結果の辞書
    """

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 画像読み込み（日本語パス対応）
    img1 = imread_unicode(img1_path)  # 元画像（Ground Truth）
    img2 = imread_unicode(img2_path)  # AI処理結果

    # エイリアス（コード全体で統一的に使用）
    img_original = img1  # 元画像
    img_ai_result = img2  # AI処理結果

    if img1 is None or img2 is None:
        print("エラー: 画像ファイルが読み込めません")
        print(f"元画像パス: {img1_path}")
        print(f"AI処理結果パス: {img2_path}")
        return

    # 画像サイズチェックと調整
    if img1.shape != img2.shape:
        print(f"\n画像サイズが異なります:")
        print(f"  元画像: {img1.shape[1]} x {img1.shape[0]} px")
        print(f"  AI処理結果: {img2.shape[1]} x {img2.shape[0]} px")
        print(f"AI処理結果を元画像のサイズにリサイズします...\n")

        # AI処理結果を元画像のサイズに合わせる
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        img_ai_result = img2  # エイリアスも更新


    # 分析パターンの表示
    if evaluation_mode == "academic":
        print("\n" + "=" * 80)
        print("【分析パターン】学術評価モード（Academic Evaluation）")
        print("=" * 80)
        print("📚 標準ベンチマーク方式: ×2 Scale Super-Resolution")
        print(f"   Ground Truth（元画像）: {img1.shape[1]}x{img1.shape[0]}px")
        print(f"   AI処理結果: {img2.shape[1]}x{img2.shape[0]}px")
        print("   比較対象: DIV2K, Set5, Set14等との定量比較")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("【分析パターン】精度評価モード（元画像基準）")
        print("=" * 80)
        print("📌 用途: AI超解像、画質改善、ノイズ除去等の精度評価")
        print(f"   元画像: {img1.shape[1]}x{img1.shape[0]}px")
        print(f"   AI処理結果: {img2.shape[1]}x{img2.shape[0]}px")
        print("=" * 80)

    # RGB/グレースケール変換（元画像）
    img_original_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_original_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # RGB変換（OpenCVはBGRなので）
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # グレースケール変換
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 結果を保存する辞書
    results = {
        'timestamp': datetime.now().isoformat(),
        'image1_path': img1_path,
        'image2_path': img2_path,
        'original_path': original_path,
        'has_original': img_original_rgb is not None
    }

    print("=" * 80)
    print("詳細画像比較分析レポート")
    print("=" * 80)
    print("📌 比較対象: 元画像（処理前/Before） vs AI超解像結果（処理後/After）")
    print("=" * 80)

    # 1. 基本情報
    print("\n【1. 基本情報】")
    print(f"超解像結果1サイズ: {img1.shape[1]} x {img1.shape[0]} px")
    print(f"超解像結果2サイズ: {img2.shape[1]} x {img2.shape[0]} px")

    size1 = os.path.getsize(img1_path) / (1024 * 1024)
    size2 = os.path.getsize(img2_path) / (1024 * 1024)
    print(f"超解像結果1ファイルサイズ: {size1:.2f} MB")
    print(f"超解像結果2ファイルサイズ: {size2:.2f} MB")
    print(f"サイズ差: {abs(size1 - size2):.2f} MB ({((size2/size1 - 1) * 100):+.1f}%)")

    if img_original is not None:
        size_original = os.path.getsize(original_path) / (1024 * 1024)
        print(f"元画像（処理前）ファイルサイズ: {size_original:.2f} MB")

    # GPU/CPU情報
    print(f"\n計算デバイス情報:")
    if LPIPS_AVAILABLE:
        if GPU_AVAILABLE:
            print(f"  GPU: {GPU_NAME}")
            print(f"  CUDA利用可能: はい")
            print(f"  VRAMサイズ: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print(f"  GPU: なし（CPU使用）")
            print(f"  CUDA利用可能: いいえ")
    else:
        print(f"  PyTorch未インストール（GPU機能無効）")

    results['basic_info'] = {
        'img1_size': [int(img1.shape[1]), int(img1.shape[0])],
        'img2_size': [int(img2.shape[1]), int(img2.shape[0])],
        'img1_filesize_mb': round(size1, 2),
        'img2_filesize_mb': round(size2, 2),
        'gpu_available': GPU_AVAILABLE,
        'gpu_name': GPU_NAME,
        'device': str(DEVICE) if DEVICE else 'N/A'
    }

    # 2. 構造類似性（SSIM）
    print("\n【2. 構造類似性（SSIM）】")
    print("1.0 = 完全一致、0.0 = 全く違う")
    if GPU_AVAILABLE:
        print(f"[GPU処理] デバイス: {DEVICE}")
    print_usage_status("SSIM計算開始（GPU使用）" if GPU_AVAILABLE else "SSIM計算開始（CPU使用）")

    if img_original_rgb is not None:
        # 元画像がある場合：それぞれ元画像との類似度を計算
        ssim_img1_vs_orig = calculate_ssim_gpu(img1_rgb, img_original_rgb)
        ssim_img2_vs_orig = calculate_ssim_gpu(img2_rgb, img_original_rgb)
        print(f"画像1 vs 元画像 SSIM: {ssim_img1_vs_orig:.4f}")
        print(f"画像2 vs 元画像 SSIM: {ssim_img2_vs_orig:.4f}")
        if ssim_img1_vs_orig > ssim_img2_vs_orig:
            print(f"→ 元画像の方が元画像に近い (+{(ssim_img1_vs_orig - ssim_img2_vs_orig):.4f})")
        else:
            print(f"→ AI処理結果の方が元画像に近い (+{(ssim_img2_vs_orig - ssim_img1_vs_orig):.4f})")
        results['ssim'] = {
            'img1_vs_original': round(ssim_img1_vs_orig, 4),
            'img2_vs_original': round(ssim_img2_vs_orig, 4)
        }
    else:
        # 元画像がない場合：元画像 vs AI処理結果
        ssim_score = calculate_ssim_gpu(img1_rgb, img2_rgb)
        print(f"SSIM (元画像 vs AI処理結果): {ssim_score:.4f}")
        results['ssim'] = round(ssim_score, 4)

    # 2.5. MS-SSIM（Multi-Scale SSIM）
    print("\n【2.5. MS-SSIM（マルチスケールSSIM）】")
    print("複数スケールでの構造類似性（1.0に近いほど類似）")
    print_usage_status("MS-SSIM計算開始")
    ms_ssim_score = calculate_ms_ssim(img1_rgb, img2_rgb)

    if ms_ssim_score is not None:
        print(f"MS-SSIM: {ms_ssim_score:.4f}")
        if ms_ssim_score >= 0.99:
            print("  評価: ほぼ完全に一致")
        elif ms_ssim_score >= 0.95:
            print("  評価: 非常に類似")
        elif ms_ssim_score >= 0.90:
            print("  評価: 類似")
        elif ms_ssim_score >= 0.80:
            print("  評価: やや類似")
        else:
            print("  評価: 異なる")
        results['ms_ssim'] = round(ms_ssim_score, 4)
    else:
        print("  ※MS-SSIM計算をスキップしました（ライブラリ未インストール）")
        results['ms_ssim'] = None

    # 3. PSNR
    print("\n【3. PSNR（ピーク信号対雑音比）】")
    print("数値が高いほど類似（30dB以上で視覚的にほぼ同一）")
    print_usage_status("PSNR計算開始（GPU使用）" if GPU_AVAILABLE else "PSNR計算開始（CPU使用）")

    if img_original_rgb is not None:
        # 元画像がある場合：それぞれ元画像とのPSNRを計算
        psnr_img1_vs_orig = calculate_psnr_gpu(img1_rgb, img_original_rgb)
        psnr_img2_vs_orig = calculate_psnr_gpu(img2_rgb, img_original_rgb)
        print(f"画像1 vs 元画像 PSNR: {psnr_img1_vs_orig:.2f} dB")
        print(f"画像2 vs 元画像 PSNR: {psnr_img2_vs_orig:.2f} dB")
        if psnr_img1_vs_orig > psnr_img2_vs_orig:
            print(f"→ 元画像の方が元画像に近い (+{(psnr_img1_vs_orig - psnr_img2_vs_orig):.2f} dB)")
        else:
            print(f"→ AI処理結果の方が元画像に近い (+{(psnr_img2_vs_orig - psnr_img1_vs_orig):.2f} dB)")
        results['psnr'] = {
            'img1_vs_original': round(psnr_img1_vs_orig, 2),
            'img2_vs_original': round(psnr_img2_vs_orig, 2)
        }
    else:
        # 元画像がない場合：元画像 vs AI処理結果
        psnr_score = calculate_psnr_gpu(img1_rgb, img2_rgb)
        print(f"PSNR (元画像 vs AI処理結果): {psnr_score:.2f} dB")
        results['psnr'] = round(psnr_score, 2)

    # 3.4. ピクセル差分（MAE - 平均絶対誤差）
    print("\n【3.4. ピクセル差分（MAE）】")
    print("元画像とのピクセル単位での絶対差分（低いほど近い、0=完全一致）")

    if img_original_rgb is not None:
        # 元画像がある場合：それぞれ元画像との差分を計算
        diff_img1 = np.abs(img1_rgb.astype(float) - img_original_rgb.astype(float))
        diff_img2 = np.abs(img2_rgb.astype(float) - img_original_rgb.astype(float))

        # 全体のMAE
        mae_img1 = np.mean(diff_img1)
        mae_img2 = np.mean(diff_img2)

        print(f"📊 全体MAE:")
        print(f"  超解像結果1（After） vs 元画像（Before）: {mae_img1:.2f} (差分率: {(mae_img1/255)*100:.1f}%)")
        print(f"  超解像結果2（After） vs 元画像（Before）: {mae_img2:.2f} (差分率: {(mae_img2/255)*100:.1f}%)")

        # テキスト領域のみのMAE（白背景を除外）
        # RGB平均が200未満のピクセルを「テキスト領域」とみなす
        text_mask_img1 = np.mean(img1_rgb, axis=2) < 200
        text_mask_img2 = np.mean(img2_rgb, axis=2) < 200
        text_mask_original = np.mean(img_original_rgb, axis=2) < 200
        # 3つの画像のいずれかにテキストがある領域を統合
        text_mask_combined = text_mask_img1 | text_mask_img2 | text_mask_original

        # テキスト領域のピクセル数を確認
        text_pixel_count = np.sum(text_mask_combined)
        total_pixel_count = text_mask_combined.size
        text_ratio = text_pixel_count / total_pixel_count

        if text_pixel_count > 0:
            # テキスト領域が存在する場合
            mae_text_img1 = np.mean(diff_img1[text_mask_combined])
            mae_text_img2 = np.mean(diff_img2[text_mask_combined])

            print(f"\n📝 テキスト領域MAE（白背景除外、{text_ratio*100:.1f}%の領域）:")
            print(f"  超解像結果1（After） vs 元画像（Before）: {mae_text_img1:.2f} (差分率: {(mae_text_img1/255)*100:.1f}%)")
            print(f"  超解像結果2（After） vs 元画像（Before）: {mae_text_img2:.2f} (差分率: {(mae_text_img2/255)*100:.1f}%)")
        else:
            # テキスト領域がない場合（純粋な白画像など）
            mae_text_img1 = None
            mae_text_img2 = None
            print(f"\n  ⚠️  テキスト領域が検出されませんでした（白背景のみの画像）")

        # 全体MAEでの比較
        print(f"\n💡 全体MAE比較:")
        if mae_img1 < mae_img2:
            print(f"  → 超解像結果1の方が元画像（Before）に近い (差分差: {mae_img2 - mae_img1:.2f})")
        else:
            print(f"  → 超解像結果2の方が元画像（Before）に近い (差分差: {mae_img1 - mae_img2:.2f})")

        # テキストMAEでの比較と評価（より重要）
        if mae_text_img1 is not None and mae_text_img2 is not None:
            print(f"\n🎯 テキスト領域MAE比較（重要度：高）:")
            if mae_text_img1 < mae_text_img2:
                print(f"  → 超解像結果1の方が元画像（Before）に近い (差分差: {mae_text_img2 - mae_text_img1:.2f})")
            else:
                print(f"  → 超解像結果2の方が元画像（Before）に近い (差分差: {mae_text_img1 - mae_text_img2:.2f})")

            # テキスト領域での評価基準（より厳格）
            print(f"\n  【テキスト領域の評価】")
            for idx, mae_text_val in enumerate([mae_text_img1, mae_text_img2], 1):
                if mae_text_val < 10:
                    eval_str = "ほぼ完全一致（同じ内容）✅"
                elif mae_text_val < 30:
                    eval_str = "類似（一部異なる可能性）"
                elif mae_text_val < 60:
                    eval_str = "⚠️ 明らかに異なる内容"
                else:
                    eval_str = "🚨 全く異なる画像（別の文書/別の患者）"
                print(f"  超解像結果{idx}: {eval_str}")

        results['mae'] = {
            'img1_vs_original': round(mae_img1, 2),
            'img2_vs_original': round(mae_img2, 2),
            'img1_diff_ratio': round((mae_img1/255)*100, 2),
            'img2_diff_ratio': round((mae_img2/255)*100, 2),
            'img1_text_mae': round(mae_text_img1, 2) if mae_text_img1 is not None else None,
            'img2_text_mae': round(mae_text_img2, 2) if mae_text_img2 is not None else None,
            'text_region_ratio': round(text_ratio * 100, 2) if text_pixel_count > 0 else 0
        }
    else:
        # 元画像がない場合：元画像 vs AI処理結果
        mae_score = np.mean(np.abs(img1_rgb.astype(float) - img2_rgb.astype(float)))
        print(f"MAE (元画像 vs AI処理結果): {mae_score:.2f} (差分率: {(mae_score/255)*100:.1f}%)")

        if mae_score < 5:
            print("  評価: ほぼ完全一致")
        elif mae_score < 10:
            print("  評価: 非常に類似")
        elif mae_score < 20:
            print("  評価: 類似")
        elif mae_score < 40:
            print("  評価: やや異なる")
        else:
            print("  評価: 大きく異なる")

        results['mae'] = {
            'value': round(mae_score, 2),
            'diff_ratio': round((mae_score/255)*100, 2)
        }

    # 3.5. LPIPS（知覚的類似度）
    print("\n【3.5. LPIPS（知覚的類似度）】")
    print("深層学習ベースの知覚的類似度（0に近いほど類似）")
    print_usage_status("LPIPS計算開始")
    lpips_score, gpu_usage = calculate_lpips(img1_rgb, img2_rgb)
    print_usage_status("LPIPS計算完了")

    if lpips_score is not None:
        print(f"LPIPS: {lpips_score:.4f}")
        if GPU_AVAILABLE and gpu_usage is not None:
            print(f"  GPU使用: はい（メモリ使用率: {gpu_usage:.1f}%）")
        elif GPU_AVAILABLE:
            print(f"  GPU使用: はい")
        else:
            print(f"  GPU使用: いいえ（CPU処理）")

        if lpips_score < 0.1:
            print("  評価: 知覚的にほぼ同一")
        elif lpips_score < 0.3:
            print("  評価: 知覚的に類似")
        elif lpips_score < 0.5:
            print("  評価: やや異なる")
        else:
            print("  評価: 大きく異なる")
        results['lpips'] = round(lpips_score, 4)
    else:
        print("  ※LPIPS計算をスキップしました（ライブラリ未インストール）")
        results['lpips'] = None

    # 3.6. CLIP Embeddings（意味的類似度）
    print("\n【3.6. CLIP Embeddings（意味的類似度）】")
    print("OpenAI CLIP モデルによる意味的類似度（1.0に近いほど意味的に類似）")
    print_usage_status("CLIP計算開始")

    # 文書画像検出（CLIPが苦手とする画像タイプ）
    is_doc_img1 = is_document_image(img1_rgb)
    is_doc_img2 = is_document_image(img2_rgb)
    is_doc_original = is_document_image(img_original_rgb) if img_original_rgb is not None else False
    is_any_document_detected = is_doc_img1 or is_doc_img2 or is_doc_original

    # 評価モードを考慮
    if evaluation_mode == 'document':
        # 文書モード：強制的に文書として扱う
        is_any_document = True
        print("📄 評価モード: 文書モード（厳格な基準で評価）")
    elif evaluation_mode == 'developer':
        # 開発者モード：自動検出結果を使用
        is_any_document = is_any_document_detected
        print("🔧 評価モード: 開発者モード（参考情報として表示）")
    else:
        # 画像モード：自動検出結果を使用
        is_any_document = is_any_document_detected
        if is_any_document:
            print("📄 文書画像を自動検出（文書モードの使用を推奨）")

    if img_original_rgb is not None:
        # 元画像がある場合：それぞれ元画像との類似度を計算
        clip_img1_vs_orig = calculate_clip_similarity(img1_rgb, img_original_rgb)
        clip_img2_vs_orig = calculate_clip_similarity(img2_rgb, img_original_rgb)
        print_usage_status("CLIP計算完了")

        if clip_img1_vs_orig is not None and clip_img2_vs_orig is not None:
            print(f"画像1 vs 元画像 CLIP: {clip_img1_vs_orig:.4f}")
            print(f"画像2 vs 元画像 CLIP: {clip_img2_vs_orig:.4f}")
            if GPU_AVAILABLE:
                print(f"  GPU使用: はい")
            else:
                print(f"  GPU使用: いいえ（CPU処理）")

            if clip_img1_vs_orig > clip_img2_vs_orig:
                print(f"→ 元画像の方が元画像に意味的に近い (+{(clip_img1_vs_orig - clip_img2_vs_orig):.4f})")
            else:
                print(f"→ AI処理結果の方が元画像に意味的に近い (+{(clip_img2_vs_orig - clip_img1_vs_orig):.4f})")

            # 各画像の評価（文書画像の場合は厳格な基準を適用）
            if is_any_document:
                print("  ⚠️  文書/カルテ画像を検出: CLIPは厳格な基準で評価します")
                # 文書画像用の厳格な閾値
                for idx, clip_val in enumerate([clip_img1_vs_orig, clip_img2_vs_orig], 1):
                    if clip_val > 0.98:
                        eval_str = "意味的にほぼ同一"
                    elif clip_val > 0.95:
                        eval_str = "意味的に類似（要注意：文書は構造類似で高スコアになりやすい）"
                    elif clip_val > 0.90:
                        eval_str = "⚠️ 構造は類似だが内容は異なる可能性 🔍"
                    else:
                        eval_str = "全く異なる画像（内容が違う）🚨"
                    print(f"  画像{idx}: {eval_str}")
            else:
                # 自然画像用の通常閾値
                for idx, clip_val in enumerate([clip_img1_vs_orig, clip_img2_vs_orig], 1):
                    if clip_val > 0.95:
                        eval_str = "意味的にほぼ同一"
                    elif clip_val > 0.85:
                        eval_str = "意味的に非常に類似"
                    elif clip_val > 0.70:
                        eval_str = "意味的に類似"
                    elif clip_val > 0.50:
                        eval_str = "やや類似"
                    else:
                        eval_str = "全く異なる画像（内容が違う）🚨"
                    print(f"  画像{idx}: {eval_str}")

            results['clip_similarity'] = {
                'img1_vs_original': round(clip_img1_vs_orig, 4),
                'img2_vs_original': round(clip_img2_vs_orig, 4),
                'is_document': is_any_document  # 文書画像フラグを追加
            }
        else:
            print("  ※CLIP計算をスキップしました（ライブラリ未インストール）")
            results['clip_similarity'] = None
    else:
        # 元画像がない場合：元画像 vs AI処理結果
        clip_similarity = calculate_clip_similarity(img1_rgb, img2_rgb)
        print_usage_status("CLIP計算完了")

        if clip_similarity is not None:
            print(f"CLIP コサイン類似度: {clip_similarity:.4f}")
            if GPU_AVAILABLE:
                print(f"  GPU使用: はい")
            else:
                print(f"  GPU使用: いいえ（CPU処理）")

            # 文書画像の場合は厳格な基準を適用
            if is_any_document:
                print("  ⚠️  文書/カルテ画像を検出: CLIPは厳格な基準で評価します")
                if clip_similarity > 0.98:
                    print("  評価: 意味的にほぼ同一の画像")
                elif clip_similarity > 0.95:
                    print("  評価: 意味的に類似（要注意：文書は構造類似で高スコアになりやすい）")
                elif clip_similarity > 0.90:
                    print("  評価: ⚠️ 構造は類似だが内容は異なる可能性 🔍")
                else:
                    print("  評価: 全く異なる画像（内容が違う）")
            else:
                # 自然画像用の通常閾値
                if clip_similarity > 0.95:
                    print("  評価: 意味的にほぼ同一の画像")
                elif clip_similarity > 0.85:
                    print("  評価: 意味的に非常に類似")
                elif clip_similarity > 0.70:
                    print("  評価: 意味的に類似")
                elif clip_similarity > 0.50:
                    print("  評価: やや類似")
                else:
                    print("  評価: 全く異なる画像（内容が違う）")

            results['clip_similarity'] = {
                'value': round(clip_similarity, 4),
                'is_document': is_any_document
            }
        else:
            print("  ※CLIP計算をスキップしました（ライブラリ未インストール）")
            results['clip_similarity'] = None

    # 4. シャープネス（鮮鋭度）
    print("\n【4. シャープネス（鮮鋭度）】")
    print_usage_status("シャープネス計算開始（GPU使用）" if GPU_AVAILABLE else "シャープネス計算開始（CPU使用）")
    sharpness1 = calculate_sharpness_gpu(img1_gray)
    sharpness2 = calculate_sharpness_gpu(img2_gray)
    print(f"画像1シャープネス: {sharpness1:.2f}")
    print(f"画像2シャープネス: {sharpness2:.2f}")
    print(f"差: {abs(sharpness1 - sharpness2):.2f} ({((sharpness2/sharpness1 - 1) * 100):+.1f}%)")

    results['sharpness'] = {
        'img1': round(sharpness1, 2),
        'img2': round(sharpness2, 2),
        'difference_pct': round((sharpness2/sharpness1 - 1) * 100, 1)
    }

    # 5. コントラスト
    print("\n【5. コントラスト】")
    contrast1 = calculate_contrast(img1_gray)
    contrast2 = calculate_contrast(img2_gray)
    print(f"画像1コントラスト: {contrast1:.2f}")
    print(f"画像2コントラスト: {contrast2:.2f}")
    print(f"差: {abs(contrast1 - contrast2):.2f} ({((contrast2/contrast1 - 1) * 100):+.1f}%)")

    results['contrast'] = {
        'img1': round(contrast1, 2),
        'img2': round(contrast2, 2),
        'difference_pct': round((contrast2/contrast1 - 1) * 100, 1)
    }

    # 6. エントロピー（情報量）
    print("\n【6. エントロピー（情報量）】")
    print("数値が高いほど情報量が多い（複雑）")
    entropy1 = calculate_entropy(img1_gray)
    entropy2 = calculate_entropy(img2_gray)
    print(f"画像1エントロピー: {entropy1:.3f}")
    print(f"画像2エントロピー: {entropy2:.3f}")
    print(f"差: {abs(entropy1 - entropy2):.3f}")

    results['entropy'] = {
        'img1': round(entropy1, 3),
        'img2': round(entropy2, 3)
    }

    # 7. ノイズレベル
    print("\n【7. ノイズレベル分析】")
    print_usage_status("ノイズ推定開始（GPU使用）" if GPU_AVAILABLE else "ノイズ推定開始（CPU使用）")
    noise1 = estimate_noise_gpu(img1_gray)
    noise2 = estimate_noise_gpu(img2_gray)
    print(f"画像1ノイズレベル: {noise1:.2f}")
    print(f"画像2ノイズレベル: {noise2:.2f}")
    print(f"差: {abs(noise1 - noise2):.2f} ({((noise2/noise1 - 1) * 100 if noise1 != 0 else 0):+.1f}%)")

    results['noise'] = {
        'img1': round(noise1, 2),
        'img2': round(noise2, 2)
    }

    # 8. アーティファクト検出
    print("\n【8. アーティファクト検出】")
    block_noise1, ringing1 = detect_artifacts(img1_gray)
    block_noise2, ringing2 = detect_artifacts(img2_gray)

    print(f"画像1ブロックノイズ: {block_noise1:.2f}")
    print(f"画像2ブロックノイズ: {block_noise2:.2f}")
    print(f"画像1リンギング: {ringing1:.2f}")
    print(f"画像2リンギング: {ringing2:.2f}")

    results['artifacts'] = {
        'img1_block_noise': round(block_noise1, 2),
        'img2_block_noise': round(block_noise2, 2),
        'img1_ringing': round(ringing1, 2),
        'img2_ringing': round(ringing2, 2)
    }

    # 9. エッジ保持率
    print("\n【9. エッジ保持率】")
    print_usage_status("エッジ検出開始（GPU使用）" if GPU_AVAILABLE else "エッジ検出開始（CPU使用）")
    edge_density1 = detect_edges_gpu(img1_gray)
    edge_density2 = detect_edges_gpu(img2_gray)

    print(f"画像1エッジ密度: {edge_density1:.2f}%")
    print(f"画像2エッジ密度: {edge_density2:.2f}%")
    print(f"差: {abs(edge_density1 - edge_density2):.2f}% ({((edge_density2/edge_density1 - 1) * 100 if edge_density1 != 0 else 0):+.1f}%)")

    results['edges'] = {
        'img1_density': round(edge_density1, 2),
        'img2_density': round(edge_density2, 2),
        'difference_pct': round((edge_density2/edge_density1 - 1) * 100 if edge_density1 != 0 else 0, 1)
    }

    # 10. 色分布分析
    print("\n【10. 色分布分析（RGB/HSV/LAB）】")
    color_stats1 = analyze_color_distribution(img1_rgb)
    color_stats2 = analyze_color_distribution(img2_rgb)

    # RGB
    print("RGB色空間:")
    for channel in ['Red', 'Green', 'Blue']:
        print(f"  {channel}チャンネル:")
        print(f"    元画像: 平均={color_stats1[channel]['mean']:.1f}, 標準偏差={color_stats1[channel]['std']:.1f}")
        print(f"    AI処理結果: 平均={color_stats2[channel]['mean']:.1f}, 標準偏差={color_stats2[channel]['std']:.1f}")

    # HSV
    print(f"\nHSV色空間 - 彩度:")
    print(f"  元画像: 平均={color_stats1['Saturation']['mean']:.1f}")
    print(f"  AI処理結果: 平均={color_stats2['Saturation']['mean']:.1f}")

    # LAB（知覚的色差）
    print(f"\nLAB色空間（知覚的色分析）:")
    print(f"  明度(L):")
    print(f"    元画像: {color_stats1['LAB']['L_mean']:.1f} ± {color_stats1['LAB']['L_std']:.1f}")
    print(f"    AI処理結果: {color_stats2['LAB']['L_mean']:.1f} ± {color_stats2['LAB']['L_std']:.1f}")
    print(f"  a(赤-緑):")
    print(f"    元画像: {color_stats1['LAB']['a_mean']:.1f} ± {color_stats1['LAB']['a_std']:.1f}")
    print(f"    AI処理結果: {color_stats2['LAB']['a_mean']:.1f} ± {color_stats2['LAB']['a_std']:.1f}")
    print(f"  b(黄-青):")
    print(f"    元画像: {color_stats1['LAB']['b_mean']:.1f} ± {color_stats1['LAB']['b_std']:.1f}")
    print(f"    AI処理結果: {color_stats2['LAB']['b_mean']:.1f} ± {color_stats2['LAB']['b_std']:.1f}")

    # Delta E (CIE2000) - 知覚的色差
    print_usage_status("色差計算開始（GPU使用）" if GPU_AVAILABLE else "色差計算開始（CPU使用）")

    if img_original_rgb is not None:
        # 元画像がある場合：それぞれ元画像との色差を計算
        delta_e_img1_vs_orig = calculate_color_difference_gpu(img1_rgb, img_original_rgb)
        delta_e_img2_vs_orig = calculate_color_difference_gpu(img2_rgb, img_original_rgb)
        print(f"\n  画像1 vs 元画像 ΔE: {delta_e_img1_vs_orig:.2f}")
        print(f"  画像2 vs 元画像 ΔE: {delta_e_img2_vs_orig:.2f}")
        if delta_e_img1_vs_orig < delta_e_img2_vs_orig:
            print(f"  → 元画像の方が元画像の色に近い (差: {delta_e_img2_vs_orig - delta_e_img1_vs_orig:.2f})")
        else:
            print(f"  → AI処理結果の方が元画像の色に近い (差: {delta_e_img1_vs_orig - delta_e_img2_vs_orig:.2f})")
        print(f"    (ΔE < 1: 人間の目では区別不可, ΔE < 5: 許容範囲, ΔE > 10: 明確な違い)")
        delta_e_result = {
            'img1_vs_original': round(delta_e_img1_vs_orig, 2),
            'img2_vs_original': round(delta_e_img2_vs_orig, 2)
        }
    else:
        # 元画像がない場合：元画像 vs AI処理結果
        delta_e_val = calculate_color_difference_gpu(img1_rgb, img2_rgb)
        print(f"\n  ΔE (色差): {delta_e_val:.2f}")
        print(f"    (ΔE < 1: 人間の目では区別不可, ΔE < 5: 許容範囲, ΔE > 10: 明確な違い)")
        delta_e_result = round(delta_e_val, 2)

    results['color_distribution'] = {
        'img1': color_stats1,
        'img2': color_stats2,
        'delta_e': delta_e_result
    }

    # 11. 周波数領域分析
    print("\n【11. 周波数領域分析（FFT）】")
    freq_analysis1 = analyze_frequency_domain(img1_gray)
    freq_analysis2 = analyze_frequency_domain(img2_gray)

    print(f"画像1低周波成分比率: {freq_analysis1['low_freq_ratio']:.3f}")
    print(f"画像2低周波成分比率: {freq_analysis2['low_freq_ratio']:.3f}")
    print(f"画像1高周波成分比率: {freq_analysis1['high_freq_ratio']:.3f}")
    print(f"画像2高周波成分比率: {freq_analysis2['high_freq_ratio']:.3f}")

    results['frequency_analysis'] = {
        'img1': freq_analysis1,
        'img2': freq_analysis2
    }

    # 12. テクスチャ分析
    print("\n【12. テクスチャ分析】")
    texture1 = analyze_texture(img1_gray)
    texture2 = analyze_texture(img2_gray)

    print(f"画像1テクスチャ複雑度: {texture1['texture_complexity']:.2f}")
    print(f"画像2テクスチャ複雑度: {texture2['texture_complexity']:.2f}")

    results['texture'] = {
        'img1': texture1,
        'img2': texture2
    }

    # 13. 局所的品質分析
    print("\n【13. 局所的品質分析（パッチベースSSIM）】")
    local_ssim = analyze_local_quality(img1_rgb, img2_rgb)

    print(f"局所SSIM 平均: {np.mean(local_ssim):.4f}")
    print(f"局所SSIM 最小: {np.min(local_ssim):.4f}")
    print(f"局所SSIM 最大: {np.max(local_ssim):.4f}")
    print(f"局所SSIM 標準偏差: {np.std(local_ssim):.4f}")

    results['local_quality'] = {
        'mean_ssim': round(np.mean(local_ssim), 4),
        'min_ssim': round(np.min(local_ssim), 4),
        'max_ssim': round(np.max(local_ssim), 4),
        'std_ssim': round(np.std(local_ssim), 4)
    }

    # 14. ヒストグラム類似度
    print("\n【14. ヒストグラム類似度】")
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f"ヒストグラム相関: {hist_corr:.4f} (1.0 = 完全一致)")

    results['histogram_correlation'] = round(hist_corr, 4)

    # 15. 総合スコア計算（絶対評価）
    print("\n【15. 総合評価スコア】")
    print("=" * 80)

    # 各指標を絶対値で評価（両画像を独立して採点）

    # SSIM/PSNRの値を取得（元画像の有無で形式が異なる）
    if img_original_rgb is not None:
        # 元画像がある場合：dictから取得
        ssim_data = results.get('ssim', {})
        if isinstance(ssim_data, dict):
            ssim_score_val = (ssim_data.get('img1_vs_original', 0) + ssim_data.get('img2_vs_original', 0)) / 2 * 100
        else:
            ssim_score_val = 0

        psnr_data = results.get('psnr', {})
        if isinstance(psnr_data, dict):
            psnr_score_val = min((psnr_data.get('img1_vs_original', 0) + psnr_data.get('img2_vs_original', 0)) / 2 * 2, 100)
        else:
            psnr_score_val = 0
    else:
        # 元画像がない場合：floatから取得
        ssim_score_val = results.get('ssim', 0) * 100
        psnr_score_val = min(results.get('psnr', 0) * 2, 100)

    # 画像1のスコア（17項目）
    # 2. MS-SSIM
    ms_ssim_score_val = (results.get('ms_ssim', 0) or 0) * 100

    # 4. LPIPS（低いほど良い、反転）
    lpips_score_val = max(0, 100 - (results.get('lpips', 0) or 0) * 1000) if results.get('lpips') else 50

    # 5. シャープネス
    sharp1_score = min(sharpness1 / 5, 100)
    sharp2_score = min(sharpness2 / 5, 100)

    # 6. コントラスト
    contrast1_score = min(contrast1, 100)
    contrast2_score = min(contrast2, 100)

    # 7. エントロピー
    entropy1_score = min(entropy1 / 8 * 100, 100)
    entropy2_score = min(entropy2 / 8 * 100, 100)

    # 8. ノイズ（低いほど良い）
    noise1_score = max(0, 100 - noise1 / 2)
    noise2_score = max(0, 100 - noise2 / 2)

    # 9. アーティファクト（低いほど良い）
    artifact1_total = block_noise1 + ringing1
    artifact2_total = block_noise2 + ringing2
    artifact1_score = max(0, 100 - artifact1_total / 50)
    artifact2_score = max(0, 100 - artifact2_total / 50)

    # 10. エッジ保持
    edge1_score = min(edge_density1 * 2, 100)
    edge2_score = min(edge_density2 * 2, 100)

    # 11. 色差（低いほど良い）
    delta_e_data = results['color_distribution'].get('delta_e', 0)
    if isinstance(delta_e_data, dict):
        # 元画像がある場合：平均値を使用
        avg_delta_e = (delta_e_data.get('img1_vs_original', 0) + delta_e_data.get('img2_vs_original', 0)) / 2
        color_diff_score = max(0, 100 - avg_delta_e * 2)
    else:
        # 元画像がない場合：単一値を使用
        color_diff_score = max(0, 100 - delta_e_data * 2)

    # 12. テクスチャ
    texture1_score = min(texture1['texture_complexity'] * 10, 100)
    texture2_score = min(texture2['texture_complexity'] * 10, 100)

    # 13. 局所品質
    local_quality_score = results['local_quality']['mean_ssim'] * 100

    # 14. ヒストグラム
    histogram_score = hist_corr * 100

    # 画像1の総合スコア（画像品質項目のみ）
    total1 = (sharp1_score + contrast1_score + entropy1_score + noise1_score +
              artifact1_score + edge1_score + texture1_score) / 7

    # 画像2の総合スコア（画像品質項目のみ）
    total2 = (sharp2_score + contrast2_score + entropy2_score + noise2_score +
              artifact2_score + edge2_score + texture2_score) / 7

    print(f"画像1総合スコア: {total1:.1f} / 100")
    print(f"画像2総合スコア: {total2:.1f} / 100")

    if total2 > total1:
        print(f"→ 画像2が {total2 - total1:.1f}点 優位")
    elif total1 > total2:
        print(f"→ 画像1が {total1 - total2:.1f}点 優位")
    else:
        print(f"→ 同等の品質")

    print("\n【スコア内訳（7項目で評価）】")
    print(f"             画像1   画像2")
    print(f"シャープネス:   {sharp1_score:5.1f}   {sharp2_score:5.1f}")
    print(f"コントラスト:   {contrast1_score:5.1f}   {contrast2_score:5.1f}")
    print(f"エントロピー:   {entropy1_score:5.1f}   {entropy2_score:5.1f}")
    print(f"ノイズ対策:     {noise1_score:5.1f}   {noise2_score:5.1f}")
    print(f"エッジ保持:     {edge1_score:5.1f}   {edge2_score:5.1f}")
    print(f"歪み抑制:       {artifact1_score:5.1f}   {artifact2_score:5.1f}")
    print(f"テクスチャ:     {texture1_score:5.1f}   {texture2_score:5.1f}")

    print(f"\n【類似度指標（参考値）】")
    print(f"  SSIM:        {ssim_score_val:.1f}/100")
    print(f"  MS-SSIM:     {ms_ssim_score_val:.1f}/100")
    print(f"  PSNR:        {psnr_score_val:.1f}/100")
    print(f"  LPIPS:       {lpips_score_val:.1f}/100")
    print(f"  色差:        {color_diff_score:.1f}/100")
    print(f"  局所品質:    {local_quality_score:.1f}/100")
    print(f"  ヒストグラム: {histogram_score:.1f}/100")

    results['total_score'] = {
        'img1': round(total1, 1),
        'img2': round(total2, 1),
        'breakdown': {
            'img1': {
                'sharpness': round(sharp1_score, 1),
                'contrast': round(contrast1_score, 1),
                'entropy': round(entropy1_score, 1),
                'noise': round(noise1_score, 1),
                'edge': round(edge1_score, 1),
                'artifact': round(artifact1_score, 1),
                'texture': round(texture1_score, 1)
            },
            'img2': {
                'sharpness': round(sharp2_score, 1),
                'contrast': round(contrast2_score, 1),
                'entropy': round(entropy2_score, 1),
                'noise': round(noise2_score, 1),
                'edge': round(edge2_score, 1),
                'artifact': round(artifact2_score, 1),
                'texture': round(texture2_score, 1)
            },
            'ssim': round(ssim_score_val, 1),
            'psnr': round(psnr_score_val, 1)
        }
    }

    # 16. 結果可視化
    print("\n【16. 結果可視化を生成中...】")
    print_usage_status("画像生成開始")

    # 詳細可視化
    create_detailed_visualizations(img1_rgb, img2_rgb, img1_gray, img2_gray, output_dir)

    # 比較レポート生成
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)
    report_path = create_comparison_report(results, img1_name, img2_name, output_dir)
    print(f"比較レポートを生成: {report_path}")

    # 差分画像
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(output_dir, 'difference.png'), diff)
    cv2.imwrite(os.path.join(output_dir, 'heatmap.png'), heatmap)

    # エッジ画像を生成して保存
    edges1_save = cv2.Canny(img1_gray, 100, 200)
    edges2_save = cv2.Canny(img2_gray, 100, 200)
    cv2.imwrite(os.path.join(output_dir, 'edges_img1.png'), edges1_save)
    cv2.imwrite(os.path.join(output_dir, 'edges_img2.png'), edges2_save)

    # 比較画像
    comparison = np.hstack([img1, img2, diff])
    cv2.imwrite(os.path.join(output_dir, 'comparison.png'), comparison)

    # JSON形式で結果を保存
    # numpy型をPython標準型に変換するためのカスタムエンコーダ
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

    json_path = os.path.join(output_dir, 'analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"結果を '{output_dir}/' に保存しました")
    print("  - comparison_report.png: ★比較レポート（グラフとスコア表示）★")
    print("  - detailed_analysis.png: 詳細分析可視化（12枚の分析画像）")
    print("  - difference.png: 差分画像")
    print("  - heatmap.png: 差分ヒートマップ")
    print("  - comparison.png: 3枚並べて比較")
    print("  - edges_*.png: エッジ検出結果")
    print("  - analysis_results.json: 分析結果データ（JSON形式）")

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)

    # 結果の解釈を追加
    # 評価モードを結果に保存
    results['evaluation_mode'] = evaluation_mode

    try:
        from result_interpreter import interpret_results, format_interpretation_text
        interpretation = interpret_results(results)
        interpretation_text = format_interpretation_text(interpretation)
        print("\n" + interpretation_text)

        # 解釈結果も保存
        results['interpretation'] = interpretation
    except Exception as e:
        print(f"解釈生成エラー: {e}")

    return results

# 使用例
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        # コマンドライン引数から画像パスを取得
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'analysis_results'
    else:
        # デフォルト値
        img1_path = 'chainner_oiran.png'
        img2_path = 'upscayl_oiran.png'
        output_dir = 'analysis_results'

    print(f"元画像: {img1_path}")
    print(f"AI処理結果: {img2_path}")
    print(f"出力先: {output_dir}")
    print()

    analyze_images(img1_path, img2_path, output_dir)
