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
    LPIPS_AVAILABLE = True

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
    GPU_AVAILABLE = False
    DEVICE = None
    GPU_NAME = None
    torch = None

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
    GPU対応シャープネス計算（ラプラシアン分散）

    Returns:
        float: シャープネス値（高いほど鮮明）
    """
    if not LPIPS_AVAILABLE or torch is None:
        # PyTorchがない場合はCPU版にフォールバック
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        return laplacian.var()

    try:
        # 画像をPyTorchテンソルに変換
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        # ラプラシアンカーネル
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # 畳み込み
        with torch.no_grad():
            laplacian = F.conv2d(img_tensor, laplacian_kernel, padding=1)
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
    GPU対応エッジ検出（Sobelフィルタ）

    Returns:
        float: エッジ密度
    """
    if not LPIPS_AVAILABLE or torch is None:
        # CPU版にフォールバック
        edges = cv2.Canny(img_gray, 100, 200)
        return np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255) * 100

    try:
        # 画像をPyTorchテンソルに変換
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        # Sobelフィルタ
        sobel_x = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        sobel_y = torch.tensor([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            edge_x = F.conv2d(img_tensor, sobel_x, padding=1)
            edge_y = F.conv2d(img_tensor, sobel_y, padding=1)
            edges = torch.sqrt(edge_x**2 + edge_y**2)
            edge_density = torch.sum(edges > 0.1) / edges.numel() * 100

        return float(edge_density.item())

    except Exception as e:
        print(f"GPU エッジ検出エラー（CPU版にフォールバック）: {e}")
        edges = cv2.Canny(img_gray, 100, 200)
        return np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255) * 100

def calculate_color_difference_gpu(img1_rgb, img2_rgb):
    """
    GPU対応LAB色空間での色差計算

    Returns:
        float: 平均色差（ΔE）
    """
    if not LPIPS_AVAILABLE or torch is None:
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

        # 簡易的な色差計算（RGB空間）
        with torch.no_grad():
            color_diff = torch.sqrt(torch.sum((img1_tensor - img2_tensor) ** 2, dim=1))
            mean_diff = torch.mean(color_diff)

        return float(mean_diff.item()) * 255  # スケール調整

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
    """テクスチャ分析 - GLCM（Gray Level Co-occurrence Matrix）"""
    # LBP (Local Binary Patterns) による簡易テクスチャ分析
    # 画像サイズを縮小して計算を高速化
    small = cv2.resize(img_gray, (img_gray.shape[1]//4, img_gray.shape[0]//4))

    # 簡易的なテクスチャ複雑度
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
    plt.title('画像1 (元画像)', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(img2_rgb)
    plt.title('画像2 (比較画像)', fontsize=12, fontweight='bold')
    plt.axis('off')

    # 2. ヒストグラム
    plt.subplot(3, 4, 3)
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img1_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, linewidth=1.5)
    plt.title('ヒストグラム - 画像1', fontsize=11)
    plt.xlim([0, 256])
    plt.xlabel('輝度値', fontsize=9)
    plt.ylabel('ピクセル数', fontsize=9)

    plt.subplot(3, 4, 4)
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img2_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, linewidth=1.5)
    plt.title('ヒストグラム - 画像2', fontsize=11)
    plt.xlim([0, 256])
    plt.xlabel('輝度値', fontsize=9)
    plt.ylabel('ピクセル数', fontsize=9)

    # 3. エッジ検出
    edges1 = cv2.Canny(img1_gray, 100, 200)
    edges2 = cv2.Canny(img2_gray, 100, 200)

    plt.subplot(3, 4, 5)
    plt.imshow(edges1, cmap='gray')
    plt.title('エッジ検出 - 画像1', fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(edges2, cmap='gray')
    plt.title('エッジ検出 - 画像2', fontsize=11)
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
    plt.title('周波数スペクトル - 画像1', fontsize=11)
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(magnitude2, cmap='gray')
    plt.title('周波数スペクトル - 画像2', fontsize=11)
    plt.axis('off')

    # 6. シャープネス可視化（ラプラシアン）
    lap1 = cv2.Laplacian(img1_gray, cv2.CV_64F)
    lap2 = cv2.Laplacian(img2_gray, cv2.CV_64F)

    plt.subplot(3, 4, 11)
    im1 = plt.imshow(np.abs(lap1), cmap='viridis')
    plt.title('シャープネスマップ - 画像1', fontsize=11)
    plt.axis('off')
    cb1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=8)

    plt.subplot(3, 4, 12)
    im2 = plt.imshow(np.abs(lap2), cmap='viridis')
    plt.title('シャープネスマップ - 画像2', fontsize=11)
    plt.axis('off')
    cb2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_analysis.png', dpi=150, bbox_inches='tight')
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

    bars1 = ax1.barh(x - width/2, img1_values, width, label='画像1', color='#3498db')
    bars2 = ax1.barh(x + width/2, img2_values, width, label='画像2', color='#e74c3c')

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

    ax2.barh(['画像1 (基準)', '画像2'], [img1_score, total_score],
             color=['#3498db', '#e74c3c' if total_score < 70 else '#f39c12' if total_score < 90 else '#2ecc71'])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('総合スコア', fontsize=11, fontweight='bold')
    ax2.set_title('総合評価', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3)

    for i, (score, name) in enumerate(zip([img1_score, total_score], ['画像1', '画像2'])):
        ax2.text(score + 2, i, f'{score:.1f}', va='center', fontsize=12, fontweight='bold')

    # 主要指標
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')

    delta_e_value = results['color_distribution'].get('delta_e', 0)

    info_text = f"""
【主要指標】

SSIM: {results['ssim']:.4f}
  (1.0 = 完全一致)

PSNR: {results['psnr']:.2f} dB
  (30dB以上で視覚的に同等)

シャープネス:
  画像1: {results['sharpness']['img1']:.2f}
  画像2: {results['sharpness']['img2']:.2f}
  差: {results['sharpness']['difference_pct']:+.1f}%

色差 (ΔE): {delta_e_value:.2f}
  (< 5: 許容, > 10: 明確な違い)
    """

    ax3.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax3.set_title('詳細データ', fontsize=13, fontweight='bold', pad=10)

    # エッジ比較
    ax4 = plt.subplot(2, 3, 4)
    edge_data = [results['edges']['img1_density'], results['edges']['img2_density']]
    ax4.bar(['画像1', '画像2'], edge_data, color=['#3498db', '#9b59b6'])
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
    ax5.set_xticklabels(['画像1', '画像2'])
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

    ax6.bar(x - width/2, freq1, width, label='画像1', color='#3498db')
    ax6.bar(x + width/2, freq2, width, label='画像2', color='#9b59b6')

    ax6.set_ylabel('比率 (%)', fontsize=11, fontweight='bold')
    ax6.set_title('周波数成分分布', fontsize=13, fontweight='bold', pad=10)
    ax6.set_xticks(x)
    ax6.set_xticklabels(['低周波', '高周波'])
    ax6.legend(fontsize=10)
    ax6.set_ylim(0, 100)
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_report.png', dpi=150, bbox_inches='tight')
    plt.close()

    return f'{output_dir}/comparison_report.png'

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

def analyze_images(img1_path, img2_path, output_dir='analysis_results'):
    """
    2つの画像を詳細に比較分析する（拡張版）

    Parameters:
    img1_path: 画像1のパス（例: chaiNNer）
    img2_path: 画像2のパス（例: Upscayl）
    output_dir: 結果保存ディレクトリ
    """

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 画像読み込み（日本語パス対応）
    img1 = imread_unicode(img1_path)
    img2 = imread_unicode(img2_path)

    if img1 is None or img2 is None:
        print("エラー: 画像ファイルが読み込めません")
        print(f"画像1パス: {img1_path}")
        print(f"画像2パス: {img2_path}")
        return

    # 画像サイズチェックと調整
    if img1.shape != img2.shape:
        print(f"\n画像サイズが異なります:")
        print(f"  画像1: {img1.shape[1]} x {img1.shape[0]} px")
        print(f"  画像2: {img2.shape[1]} x {img2.shape[0]} px")
        print(f"画像2を画像1のサイズにリサイズします...\n")

        # 画像2を画像1のサイズに合わせる
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

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
        'image2_path': img2_path
    }

    print("=" * 80)
    print("詳細画像比較分析レポート")
    print("=" * 80)

    # 1. 基本情報
    print("\n【1. 基本情報】")
    print(f"画像1サイズ: {img1.shape[1]} x {img1.shape[0]} px")
    print(f"画像2サイズ: {img2.shape[1]} x {img2.shape[0]} px")

    size1 = os.path.getsize(img1_path) / (1024 * 1024)
    size2 = os.path.getsize(img2_path) / (1024 * 1024)
    print(f"画像1ファイルサイズ: {size1:.2f} MB")
    print(f"画像2ファイルサイズ: {size2:.2f} MB")
    print(f"サイズ差: {abs(size1 - size2):.2f} MB ({((size2/size1 - 1) * 100):+.1f}%)")

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
    ssim_score = calculate_ssim_gpu(img1_rgb, img2_rgb)
    print(f"SSIM: {ssim_score:.4f}")

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
    psnr_score = calculate_psnr_gpu(img1_rgb, img2_rgb)
    print(f"PSNR: {psnr_score:.2f} dB")

    results['psnr'] = round(psnr_score, 2)

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
        print(f"    画像1: 平均={color_stats1[channel]['mean']:.1f}, 標準偏差={color_stats1[channel]['std']:.1f}")
        print(f"    画像2: 平均={color_stats2[channel]['mean']:.1f}, 標準偏差={color_stats2[channel]['std']:.1f}")

    # HSV
    print(f"\nHSV色空間 - 彩度:")
    print(f"  画像1: 平均={color_stats1['Saturation']['mean']:.1f}")
    print(f"  画像2: 平均={color_stats2['Saturation']['mean']:.1f}")

    # LAB（知覚的色差）
    print(f"\nLAB色空間（知覚的色分析）:")
    print(f"  明度(L):")
    print(f"    画像1: {color_stats1['LAB']['L_mean']:.1f} ± {color_stats1['LAB']['L_std']:.1f}")
    print(f"    画像2: {color_stats2['LAB']['L_mean']:.1f} ± {color_stats2['LAB']['L_std']:.1f}")
    print(f"  a(赤-緑):")
    print(f"    画像1: {color_stats1['LAB']['a_mean']:.1f} ± {color_stats1['LAB']['a_std']:.1f}")
    print(f"    画像2: {color_stats2['LAB']['a_mean']:.1f} ± {color_stats2['LAB']['a_std']:.1f}")
    print(f"  b(黄-青):")
    print(f"    画像1: {color_stats1['LAB']['b_mean']:.1f} ± {color_stats1['LAB']['b_std']:.1f}")
    print(f"    画像2: {color_stats2['LAB']['b_mean']:.1f} ± {color_stats2['LAB']['b_std']:.1f}")

    # Delta E (CIE2000) - 知覚的色差
    print_usage_status("色差計算開始（GPU使用）" if GPU_AVAILABLE else "色差計算開始（CPU使用）")
    delta_e = calculate_color_difference_gpu(img1_rgb, img2_rgb)

    print(f"\n  ΔE (色差): {delta_e:.2f}")
    print(f"    (ΔE < 1: 人間の目では区別不可, ΔE < 5: 許容範囲, ΔE > 10: 明確な違い)")

    results['color_distribution'] = {
        'img1': color_stats1,
        'img2': color_stats2,
        'delta_e': round(delta_e, 2)
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

    # 画像1のスコア（17項目）
    # 1. SSIM（類似度として参考値）
    ssim_score_val = ssim_score * 100

    # 2. MS-SSIM
    ms_ssim_score_val = (results.get('ms_ssim', 0) or 0) * 100

    # 3. PSNR
    psnr_score_val = min(psnr_score * 2, 100)

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
    color_diff_score = max(0, 100 - delta_e * 2)

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

    cv2.imwrite(f'{output_dir}/difference.png', diff)
    cv2.imwrite(f'{output_dir}/heatmap.png', heatmap)

    # エッジ画像を生成して保存
    edges1_save = cv2.Canny(img1_gray, 100, 200)
    edges2_save = cv2.Canny(img2_gray, 100, 200)
    cv2.imwrite(f'{output_dir}/edges_img1.png', edges1_save)
    cv2.imwrite(f'{output_dir}/edges_img2.png', edges2_save)

    # 比較画像
    comparison = np.hstack([img1, img2, diff])
    cv2.imwrite(f'{output_dir}/comparison.png', comparison)

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
            return super(NumpyEncoder, self).default(obj)

    with open(f'{output_dir}/analysis_results.json', 'w', encoding='utf-8') as f:
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

    print(f"画像1: {img1_path}")
    print(f"画像2: {img2_path}")
    print(f"出力先: {output_dir}")
    print()

    analyze_images(img1_path, img2_path, output_dir)
