#!/usr/bin/env python3
"""
Translation dictionaries for analyze_results.py
Supports Japanese (ja) and English (en)
"""

import json
import os

class I18n:
    """Internationalization class for loading translations from JSON files"""

    def __init__(self, lang='ja'):
        """
        Initialize I18n with specified language

        Args:
            lang: Language code ('ja' or 'en')
        """
        self.lang = lang
        self.translations = {}
        self._load_translations()

    def _load_translations(self):
        """Load translations from JSON files"""
        locales_dir = os.path.join(os.path.dirname(__file__), 'locales')
        locale_file = os.path.join(locales_dir, f'{self.lang}.json')

        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Flatten nested structure if analyzer key exists
                if 'analyzer' in data:
                    self.translations = data['analyzer']
                else:
                    self.translations = data
        except FileNotFoundError:
            print(f"Warning: Translation file not found: {locale_file}")
            self.translations = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing translation file: {e}")
            self.translations = {}

    def t(self, key):
        """
        Get translated string

        Args:
            key: Translation key (e.g., 'analyzer.section_1')

        Returns:
            Translated string or key if not found
        """
        # Remove 'analyzer.' prefix if present since we already loaded analyzer namespace
        if key.startswith('analyzer.'):
            key = key[9:]  # Remove 'analyzer.' prefix

        return self.translations.get(key, key)

    def set_language(self, lang):
        """
        Change language

        Args:
            lang: Language code ('ja' or 'en')
        """
        self.lang = lang
        self._load_translations()

TRANSLATIONS = {
    'ja': {
        # Font settings
        'font_family': ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif'],

        # Plot titles
        'model_comparison': 'モデル別総合スコア比較',
        'correlation_matrix': '17項目の相関マトリックス',
        'strategy_map': 'AIモデルの戦略マップ：忠実度 vs 鮮明度',
        'lpips_distribution': 'モデル別 LPIPS 分布（安定性評価）\n箱が小さい = 安定',
        'ssim_psnr_correlation': 'SSIM vs PSNR 相関分析\n相関から外れる点 = ハルシネーション候補',
        'noise_artifact': 'ノイズ vs アーティファクト\n左下が理想（両方少ない）',
        'radar_chart': 'モデル別性能プロファイル（レーダーチャート）\n外側ほど高性能',
        'violin_plot': '17項目の分布（バイオリンプロット）',
        'hallucination_ssim_psnr': 'ハルシネーション検出：SSIM高 & PSNR低\n右下領域 = 構造を模倣したが忠実性が低い',
        'hallucination_sharpness_noise': '過剰処理検出：高シャープネス & 高ノイズ\n右上領域 = シャープ化でノイズ増幅',
        'hallucination_edge_quality': '不自然なエッジ検出：エッジ増加 & 局所品質ばらつき大\n右上領域 = エッジ追加が不均一',
        'hallucination_freq_entropy': '人工パターン検出：高周波 vs エントロピー\n相関から外れる = 反復パターン疑い',
        'tradeoff_ssim_noise': '品質トレードオフ：構造類似性 vs ノイズ',
        'tradeoff_psnr_contrast': '品質トレードオフ：忠実度 vs コントラスト強調',
        'tradeoff_sharpness_artifact': '品質トレードオフ：鮮明化 vs 歪み',
        'tradeoff_lpips_msssim': '知覚 vs 構造：LPIPS vs MS-SSIM\n負の相関が期待される',
        'tradeoff_texture_freq': 'テクスチャ vs 周波数成分の一貫性',
        'medical_contrast_histogram': '医療画像品質：コントラスト強調が濃度分布を崩していないか',
        'medical_edge_quality': '医療画像品質：エッジ保持と局所品質の関係',
        'medical_noise_std': '医療画像品質：ノイズの局所的偏在性',
        'medical_deltae_lab': '医療画像品質：色変化と明度の関係（病理画像で重要）',
        'total_score_histogram': '総合スコア分布（モデル別）',
        'pca_2d': '主成分分析（PCA）：17項目を2次元に圧縮',
        'pca_variance': 'PCA累積寄与率：何次元で95%説明できるか',
        'percentile_bands': 'パーセンタイルバンド（25%-75%）プロット',

        # Axis labels
        'total_score': '総合スコア',
        'total_score_avg': '総合スコア（平均）',
        'psnr': 'PSNR（信号対雑音比）[dB]',
        'psnr_fidelity': 'PSNR（忠実度）[dB]',
        'psnr_db': 'PSNR [dB]',
        'ssim': 'SSIM（構造類似性）',
        'sharpness': 'シャープネス（鮮明度）',
        'sharpness_clarity': 'シャープネス',
        'lpips': 'LPIPS（知覚的類似度）',
        'noise': 'ノイズレベル',
        'noise_level': 'ノイズ',
        'artifacts': 'アーティファクト（ブロック+リンギング）',
        'artifact_total': 'アーティファクト',
        'edge_density': 'エッジ密度',
        'local_quality_std': '局所品質標準偏差',
        'local_quality_mean': '局所品質平均',
        'high_freq_ratio': '高周波成分比率',
        'entropy': 'エントロピー（情報量）',
        'contrast': 'コントラスト',
        'texture_complexity': 'テクスチャ複雑度',
        'histogram_corr': 'ヒストグラム相関',
        'delta_e': '色差 ΔE',
        'lab_lightness': 'LAB明度',
        'frequency': '頻度',
        'pc1': '第1主成分',
        'pc2': '第2主成分',
        'num_components': '主成分数',
        'cumulative_variance': '累積寄与率',
        'ai_model': 'AIモデル',

        # Annotations
        'median': '中央値',
        'regression_line': '回帰直線',
        'ideal_region': '理想領域',
        'ideal_region_detail': '理想領域\n（高忠実・高鮮明）',
        'over_processing': '過剰処理領域\n（低忠実・高鮮明）\nハルシネーション疑い',
        'hallucination_suspected': 'ハルシネーション疑い',
        'over_processing_suspected': '過剰処理疑い',
        'unnatural_edges_suspected': '不自然なエッジ疑い',
        'psnr_threshold': 'PSNR閾値',
        'ssim_threshold': 'SSIM閾値',
        'noise_inverted': 'ノイズ\n（反転）',
        'artifact_inverted': 'アーティファクト\n（反転）',
        'line_95': '95%ライン',
        'distribution_of': 'の分布',
        'cumulative_variance_prefix': '累積寄与率:',
        'cases': '件',
    },
    'en': {
        # Font settings
        'font_family': ['DejaVu Sans', 'Arial', 'sans-serif'],

        # Plot titles
        'model_comparison': 'Model Comparison by Total Score',
        'correlation_matrix': 'Correlation Matrix of 17 Metrics',
        'strategy_map': 'AI Model Strategy Map: Fidelity vs Clarity',
        'lpips_distribution': 'LPIPS Distribution by Model (Stability)\nSmaller box = More stable',
        'ssim_psnr_correlation': 'SSIM vs PSNR Correlation\nOutliers = Hallucination Candidates',
        'noise_artifact': 'Noise vs Artifacts\nLower-left is ideal (both low)',
        'radar_chart': 'Model Performance Profile (Radar Chart)\nOuter = Better',
        'violin_plot': 'Distribution of 17 Metrics (Violin Plot)',
        'hallucination_ssim_psnr': 'Hallucination Detection: High SSIM & Low PSNR\nLower-right = Mimicked structure, low fidelity',
        'hallucination_sharpness_noise': 'Over-processing Detection: High Sharpness & Noise\nUpper-right = Noise amplified by sharpening',
        'hallucination_edge_quality': 'Unnatural Edge Detection: High Edge & Quality Variance\nUpper-right = Uneven edge addition',
        'hallucination_freq_entropy': 'Artificial Pattern Detection: High Freq vs Entropy\nOutliers = Repetitive pattern suspected',
        'tradeoff_ssim_noise': 'Quality Tradeoff: Structural Similarity vs Noise',
        'tradeoff_psnr_contrast': 'Quality Tradeoff: Fidelity vs Contrast Enhancement',
        'tradeoff_sharpness_artifact': 'Quality Tradeoff: Sharpening vs Distortion',
        'tradeoff_lpips_msssim': 'Perception vs Structure: LPIPS vs MS-SSIM\nNegative correlation expected',
        'tradeoff_texture_freq': 'Texture vs Frequency Component Consistency',
        'medical_contrast_histogram': 'Medical Image Quality: Contrast Enhancement vs Intensity Distribution',
        'medical_edge_quality': 'Medical Image Quality: Edge Preservation vs Local Quality',
        'medical_noise_std': 'Medical Image Quality: Local Noise Distribution',
        'medical_deltae_lab': 'Medical Image Quality: Color vs Lightness (Important for pathology)',
        'total_score_histogram': 'Total Score Distribution by Model',
        'pca_2d': 'Principal Component Analysis (PCA): 17 Metrics to 2D',
        'pca_variance': 'PCA Cumulative Variance: Dimensions to Explain 95%',
        'percentile_bands': 'Percentile Band (25%-75%) Plot',

        # Axis labels
        'total_score': 'Total Score',
        'total_score_avg': 'Total Score (Average)',
        'psnr': 'PSNR (Signal-to-Noise Ratio) [dB]',
        'psnr_fidelity': 'PSNR (Fidelity) [dB]',
        'psnr_db': 'PSNR [dB]',
        'ssim': 'SSIM (Structural Similarity)',
        'sharpness': 'Sharpness (Clarity)',
        'sharpness_clarity': 'Sharpness',
        'lpips': 'LPIPS (Perceptual Similarity)',
        'noise': 'Noise Level',
        'noise_level': 'Noise',
        'artifacts': 'Artifacts (Blocking + Ringing)',
        'artifact_total': 'Artifacts',
        'edge_density': 'Edge Density',
        'local_quality_std': 'Local Quality Std Dev',
        'local_quality_mean': 'Local Quality Mean',
        'high_freq_ratio': 'High Frequency Ratio',
        'entropy': 'Entropy (Information)',
        'contrast': 'Contrast',
        'texture_complexity': 'Texture Complexity',
        'histogram_corr': 'Histogram Correlation',
        'delta_e': 'Color Difference ΔE',
        'lab_lightness': 'LAB Lightness',
        'frequency': 'Frequency',
        'pc1': 'PC1',
        'pc2': 'PC2',
        'num_components': 'Number of Components',
        'cumulative_variance': 'Cumulative Variance Ratio',
        'ai_model': 'AI Model',

        # Annotations
        'median': 'Median',
        'regression_line': 'Regression Line',
        'ideal_region': 'Ideal Region',
        'ideal_region_detail': 'Ideal Region\n(High Fidelity & Clarity)',
        'over_processing': 'Over-processing\n(Low Fidelity)\nHallucination Risk',
        'hallucination_suspected': 'Hallucination Suspected',
        'over_processing_suspected': 'Over-processing Suspected',
        'unnatural_edges_suspected': 'Unnatural Edges Suspected',
        'psnr_threshold': 'PSNR Threshold',
        'ssim_threshold': 'SSIM Threshold',
        'noise_inverted': 'Noise\n(Inverted)',
        'artifact_inverted': 'Artifacts\n(Inverted)',
        'line_95': '95% Line',
        'distribution_of': ' Distribution',
        'cumulative_variance_prefix': 'Cumulative variance:',
        'cases': 'cases',
    }
}


def get_label(key, lang='en'):
    """
    Get translated label

    Args:
        key: Translation key
        lang: Language code ('ja' or 'en')

    Returns:
        Translated string
    """
    if lang not in TRANSLATIONS:
        lang = 'en'  # Default to English

    return TRANSLATIONS[lang].get(key, key)


def get_font_family(lang='en'):
    """
    Get appropriate font family for the language

    Args:
        lang: Language code ('ja' or 'en')

    Returns:
        List of font names
    """
    if lang not in TRANSLATIONS:
        lang = 'en'

    return TRANSLATIONS[lang]['font_family']
