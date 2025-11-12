"""
統計分析スクリプト：バッチ処理結果から閾値を決定（26パターン対応版）

26パターンのハルシネーション検出:
- 9つの組み合わせパターン（複合異常）
- 16の単独閾値判定（各指標の異常値）

使い方:
python analyze_results.py results/batch_analysis.csv
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from translations import get_label, get_font_family, I18n

# Global language setting (will be set by command line argument)
LANG = 'en'
i18n = None


def analyze_batch_results(csv_file, lang='en'):
    """
    バッチ処理結果の統計分析
    """
    # Set global language and i18n
    global LANG, i18n
    LANG = lang
    i18n = I18n(lang=lang)

    # Set font based on language
    plt.rcParams['font.family'] = get_font_family(LANG)
    plt.rcParams['axes.unicode_minus'] = False

    # CSV読み込み
    df = pd.read_csv(csv_file)

    print(f"\n{'='*80}")
    print(i18n.t('stats_analysis.report_header'))
    print(f"{'='*80}")
    print(i18n.t('stats_analysis.data_file').format(file=csv_file))
    print(i18n.t('stats_analysis.image_count').format(count=df['image_id'].nunique()))
    print(i18n.t('stats_analysis.model_count').format(count=df['model'].nunique()))
    print(i18n.t('stats_analysis.total_data').format(count=len(df)))
    print(f"{'='*80}\n")

    # 出力ディレクトリ作成（タイムスタンプ付きサブフォルダ）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('analysis_output') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 基本統計量
    print_basic_statistics(df)

    # 2. モデル別比較
    compare_models(df, output_dir)

    # 3. 相関分析
    analyze_correlations(df, output_dir)

    # 4. 閾値提案
    suggest_thresholds(df, output_dir)

    # 5. ハルシネーション検出ロジック提案
    suggest_hallucination_logic(df, output_dir)

    # 6. 研究用プロット生成
    generate_research_plots(df, output_dir, csv_file)

    print(i18n.t('stats_analysis.analysis_complete'))
    print(i18n.t('stats_analysis.results_saved').format(dir=output_dir))

    return str(output_dir)


def print_basic_statistics(df):
    """
    基本統計量の表示
    """
    global i18n

    print(i18n.t('stats_analysis.basic_stats_header'))
    print(f"{'='*80}")

    # 16項目（total_scoreは除外）
    metrics = ['ssim', 'ms_ssim', 'psnr', 'lpips', 'sharpness', 'contrast',
               'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
               'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
               'histogram_corr', 'lab_L_mean']

    stats = df[metrics].describe().T
    stats.columns = [
        i18n.t('stats_analysis.stats_col_count'),
        i18n.t('stats_analysis.stats_col_mean'),
        i18n.t('stats_analysis.stats_col_std'),
        i18n.t('stats_analysis.stats_col_min'),
        i18n.t('stats_analysis.stats_col_25'),
        i18n.t('stats_analysis.stats_col_50'),
        i18n.t('stats_analysis.stats_col_75'),
        i18n.t('stats_analysis.stats_col_max')
    ]

    print(stats.round(4).to_string())
    print(f"{'='*80}\n")


def compare_models(df, output_dir):
    """
    モデル別比較
    """
    global i18n

    print(i18n.t('stats_analysis.model_ranking_header'))
    print(f"{'='*80}")

    # 主要指標でグループ化（total_score除外）
    model_comparison = df.groupby('model').agg({
        'ssim': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'lpips': ['mean', 'std'],
        'noise': ['mean', 'std'],
        'artifact_total': ['mean', 'std'],
        'sharpness': ['mean', 'std'],
        'edge_density': ['mean', 'std']
    }).round(4)

    # SSIMでソート（高いほど良い）
    model_comparison = model_comparison.sort_values(('ssim', 'mean'), ascending=False)

    print(model_comparison.to_string())
    print(f"{'='*80}\n")

    # CSV保存
    model_comparison.to_csv(output_dir / 'model_comparison.csv', encoding='utf-8-sig')

    # Visualization: Model SSIM Score
    plt.figure(figsize=(12, 6))
    model_scores = df.groupby('model')['ssim'].mean().sort_values(ascending=False)

    plt.bar(range(len(model_scores)), model_scores.values)
    plt.xticks(range(len(model_scores)), model_scores.index, rotation=45, ha='right')
    plt.ylabel('SSIM (Average)')
    # TITLE_BOTTOM:{get_label('model_comparison', LANG)}
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('model_comparison', LANG), ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'model_scores.png', dpi=150)
    plt.close()

    print(i18n.t('stats_analysis.graph_saved').format(path=f"{output_dir}/model_scores.png"))


def analyze_correlations(df, output_dir):
    """
    17項目間の相関分析
    """
    global i18n

    print(i18n.t('stats_analysis.correlation_header'))
    print(f"{'='*80}")

    # 数値列のみ抽出（total_score除外）
    numeric_cols = ['ssim', 'psnr', 'lpips', 'ms_ssim', 'sharpness', 'contrast',
                    'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
                    'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
                    'histogram_corr', 'lab_L_mean']

    corr_matrix = df[numeric_cols].corr()

    # 相関行列をヒートマップ表示
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    # TITLE_BOTTOM:{'Correlation Matrix of 16 Metrics', fontsize=16, fontweight='bold'}
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150)
    plt.close()

    print(i18n.t('stats_analysis.correlation_matrix_saved').format(path=f"{output_dir}/correlation_matrix.png"))

    # 高相関ペアを表示
    print(i18n.t('stats_analysis.high_correlation_pairs'))
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr.append({
                    'metric1': corr_matrix.columns[i],
                    'metric2': corr_matrix.columns[j],
                    'correlation': corr_value
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('correlation', ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print(i18n.t('stats_analysis.none'))

    print(f"{'='*80}\n")


def suggest_thresholds(df, output_dir):
    """
    Suggest evidence-based thresholds
    """
    global LANG

    print(f"\n[TIP] Recommended Threshold Suggestions:")
    print(f"{'='*80}")

    thresholds = {}

    # Determine thresholds from statistical values for each metric
    # Propose thresholds for all 17 metrics
    metrics_config = {
        'ssim': {'direction': 'high', 'percentile': 25, 'name': get_label('ssim', LANG)},
        'ms_ssim': {'direction': 'high', 'percentile': 25, 'name': 'MS-SSIM (Multi-Scale SSIM)'},
        'psnr': {'direction': 'high', 'percentile': 25, 'name': 'PSNR (Peak Signal-to-Noise Ratio)'},
        'lpips': {'direction': 'low', 'percentile': 75, 'name': get_label('lpips', LANG)},
        'sharpness': {'direction': 'high', 'percentile': 25, 'name': get_label('sharpness_clarity', LANG)},
        'contrast': {'direction': 'high', 'percentile': 25, 'name': get_label('contrast', LANG)},
        'entropy': {'direction': 'high', 'percentile': 25, 'name': 'Entropy (Information Content)'},
        'noise': {'direction': 'low', 'percentile': 75, 'name': get_label('noise', LANG)},
        'edge_density': {'direction': 'high', 'percentile': 25, 'name': get_label('edge_density', LANG)},
        'artifact_total': {'direction': 'low', 'percentile': 75, 'name': get_label('artifact_total', LANG)},
        'delta_e': {'direction': 'low', 'percentile': 75, 'name': 'Color Difference (ΔE)'},
        'high_freq_ratio': {'direction': 'high', 'percentile': 25, 'name': get_label('high_freq_ratio', LANG)},
        'texture_complexity': {'direction': 'high', 'percentile': 25, 'name': get_label('texture_complexity', LANG)},
        'local_quality_mean': {'direction': 'high', 'percentile': 25, 'name': get_label('local_quality_mean', LANG)},
        'histogram_corr': {'direction': 'high', 'percentile': 25, 'name': get_label('histogram_corr', LANG)},
        'lab_L_mean': {'direction': 'neutral', 'percentile': 50, 'name': 'LAB Lightness (Reference)'},
    }

    for metric, config in metrics_config.items():
        data = df[metric].dropna()

        if config['direction'] == 'neutral':
            # Neutral metric (e.g., lightness): Display median as reference value
            threshold = np.percentile(data, config['percentile'])
            condition = f"Reference: {threshold:.4f}"
        elif config['direction'] == 'high':
            # Higher is better: Recommend 25th percentile or above
            threshold = np.percentile(data, config['percentile'])
            condition = f">= {threshold:.4f}"
        else:
            # Lower is better: Recommend 75th percentile or below
            threshold = np.percentile(data, config['percentile'])
            condition = f"<= {threshold:.4f}"

        thresholds[metric] = {
            'name': config['name'],
            'threshold': threshold,
            'condition': condition,
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        }

        print(f"{config['name']:35s}: {condition:20s} (mean: {data.mean():.4f}, std: {data.std():.4f})")

    print(f"{'='*80}")
    print(f"[TIP] Interpretation:")
    print(f"   - These thresholds are based on statistical distribution of all data")
    print(f"   - 25th percentile = Standard to pass top 75% quality")
    print(f"   - 75th percentile = Standard to pass bottom 75% quality")
    print(f"{'='*80}\n")

    # Save JSON
    import json
    with open(output_dir / 'recommended_thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] Thresholds saved: {output_dir}/recommended_thresholds.json\n")


def suggest_hallucination_logic(df, output_dir):
    """
    ハルシネーション検出ロジックの提案（26パターン完全対応版）
    - 9つの組み合わせパターン
    - 17の単独閾値判定
    """
    global i18n

    print(i18n.t('stats_analysis.hallucination_detection_header'))
    print(f"{'='*80}")

    # 検出カウント用
    detection_count = pd.Series(0, index=df.index)
    detected_patterns = {idx: [] for idx in df.index}
    pattern_stats = {}

    # ========== 組み合わせパターン（9つ） ==========
    print(i18n.t('stats_analysis.combination_patterns'))
    print(f"{'='*80}")

    # === パターン1: SSIM高 × PSNR低（2方式統合） ===
    # 方法A: 固定閾値
    pattern1a = df[(df['ssim'] > 0.97) & (df['psnr'] < 25)]
    # 方法B: 動的閾値
    ssim_high = df['ssim'].quantile(0.75)
    psnr_low = df['psnr'].quantile(0.25)
    pattern1b = df[(df['ssim'] >= ssim_high) & (df['psnr'] <= psnr_low)]
    # 統合
    pattern1 = pd.concat([pattern1a, pattern1b]).drop_duplicates()
    detection_count[pattern1.index] += 1
    for idx in pattern1.index:
        detected_patterns[idx].append('P1:SSIM高×PSNR低')
    pattern_stats['P1'] = {'count': len(pattern1), 'rate': len(pattern1)/len(df)*100}

    print(i18n.t('stats_analysis.p1_title'))
    print(i18n.t('stats_analysis.fixed_threshold').format(count=len(pattern1a)))
    print(i18n.t('stats_analysis.dynamic_threshold').format(ssim=ssim_high, psnr=psnr_low, count=len(pattern1b)))
    print(i18n.t('stats_analysis.p1_unified').format(count=len(pattern1), percent=len(pattern1)/len(df)*100))
    print(i18n.t('stats_analysis.p1_risk'))

    # === パターン2: Sharpness高 × Noise高 ===
    sharp_high = df['sharpness'].quantile(0.75)
    noise_high = df['noise'].quantile(0.75)
    pattern2 = df[(df['sharpness'] > sharp_high) & (df['noise'] > noise_high)]
    detection_count[pattern2.index] += 1
    for idx in pattern2.index:
        detected_patterns[idx].append('P2:シャープ高×Noise高')
    pattern_stats['P2'] = {'count': len(pattern2), 'rate': len(pattern2)/len(df)*100}

    print(i18n.t('stats_analysis.p2_title'))
    print(i18n.t('stats_analysis.p2_condition').format(sharp=sharp_high, noise=noise_high))
    print(i18n.t('stats_analysis.p2_match').format(count=len(pattern2), percent=len(pattern2)/len(df)*100))
    print(i18n.t('stats_analysis.p2_risk'))

    # === パターン3: Edge Density高 × 局所品質低 ===
    edge_90 = df['edge_density'].quantile(0.90)
    quality_25 = df['local_quality_mean'].quantile(0.25)
    pattern3 = df[(df['edge_density'] > edge_90) & (df['local_quality_mean'] < quality_25)]
    detection_count[pattern3.index] += 1
    for idx in pattern3.index:
        detected_patterns[idx].append('P3:エッジ高×品質低')
    pattern_stats['P3'] = {'count': len(pattern3), 'rate': len(pattern3)/len(df)*100}

    print(i18n.t('stats_analysis.p3_title'))
    print(i18n.t('stats_analysis.p3_condition').format(edge=edge_90, quality=quality_25))
    print(i18n.t('stats_analysis.p3_match').format(count=len(pattern3), percent=len(pattern3)/len(df)*100))
    print(i18n.t('stats_analysis.p3_risk'))

    # === パターン4: Artifacts異常高 ===
    artifact_90 = df['artifact_total'].quantile(0.90)
    pattern4 = df[df['artifact_total'] > artifact_90]
    detection_count[pattern4.index] += 1
    for idx in pattern4.index:
        detected_patterns[idx].append('P4:Artifacts高')
    pattern_stats['P4'] = {'count': len(pattern4), 'rate': len(pattern4)/len(df)*100}

    print(i18n.t('stats_analysis.p4_title'))
    print(i18n.t('stats_analysis.p4_condition').format(artifact=artifact_90))
    print(i18n.t('stats_analysis.p4_match').format(count=len(pattern4), percent=len(pattern4)/len(df)*100))
    print(i18n.t('stats_analysis.p4_risk'))

    # === パターン5: LPIPS高 × SSIM高 ===
    lpips_75 = df['lpips'].quantile(0.75)
    ssim_75 = df['ssim'].quantile(0.75)
    pattern5 = df[(df['lpips'] > lpips_75) & (df['ssim'] > ssim_75)]
    detection_count[pattern5.index] += 1
    for idx in pattern5.index:
        detected_patterns[idx].append('P5:LPIPS高×SSIM高')
    pattern_stats['P5'] = {'count': len(pattern5), 'rate': len(pattern5)/len(df)*100}

    print(i18n.t('stats_analysis.p5_title'))
    print(i18n.t('stats_analysis.p5_condition').format(lpips=lpips_75, ssim=ssim_75))
    print(i18n.t('stats_analysis.p5_match').format(count=len(pattern5), percent=len(pattern5)/len(df)*100))
    print(i18n.t('stats_analysis.p5_risk'))

    # === パターン6: 局所品質ばらつき大 ===
    if 'local_quality_std' in df.columns:
        quality_std_75 = df['local_quality_std'].quantile(0.75)
        pattern6 = df[df['local_quality_std'] > quality_std_75]
        detection_count[pattern6.index] += 1
        for idx in pattern6.index:
            detected_patterns[idx].append('P6:品質ばらつき大')
        pattern_stats['P6'] = {'count': len(pattern6), 'rate': len(pattern6)/len(df)*100}

        print(i18n.t('stats_analysis.p6_title'))
        print(i18n.t('stats_analysis.p6_condition').format(std=quality_std_75))
        print(i18n.t('stats_analysis.p6_match').format(count=len(pattern6), percent=len(pattern6)/len(df)*100))
        print(i18n.t('stats_analysis.p6_risk'))
    else:
        pattern_stats['P6'] = {'count': 0, 'rate': 0}
        print(i18n.t('stats_analysis.local_quality_no_data'))

    # === パターン7: Entropy低 × High-Freq高 ===
    entropy_25 = df['entropy'].quantile(0.25)
    highfreq_75 = df['high_freq_ratio'].quantile(0.75)
    pattern7 = df[(df['entropy'] < entropy_25) & (df['high_freq_ratio'] > highfreq_75)]
    detection_count[pattern7.index] += 1
    for idx in pattern7.index:
        detected_patterns[idx].append('P7:Entropy低×高周波高')
    pattern_stats['P7'] = {'count': len(pattern7), 'rate': len(pattern7)/len(df)*100}

    print(i18n.t('stats_analysis.p7_title'))
    print(i18n.t('stats_analysis.p7_condition').format(entropy=entropy_25, freq=highfreq_75))
    print(i18n.t('stats_analysis.p7_match').format(count=len(pattern7), percent=len(pattern7)/len(df)*100))
    print(i18n.t('stats_analysis.p7_risk'))

    # === パターン8: Contrast異常 × Histogram相関低 ===
    contrast_90 = df['contrast'].quantile(0.90)
    histcorr_25 = df['histogram_corr'].quantile(0.25)
    pattern8 = df[(df['contrast'] > contrast_90) & (df['histogram_corr'] < histcorr_25)]
    detection_count[pattern8.index] += 1
    for idx in pattern8.index:
        detected_patterns[idx].append('P8:Contrast異常×Hist相関低')
    pattern_stats['P8'] = {'count': len(pattern8), 'rate': len(pattern8)/len(df)*100}

    print(i18n.t('stats_analysis.p8_title'))
    print(i18n.t('stats_analysis.p8_condition').format(contrast=contrast_90, hist=histcorr_25))
    print(i18n.t('stats_analysis.p8_match').format(count=len(pattern8), percent=len(pattern8)/len(df)*100))
    print(i18n.t('stats_analysis.p8_risk'))

    # === パターン9: MS-SSIM低 × SSIM低 ===
    msssim_25 = df['ms_ssim'].quantile(0.25)
    ssim_25 = df['ssim'].quantile(0.25)
    pattern9 = df[(df['ms_ssim'] < msssim_25) & (df['ssim'] < ssim_25)]
    detection_count[pattern9.index] += 1
    for idx in pattern9.index:
        detected_patterns[idx].append('P9:MS-SSIM低×SSIM低')
    pattern_stats['P9'] = {'count': len(pattern9), 'rate': len(pattern9)/len(df)*100}

    print(i18n.t('stats_analysis.p9_title'))
    print(i18n.t('stats_analysis.p9_condition').format(msssim=msssim_25, ssim=ssim_25))
    print(i18n.t('stats_analysis.p9_match').format(count=len(pattern9), percent=len(pattern9)/len(df)*100))
    print(i18n.t('stats_analysis.p9_risk'))

    # ========== 単独パターン（17項目） ==========
    print(f"\n{'='*80}")
    print(i18n.t('stats_analysis.single_threshold_patterns'))
    print(f"{'='*80}")

    single_pattern_count = 0

    # 高い方が良い指標（異常に低い = 下位10%）
    if LANG == 'en':
        high_is_good = [
            ('ssim', 'Low SSIM'), ('ms_ssim', 'Low MS-SSIM'), ('psnr', 'Low PSNR'),
            ('sharpness', 'Low Sharpness'), ('contrast', 'Low Contrast'), ('entropy', 'Low Entropy'),
            ('edge_density', 'Low EdgeDensity'), ('high_freq_ratio', 'Low HighFreq'),
            ('texture_complexity', 'Low Texture'), ('local_quality_mean', 'Low LocalQuality'),
            ('histogram_corr', 'Low HistCorr')
        ]
    else:
        high_is_good = [
            ('ssim', 'SSIM低'), ('ms_ssim', 'MS-SSIM低'), ('psnr', 'PSNR低'),
            ('sharpness', 'Sharpness低'), ('contrast', 'Contrast低'), ('entropy', 'Entropy低'),
            ('edge_density', 'EdgeDensity低'), ('high_freq_ratio', 'HighFreq低'),
            ('texture_complexity', 'Texture低'), ('local_quality_mean', 'LocalQuality低'),
            ('histogram_corr', 'HistCorr低')
        ]

    print(i18n.t('stats_analysis.high_is_better'))
    for col, name in high_is_good:
        if col in df.columns:
            threshold = df[col].quantile(0.10)
            detected = df[df[col] < threshold]
            detection_count[detected.index] += 1
            for idx in detected.index:
                detected_patterns[idx].append(f'単独:{name}')
            single_pattern_count += len(detected)
            print(i18n.t('stats_analysis.single_pattern_format').format(
                name=name, threshold=threshold, count=len(detected), percent=len(detected)/len(df)*100))

    # 低い方が良い指標（異常に高い = 上位10%）
    if LANG == 'en':
        low_is_good = [
            ('lpips', 'High LPIPS'), ('noise', 'High Noise'),
            ('artifact_total', 'High Artifacts'), ('delta_e', 'High DeltaE')
        ]
    else:
        low_is_good = [
            ('lpips', 'LPIPS高'), ('noise', 'Noise高'),
            ('artifact_total', 'Artifacts高'), ('delta_e', 'DeltaE高')
        ]

    print(i18n.t('stats_analysis.low_is_better'))
    for col, name in low_is_good:
        if col in df.columns:
            threshold = df[col].quantile(0.90)
            detected = df[df[col] > threshold]
            detection_count[detected.index] += 1
            for idx in detected.index:
                detected_patterns[idx].append(f'単独:{name}')
            single_pattern_count += len(detected)
            print(i18n.t('stats_analysis.single_pattern_format_high').format(
                name=name, threshold=threshold, count=len(detected), percent=len(detected)/len(df)*100))

    if 'local_quality_std' in df.columns:
        threshold = df['local_quality_std'].quantile(0.90)
        detected = df[df['local_quality_std'] > threshold]
        detection_count[detected.index] += 1
        local_quality_std_name = 'High LocalQualityStd' if LANG == 'en' else 'LocalQualityStd高'
        for idx in detected.index:
            detected_patterns[idx].append(f'単独:{local_quality_std_name}')
        single_pattern_count += len(detected)
        print(i18n.t('stats_analysis.single_pattern_format_high').format(
            name=local_quality_std_name, threshold=threshold, count=len(detected), percent=len(detected)/len(df)*100))

    print(i18n.t('stats_analysis.single_pattern_total').format(count=single_pattern_count))

    # ========== 総合リスクスコア計算 ==========
    print(f"\n{'='*80}")
    print(i18n.t('stats_analysis.risk_score_header'))
    print(f"{'='*80}")

    # 信頼度分類（多数決）
    high_confidence = df[detection_count >= 5]  # 5パターン以上
    medium_confidence = df[(detection_count >= 3) & (detection_count < 5)]  # 3-4パターン
    low_confidence = df[(detection_count >= 1) & (detection_count < 3)]  # 1-2パターン
    no_detection = df[detection_count == 0]  # 検出なし（正常）

    print(i18n.t('stats_analysis.confidence_classification'))
    print(i18n.t('stats_analysis.high_confidence').format(count=len(high_confidence), percent=len(high_confidence)/len(df)*100))
    print(i18n.t('stats_analysis.medium_confidence').format(count=len(medium_confidence), percent=len(medium_confidence)/len(df)*100))
    print(i18n.t('stats_analysis.low_confidence').format(count=len(low_confidence), percent=len(low_confidence)/len(df)*100))
    print(i18n.t('stats_analysis.normal').format(count=len(no_detection), percent=len(no_detection)/len(df)*100))

    # DataFrameに結果を追加
    df['detection_count'] = detection_count
    df['detected_patterns'] = df.index.map(lambda idx: ', '.join(detected_patterns[idx]) if detected_patterns[idx] else 'None')

    # 信頼度レベル
    df['confidence_level'] = 'Normal'
    df.loc[detection_count >= 1, 'confidence_level'] = 'Low'
    df.loc[detection_count >= 3, 'confidence_level'] = 'Medium'
    df.loc[detection_count >= 5, 'confidence_level'] = 'High'

    # リスク付きCSV保存
    output_csv = output_dir / 'results_with_26pattern_detection.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(i18n.t('stats_analysis.detection_results_saved').format(path=output_csv))

    # サマリーCSV保存
    summary_data = {
        'pattern_name': [],
        'detection_count': [],
        'detection_rate_%': []
    }

    for p_name, stats in pattern_stats.items():
        summary_data['pattern_name'].append(p_name)
        summary_data['detection_count'].append(stats['count'])
        summary_data['detection_rate_%'].append(stats['rate'])

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'pattern_detection_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(i18n.t('stats_analysis.pattern_summary_saved').format(path=summary_path))

    print(f"{'='*80}\n")


def generate_research_plots(df, output_dir, csv_file):
    """
    研究用プロット画像を生成（論文・発表用）

    Args:
        df: バッチ処理結果のDataFrame
        output_dir: 出力ディレクトリ
        csv_file: CSVファイルのパス（BBI計算でdetailedディレクトリ検索に使用）
    """
    global i18n, LANG

    print(i18n.t('stats_analysis.research_plots_generating'))
    print(f"{'='*80}")

    # 1. Sharpness vs PSNR 散布図（AI Modelの戦略を示す）
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['psnr'], model_data['sharpness'],
                   label=model, alpha=0.6, s=50)

    plt.xlabel(get_label('psnr_fidelity', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('sharpness', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('strategy_map', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 戦略領域の注釈
    plt.axhline(y=df['sharpness'].median(), color='red', linestyle='--', alpha=0.3, label=get_label('median', LANG))
    plt.axvline(x=df['psnr'].median(), color='red', linestyle='--', alpha=0.3)

    # 領域ラベル
    max_psnr = df['psnr'].max()
    max_sharp = df['sharpness'].max()
    plt.text(max_psnr * 0.95, max_sharp * 0.95, get_label('ideal_region_detail', LANG),
             fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.text(df['psnr'].min() * 1.05, max_sharp * 0.95, get_label('over_processing', LANG),
             fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plot1_path = output_dir / 'strategy_map_sharpness_vs_psnr.png'
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('strategy_map', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_strategy_map').format(path=plot1_path))


    # 2. LPIPS 箱ひげ図（安定性を示す）
    plt.figure(figsize=(10, 6))

    lpips_data = [df[df['model'] == model]['lpips'].values for model in df['model'].unique()]
    models = df['model'].unique()

    bp = plt.boxplot(lpips_data, labels=models, patch_artist=True,
                     showmeans=True, meanline=True)

    # カラーリング
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel(get_label('lpips', LANG), fontsize=14, fontweight='bold')
    plt.xlabel(get_label('ai_model', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'LPIPS Distribution by Model (Stability)\nSmaller box = More stable', fontsize=16, fontweight='bold'}
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plot2_path = output_dir / 'stability_lpips_boxplot.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_lpips_boxplot').format(path=plot2_path))


    # 3. SSIM vs PSNR 散布図（相関確認・異常検出）
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ssim'], model_data['psnr'],
                   label=model, alpha=0.6, s=50)

    plt.xlabel(get_label('ssim', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('psnr', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'SSIM vs PSNR Correlation\nOutliers = Hallucination Candidates', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 近似直線
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['ssim'], df['psnr'])
    x_line = np.array([df['ssim'].min(), df['ssim'].max()])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r--', label=f'Regression Line (R²={r_value**2:.3f})', linewidth=2)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plot3_path = output_dir / 'correlation_ssim_vs_psnr.png'
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'SSIM vs PSNR Correlation\nOutliers = Hallucination Candidates', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_ssim_psnr_correlation').format(path=plot3_path))


    # 4. Noise vs Artifacts 散布図
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['noise'], model_data['artifact_total'],
                   label=model, alpha=0.6, s=50)

    plt.xlabel(get_label('noise', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('artifacts', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Noise vs Artifacts\nLower-left is ideal (both low)', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Ideal Regionの表示
    low_noise = df['noise'].quantile(0.25)
    low_artifact = df['artifact_total'].quantile(0.25)
    plt.axvline(x=low_noise, color='green', linestyle='--', alpha=0.3)
    plt.axhline(y=low_artifact, color='green', linestyle='--', alpha=0.3)
    plt.fill_between([0, low_noise], 0, low_artifact, alpha=0.1, color='green', label=get_label('ideal_region', LANG))
    plt.legend(fontsize=12)

    plt.tight_layout()
    plot4_path = output_dir / 'quality_noise_vs_artifact.png'
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('noise_artifact', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_noise_artifact').format(path=plot4_path))


    # 5. モデル別レーダーチャート（主要6指標）
    fig = plt.figure(figsize=(14, 10))

    categories = ['SSIM', 'PSNR', get_label('sharpness_clarity', LANG), get_label('edge_density', LANG), get_label('noise_inverted', LANG), get_label('artifact_inverted', LANG)]
    num_vars = len(categories)

    # 正規化（0-1スケール）
    df_norm = df.copy()
    df_norm['ssim_norm'] = df['ssim']
    df_norm['psnr_norm'] = df['psnr'] / df['psnr'].max()
    df_norm['sharpness_norm'] = df['sharpness'] / df['sharpness'].max()
    df_norm['edge_norm'] = df['edge_density'] / df['edge_density'].max()
    df_norm['noise_norm'] = 1 - (df['noise'] / df['noise'].max())  # 反転（少ない方が良い）
    df_norm['artifact_norm'] = 1 - (df['artifact_total'] / df['artifact_total'].max())  # 反転

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(111, polar=True)

    for model in df['model'].unique():
        model_data = df_norm[df_norm['model'] == model]
        values = [
            model_data['ssim_norm'].mean(),
            model_data['psnr_norm'].mean(),
            model_data['sharpness_norm'].mean(),
            model_data['edge_norm'].mean(),
            model_data['noise_norm'].mean(),
            model_data['artifact_norm'].mean()
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Profile (Radar Chart)\nOuter = Better', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plot5_path = output_dir / 'radar_chart_model_comparison.png'
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.radar_chart_saved').format(path=plot5_path))


    # 6. 17項目のバイオリンプロット（分布の可視化）
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    # TITLE_BOTTOM:{get_label('violin_plot', LANG), fontsize=20, fontweight='bold'}

    metrics_for_violin = [
        'ssim', 'ms_ssim', 'psnr', 'lpips', 'sharpness', 'contrast',
        'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
        'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
        'histogram_corr', 'lab_L_mean'
    ]

    for i, metric in enumerate(metrics_for_violin):
        ax = axes[i // 6, i % 6]

        violin_data = [df[df['model'] == model][metric].values for model in df['model'].unique()]
        parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)

        # カラーリング
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(df['model'].unique()) + 1))
        ax.set_xticklabels(df['model'].unique(), rotation=45, ha='right', fontsize=8)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # 余った軸を非表示
    for i in range(len(metrics_for_violin), 18):
        axes[i // 6, i % 6].axis('off')

    plt.tight_layout()
    plot6_path = output_dir / 'violin_plots_all_metrics.png'
    plt.savefig(plot6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_violin').format(path=plot6_path))

    # ===== ハルシネーション検出系プロット =====

    # 7. SSIM高 × PSNR低 のHallucination Suspected可視化
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ssim'], model_data['psnr'],
                   label=model, alpha=0.6, s=50)

    # Hallucination Suspected領域を赤で強調
    ssim_high = df['ssim'].quantile(0.75)
    psnr_low = df['psnr'].quantile(0.25)
    hallucination_candidates = df[(df['ssim'] >= ssim_high) & (df['psnr'] <= psnr_low)]

    if len(hallucination_candidates) > 0:
        plt.scatter(hallucination_candidates['ssim'], hallucination_candidates['psnr'],
                   color='red', s=200, marker='x', linewidths=3,
                   label=f"{get_label('hallucination_suspected', LANG)} ({len(hallucination_candidates)} {get_label('cases', LANG)})", zorder=10)

    plt.axhline(y=psnr_low, color='orange', linestyle='--', alpha=0.5, label=f"{get_label('psnr_threshold', LANG)} ({psnr_low:.1f})")
    plt.axvline(x=ssim_high, color='orange', linestyle='--', alpha=0.5, label=f"{get_label('ssim_threshold', LANG)} ({ssim_high:.3f})")

    plt.xlabel(get_label('ssim', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('psnr_db', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Hallucination Detection: High SSIM & Low PSNR\nLower-right = Mimicked structure, low fidelity', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_ssim_high_psnr_low.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.hallucination_plot1'))


    # 8. Sharpness × Noise（過剰処理検出）
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['sharpness'], model_data['noise'],
                   label=model, alpha=0.6, s=50)

    sharp_high = df['sharpness'].quantile(0.75)
    noise_high = df['noise'].quantile(0.75)
    over_processed = df[(df['sharpness'] >= sharp_high) & (df['noise'] >= noise_high)]

    if len(over_processed) > 0:
        plt.scatter(over_processed['sharpness'], over_processed['noise'],
                   color='red', s=200, marker='x', linewidths=3,
                   label=f"{get_label('over_processing_suspected', LANG)} ({len(over_processed)} {get_label('cases', LANG)})", zorder=10)

    plt.axhline(y=noise_high, color='orange', linestyle='--', alpha=0.5)
    plt.axvline(x=sharp_high, color='orange', linestyle='--', alpha=0.5)

    plt.xlabel(get_label('sharpness_clarity', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('noise', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Over-processing Detection: High Sharpness & Noise\nUpper-right = Noise amplified by sharpening', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_sharpness_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.hallucination_plot2'))


    # 9. Edge Density × Local Quality Std Dev
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['edge_density'], model_data['local_quality_std'],
                   label=model, alpha=0.6, s=50)

    edge_high = df['edge_density'].quantile(0.75)
    local_std_high = df['local_quality_std'].quantile(0.75)
    unnatural_edges = df[(df['edge_density'] >= edge_high) & (df['local_quality_std'] >= local_std_high)]

    if len(unnatural_edges) > 0:
        plt.scatter(unnatural_edges['edge_density'], unnatural_edges['local_quality_std'],
                   color='red', s=200, marker='x', linewidths=3,
                   label=f"{get_label('unnatural_edges_suspected', LANG)} ({len(unnatural_edges)} {get_label('cases', LANG)})", zorder=10)

    plt.xlabel(get_label('edge_density', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('local_quality_std', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Unnatural Edge Detection: High Edge & Quality Variance\nUpper-right = Uneven edge addition', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_edge_vs_local_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.hallucination_plot3'))


    # 10. 高周波成分 × Entropy
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['high_freq_ratio'], model_data['entropy'],
                   label=model, alpha=0.6, s=50)

    # Regression Line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['high_freq_ratio'], df['entropy'])
    x_line = np.array([df['high_freq_ratio'].min(), df['high_freq_ratio'].max()])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r--', label=f'Regression Line (R²={r_value**2:.3f})', linewidth=2)

    plt.xlabel(get_label('high_freq_ratio', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('entropy', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Artificial Pattern Detection: High Freq vs Entropy\nOutliers = Repetitive pattern suspected', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_highfreq_vs_entropy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.hallucination_plot4'))


    # ===== 品質トレードオフ系プロット =====

    # 11. SSIM × Noise
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ssim'], model_data['noise'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('ssim', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('noise', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('tradeoff_ssim_noise', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tradeoff_ssim_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_tradeoff_1'))


    # 12. PSNR × Contrast
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['psnr'], model_data['contrast'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('psnr_db', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('contrast', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('tradeoff_psnr_contrast', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('tradeoff_psnr_contrast', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_psnr_vs_contrast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_tradeoff_2'))


    # 13. Sharpness × Artifacts
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['sharpness'], model_data['artifact_total'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('sharpness_clarity', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('artifacts', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('tradeoff_sharpness_artifact', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('tradeoff_sharpness_artifact', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_sharpness_vs_artifact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_tradeoff_3'))


    # 14. LPIPS × MS-SSIM
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['lpips'], model_data['ms_ssim'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('lpips', LANG), fontsize=14, fontweight='bold')
    plt.ylabel('MS-SSIM', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Perception vs Structure: LPIPS vs MS-SSIM\nNegative correlation expected', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('tradeoff_lpips_msssim', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_lpips_vs_msssim.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_tradeoff_4'))


    # 15. テクスチャ × 高周波成分
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['texture_complexity'], model_data['high_freq_ratio'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('texture_complexity', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('high_freq_ratio', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('tradeoff_texture_freq', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('tradeoff_texture_freq', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_texture_vs_highfreq.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_tradeoff_5'))


    # ===== 医療画像特化系プロット =====

    # 16. Contrast × Histogram Correlation
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['contrast'], model_data['histogram_corr'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('contrast', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('histogram_corr', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('medical_contrast_histogram', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('medical_contrast_histogram', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_contrast_vs_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_medical_1'))


    # 17. Edge Density × Local Quality Mean
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['edge_density'], model_data['local_quality_mean'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('edge_density', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('local_quality_mean', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('medical_edge_quality', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('medical_edge_quality', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_edge_vs_local_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_medical_2'))


    # 18. Noise × Local Quality Std Dev
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['noise'], model_data['local_quality_std'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('noise', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('local_quality_std', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('medical_noise_std', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('medical_noise_std', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_noise_vs_local_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_medical_3'))


    # 19. 色差ΔE × LAB Lightness
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['delta_e'], model_data['lab_L_mean'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel(get_label('delta_e', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('lab_lightness', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{get_label('medical_deltae_lab', LANG), fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('medical_deltae_lab', LANG), fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_deltae_vs_lab.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_medical_4'))


    # ===== 分布・PCA系プロット =====

    # 20. Total Scoreヒストグラム（削除: total_scoreは分析対象外）
    print(i18n.t('stats_analysis.skip_histogram'))


    # 21. 主成分分析（PCA）プロット
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # 16項目を標準化（total_score除外）
    metrics_for_pca = ['ssim', 'ms_ssim', 'psnr', 'lpips', 'sharpness', 'contrast',
                       'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
                       'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
                       'histogram_corr', 'lab_L_mean']

    X = df[metrics_for_pca].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA実行
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        mask = df['model'] == model
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=model, alpha=0.6, s=50)

    plt.xlabel(f"{get_label('pc1', LANG)} ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.ylabel(f"{get_label('pc2', LANG)} ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{f'Principal Component Analysis (PCA): 16 Metrics to 2D\nCumulative variance: {sum(pca.explained_variance_ratio_)*100:.1f}%', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, f"{get_label('pca_2d', LANG)}\n{get_label('cumulative_variance_prefix', LANG)} {sum(pca.explained_variance_ratio_)*100:.1f}%", fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'pca_2d_projection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_pca_2d'))


    # 22. パーセンタイルバンドプロット（主要指標）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # TITLE_BOTTOM:{get_label('percentile_bands', LANG), fontsize=18, fontweight='bold'}

    key_metrics = ['ssim', 'psnr', 'sharpness', 'noise', 'edge_density', 'contrast']
    metric_names = ['SSIM', get_label('psnr_db', LANG), get_label('sharpness_clarity', LANG), 'Noise', get_label('edge_density', LANG), get_label('contrast', LANG)]

    for idx, (metric, name) in enumerate(zip(key_metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]

        models = df['model'].unique()
        positions = range(len(models))

        for i, model in enumerate(models):
            model_data = df[df['model'] == model][metric]
            q25 = model_data.quantile(0.25)
            q50 = model_data.quantile(0.50)
            q75 = model_data.quantile(0.75)

            ax.plot([i, i], [q25, q75], 'b-', linewidth=8, alpha=0.3)
            ax.plot(i, q50, 'ro', markersize=10)

        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(name, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'{name} Distribution', fontsize=12)

    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, f'Principal Component Analysis (PCA): 16 Metrics to 2D\nCumulative variance: {sum(pca.explained_variance_ratio_)*100:.1f}%', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'percentile_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_percentile'))


    # 23. 寄与率グラフ（PCA）
    pca_full = PCA()
    pca_full.fit(X_scaled)

    plt.figure(figsize=(12, 6))
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum)+1), cumsum, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label=get_label('line_95', LANG))
    plt.xlabel(get_label('num_components', LANG), fontsize=14, fontweight='bold')
    plt.ylabel(get_label('cumulative_variance', LANG), fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'PCACumulative Variance Ratio：何次元で95%説明できるか', fontsize=16, fontweight='bold'}
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, get_label('pca_variance', LANG), fontsize=18, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'pca_cumulative_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.plot_pca_variance'))


    # ========================================
    # LFV法則の証明用プロット
    # ========================================

    # 24. テクスチャ依存の証明: Texture Complexity vs Local Quality Min
    print(f"\n{'='*80}")
    print(i18n.t('stats_analysis.lfv_proof_header'))
    print(f"{'='*80}")

    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['texture_complexity'], model_data['local_quality_min'],
                   label=model, alpha=0.6, s=50)

    # 全体の相関係数を計算
    correlation = df['texture_complexity'].corr(df['local_quality_min'])

    # 回帰線
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['texture_complexity'], df['local_quality_min']
    )
    x_line = np.linspace(df['texture_complexity'].min(), df['texture_complexity'].max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
             label=f'Regression (r={correlation:.3f}, p={p_value:.2e})')

    # 相関の強さを注釈
    if correlation < -0.7:
        corr_text = f'Strong Negative Correlation\nr = {correlation:.3f} (p < 0.001)\nLFV is Texture-Dependent'
        text_color = '#ff4444'
    elif correlation < -0.5:
        corr_text = f'Moderate Negative Correlation\nr = {correlation:.3f}\nLFV shows Texture-Dependency'
        text_color = '#ffa500'
    else:
        corr_text = f'Weak Correlation\nr = {correlation:.3f}\nTexture-Dependency unclear'
        text_color = '#888888'

    plt.text(0.05, 0.95, corr_text,
             transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=text_color, linewidth=2))

    plt.xlabel('Texture Complexity', fontsize=14, fontweight='bold')
    plt.ylabel('Local Quality Min (Worst Patch SSIM)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Place title at bottom for academic papers
    fig = plt.gcf()
    title_text = 'LFV Proof 1: Texture-Dependency\nTexture Complexity vs Local Quality Min'
    fig.text(0.5, -0.05, title_text, fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'lfv_proof_texture_dependency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.lfv_proof_1').format(correlation=correlation))


    # 25. 空間依存の証明: Local Quality Min の分布（境界への偏り）
    plt.figure(figsize=(14, 6))

    # 左: ヒストグラム
    plt.subplot(1, 2, 1)
    counts, bins, patches = plt.hist(df['local_quality_min'], bins=30,
                                     color='#4A90E2', alpha=0.7, edgecolor='black')

    # 閾値ラインを追加
    threshold = df['local_quality_min'].quantile(0.25)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'25th Percentile ({threshold:.3f})')

    # LFV領域を強調
    lfv_count = len(df[df['local_quality_min'] < threshold])
    plt.text(0.05, 0.95, f'LFV Cases (Min SSIM < {threshold:.2f}):\n{lfv_count} samples ({lfv_count/len(df)*100:.1f}%)',
             transform=plt.gca().transAxes,
             fontsize=11, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

    plt.xlabel('Local Quality Min (Worst Patch SSIM)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Number of Images)', fontsize=12, fontweight='bold')
    plt.title('Distribution of Local Quality Min', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    # 右: Boxplot（モデル別）
    plt.subplot(1, 2, 2)
    model_names = df['model'].unique()
    data_by_model = [df[df['model'] == model]['local_quality_min'].values for model in model_names]

    bp = plt.boxplot(data_by_model, labels=model_names, patch_artist=True,
                     showmeans=True, meanline=True)

    # 色付け
    colors = ['#4A90E2', '#ffa500', '#2d7d46', '#9b59b6', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label=f'LFV Threshold ({threshold:.3f})')

    plt.ylabel('Local Quality Min', fontsize=12, fontweight='bold')
    plt.xlabel('AI Model', fontsize=12, fontweight='bold')
    plt.title('Model Comparison: Spatial Quality Variance', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Place title at bottom for academic papers
    fig = plt.gcf()
    title_text = 'LFV Proof 2: Spatial-Dependency\nLocal Quality Min Distribution (Boundary Bias Detection)'
    fig.text(0.5, -0.05, title_text, fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'lfv_proof_spatial_dependency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(i18n.t('stats_analysis.lfv_proof_2'))


    # ===== 新規: BBI（Boundary Bias Index）計算と統計的検定 =====
    print(i18n.t('stats_analysis.bbi_calc_start'))

    # CSVファイルからdetailedディレクトリのパスを推定
    csv_path = Path(csv_file)
    # results/batch_analysis.csv → results/detailed_YYYYMMDD_HHMMSS/
    results_dir = csv_path.parent
    detailed_dirs = list(results_dir.glob('detailed_*'))

    if not detailed_dirs:
        print(i18n.t('stats_analysis.bbi_warning_no_dir'))
        bbi_mean = None
        boundary_p_value = None
        spatial_strength = 'N/A'
    else:
        # 最新のdetailedディレクトリを使用
        detailed_dir = sorted(detailed_dirs)[-1]
        print(i18n.t('stats_analysis.bbi_detailed_dir').format(path=detailed_dir))

        # LFVケースのみを抽出（25th percentile以下）
        lfv_df = df[df['local_quality_min'] < threshold].copy()

        # 各画像のMin SSIMパッチ座標を収集
        spatial_data = []

        for idx, row in lfv_df.iterrows():
            image_id = row['image_id']
            model = row['model']
            min_ssim_value = row['local_quality_min']

            # p6_local_quality_data.csvのパスを構築
            p6_path = detailed_dir / model / image_id / 'p6_local_quality_data.csv'

            if not p6_path.exists():
                continue

            try:
                # p6データ読み込み
                p6_df = pd.read_csv(p6_path)

                # Min SSIMパッチを特定（最小値）
                min_idx = p6_df['local_ssim'].idxmin()
                min_row = p6_df.loc[min_idx, 'row']
                min_col = p6_df.loc[min_idx, 'col']

                # 画像のパッチサイズを取得（最大row/col + 1）
                total_rows = p6_df['row'].max() + 1
                total_cols = p6_df['col'].max() + 1

                # 境界からの正規化距離を計算
                dist_top = min_row
                dist_bottom = total_rows - 1 - min_row
                dist_left = min_col
                dist_right = total_cols - 1 - min_col

                min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
                max_possible_dist = min(total_rows, total_cols) / 2
                normalized_dist = min_dist / max_possible_dist if max_possible_dist > 0 else 0

                # 境界判定（外周25%エリア）
                boundary_threshold = 0.25
                is_boundary = (min_row < total_rows * boundary_threshold or
                             min_row > total_rows * (1 - boundary_threshold) or
                             min_col < total_cols * boundary_threshold or
                             min_col > total_cols * (1 - boundary_threshold))

                spatial_data.append({
                    'image_id': image_id,
                    'model': model,
                    'min_ssim_value': min_ssim_value,
                    'min_row': min_row,
                    'min_col': min_col,
                    'total_rows': total_rows,
                    'total_cols': total_cols,
                    'distance_to_boundary': min_dist,
                    'normalized_distance': normalized_dist,
                    'is_boundary': is_boundary
                })

            except Exception as e:
                print(i18n.t('stats_analysis.bbi_warning_error').format(id=image_id, error=str(e)))
                continue

        if len(spatial_data) == 0:
            print(i18n.t('stats_analysis.bbi_warning_no_data'))
            bbi_mean = None
            boundary_p_value = None
            spatial_strength = 'N/A'
        else:
            spatial_df = pd.DataFrame(spatial_data)

            # BBI計算: 1 - 平均正規化距離
            # BBI = 1.0 → 完全に境界, BBI = 0.0 → 完全に中央
            bbi_mean = 1.0 - spatial_df['normalized_distance'].mean()

            # カイ二乗検定: 観測分布 vs ランダム分布（一様分布）
            from scipy.stats import chisquare

            boundary_count = spatial_df['is_boundary'].sum()
            center_count = len(spatial_df) - boundary_count

            # ランダムなら面積比で分布（外周25% → 約44%の面積）
            # (1 - (1-0.25*2)^2) = 1 - 0.5^2 = 0.75 → 境界エリア
            # より保守的に、外周25%を単純計算: 0.25 * 4 = 1.0 → 約43.75%
            expected_boundary_ratio = 1 - (0.5 ** 2)  # 0.75
            expected_boundary = len(spatial_df) * expected_boundary_ratio
            expected_center = len(spatial_df) * (1 - expected_boundary_ratio)

            observed = [boundary_count, center_count]
            expected = [expected_boundary, expected_center]

            chi2, boundary_p_value = chisquare(observed, expected)

            # 強度判定
            if bbi_mean > 0.7 and boundary_p_value < 0.001:
                spatial_strength = 'Strong'
            elif bbi_mean > 0.5 and boundary_p_value < 0.05:
                spatial_strength = 'Moderate'
            else:
                spatial_strength = 'Weak'

            # 空間データCSV出力
            spatial_df.to_csv(output_dir / 'lfv_spatial_analysis.csv', index=False)
            print(i18n.t('stats_analysis.bbi_spatial_data').format(count=len(spatial_df)))

            print(i18n.t('stats_analysis.bbi_stats_index').format(value=bbi_mean))
            print(i18n.t('stats_analysis.bbi_stats_pvalue').format(value=boundary_p_value))
            print(i18n.t('stats_analysis.bbi_stats_cases').format(boundary=boundary_count, total=len(spatial_df), percent=boundary_count/len(spatial_df)*100))
            print(i18n.t('stats_analysis.bbi_stats_dependency').format(strength=spatial_strength))

            # 26. Plot: LFV Min SSIMパッチ座標の散布図（境界偏り可視化）
            plt.figure(figsize=(12, 10))

            # 全画像で同じパッチサイズと仮定（最頻値を使用）
            common_rows = spatial_df['total_rows'].mode()[0]
            common_cols = spatial_df['total_cols'].mode()[0]

            # 境界エリアを矩形で表示
            boundary_width = common_cols * boundary_threshold
            boundary_height = common_rows * boundary_threshold

            # 背景に境界エリアを描画
            from matplotlib.patches import Rectangle
            ax = plt.gca()

            # 4つの境界エリア（薄い赤）
            boundary_rects = [
                Rectangle((0, 0), common_cols, boundary_height,
                         facecolor='red', alpha=0.1, label='Boundary Zone (25%)'),
                Rectangle((0, common_rows - boundary_height), common_cols, boundary_height,
                         facecolor='red', alpha=0.1),
                Rectangle((0, 0), boundary_width, common_rows,
                         facecolor='red', alpha=0.1),
                Rectangle((common_cols - boundary_width, 0), boundary_width, common_rows,
                         facecolor='red', alpha=0.1)
            ]
            for rect in boundary_rects[:1]:  # 最初の1つだけラベル付き
                ax.add_patch(rect)
            for rect in boundary_rects[1:]:
                ax.add_patch(rect)

            # 中央エリア（薄い緑）
            center_rect = Rectangle(
                (boundary_width, boundary_height),
                common_cols - 2 * boundary_width,
                common_rows - 2 * boundary_height,
                facecolor='green', alpha=0.05, label='Center Zone (75%)'
            )
            ax.add_patch(center_rect)

            # LFVパッチ座標をプロット
            boundary_df = spatial_df[spatial_df['is_boundary'] == True]
            center_df = spatial_df[spatial_df['is_boundary'] == False]

            plt.scatter(boundary_df['min_col'], boundary_df['min_row'],
                       c='red', s=80, alpha=0.6, edgecolors='darkred', linewidth=1,
                       label=f'Boundary LFV ({len(boundary_df)})')
            plt.scatter(center_df['min_col'], center_df['min_row'],
                       c='blue', s=80, alpha=0.6, edgecolors='darkblue', linewidth=1,
                       label=f'Center LFV ({len(center_df)})')

            # 統計情報を注釈
            stats_text = f'BBI = {bbi_mean:.3f}\np < {boundary_p_value:.6f}\n' + \
                        f'Boundary: {boundary_count}/{len(spatial_df)} ({boundary_count/len(spatial_df)*100:.1f}%)\n' + \
                        f'Spatial Dependency: {spatial_strength}'
            plt.text(0.98, 0.98, stats_text,
                    transform=ax.transAxes,
                    fontsize=11, fontweight='bold',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

            plt.xlabel('Column (Horizontal Position)', fontsize=12, fontweight='bold')
            plt.ylabel('Row (Vertical Position)', fontsize=12, fontweight='bold')
            plt.xlim(-1, common_cols)
            plt.ylim(-1, common_rows)
            plt.gca().invert_yaxis()  # 画像座標系に合わせる
            plt.legend(fontsize=10, loc='lower left')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # タイトルを下に配置
            fig = plt.gcf()
            title_text = f'LFV Proof 3: Spatial Coordinate Distribution\nMin SSIM Patch Locations (Boundary Bias Index = {bbi_mean:.3f}, p < {boundary_p_value:.6f})'
            fig.text(0.5, -0.05, title_text, fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
            plt.savefig(output_dir / 'lfv_proof_coordinate_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(i18n.t('stats_analysis.lfv_proof_3'))


    # LFV証明サマリーをCSVに出力（BBIと空間依存性を追加）
    lfv_summary = {
        'correlation_texture_localmin': [correlation],
        'correlation_p_value': [p_value],
        'lfv_threshold_25th': [threshold],
        'lfv_cases_count': [lfv_count],
        'lfv_cases_percentage': [lfv_count/len(df)*100],
        'texture_dependency_strength': ['Strong' if correlation < -0.7 else 'Moderate' if correlation < -0.5 else 'Weak'],
        'boundary_bias_index': [bbi_mean if bbi_mean is not None else 'N/A'],
        'boundary_p_value': [boundary_p_value if boundary_p_value is not None else 'N/A'],
        'spatial_dependency_strength': [spatial_strength]
    }
    pd.DataFrame(lfv_summary).to_csv(output_dir / 'lfv_proof_summary.csv', index=False)
    print(i18n.t('stats_analysis.lfv_summary'))

    print(f"\n{'='*80}")
    print(i18n.t('stats_analysis.lfv_complete'))
    print(i18n.t('stats_analysis.lfv_complete_detail'))
    print(i18n.t('stats_analysis.lfv_complete_plot24'))
    print(i18n.t('stats_analysis.lfv_complete_plot25'))
    print(i18n.t('stats_analysis.lfv_complete_plot26') + '\n')

    # 出力ディレクトリのパスを返す
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistical analysis of batch image processing results'
    )
    parser.add_argument(
        'csv_file',
        help='Path to batch_analysis.csv file'
    )
    parser.add_argument(
        '--lang',
        choices=['ja', 'en'],
        default='en',
        help='Output language: ja (Japanese) or en (English, default)'
    )

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"[ERROR] CSV file not found: {args.csv_file}")
        sys.exit(1)

    analyze_batch_results(args.csv_file, lang=args.lang)
