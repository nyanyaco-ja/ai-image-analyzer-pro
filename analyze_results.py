"""
統計分析スクリプト：バッチ処理結果から閾値を決定（26パターン対応版）

26パターンのハルシネーション検出:
- 9つの組み合わせパターン（複合異常）
- 17の単独閾値判定（各指標の異常値）

使い方:
python analyze_results.py results/batch_analysis.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Font settings for English (academic papers)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def analyze_batch_results(csv_file):
    """
    バッチ処理結果の統計分析
    """

    # CSV読み込み
    df = pd.read_csv(csv_file)

    print(f"\n{'='*80}")
    print(f"[STATS] 統計分析レポート")
    print(f"{'='*80}")
    print(f"[FILE] データファイル: {csv_file}")
    print(f"[IMAGE] 画像数: {df['image_id'].nunique()}")
    print(f"[MODEL] モデル数: {df['model'].nunique()}")
    print(f"[STATS] 総データ数: {len(df)}")
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
    generate_research_plots(df, output_dir)

    print(f"\n[OK] 分析完了！")
    print(f"[FOLDER] 結果保存先: {output_dir}/")

    return str(output_dir)


def print_basic_statistics(df):
    """
    基本統計量の表示
    """

    print(f"\n[STATS] 主要指標の基本統計量:")
    print(f"{'='*80}")

    # 17項目すべて
    metrics = ['ssim', 'ms_ssim', 'psnr', 'lpips', 'sharpness', 'contrast',
               'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
               'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
               'histogram_corr', 'lab_L_mean', 'total_score']

    stats = df[metrics].describe().T
    stats.columns = ['件数', '平均', '標準偏差', '最小', '25%', '50%', '75%', '最大']

    print(stats.round(4).to_string())
    print(f"{'='*80}\n")


def compare_models(df, output_dir):
    """
    モデル別比較
    """

    print(f"\n[RANK] モデル別ランキング:")
    print(f"{'='*80}")

    # 主要指標でグループ化
    model_comparison = df.groupby('model').agg({
        'ssim': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'lpips': ['mean', 'std'],
        'total_score': ['mean', 'std'],
        'noise': ['mean', 'std'],
        'artifact_total': ['mean', 'std'],
        'sharpness': ['mean', 'std'],
        'edge_density': ['mean', 'std']
    }).round(4)

    # Total Scoreでソート
    model_comparison = model_comparison.sort_values(('total_score', 'mean'), ascending=False)

    print(model_comparison.to_string())
    print(f"{'='*80}\n")

    # CSV保存
    model_comparison.to_csv(output_dir / 'model_comparison.csv', encoding='utf-8-sig')

    # Visualization: Model Total Score
    plt.figure(figsize=(12, 6))
    model_scores = df.groupby('model')['total_score'].mean().sort_values(ascending=False)

    plt.bar(range(len(model_scores)), model_scores.values)
    plt.xticks(range(len(model_scores)), model_scores.index, rotation=45, ha='right')
    plt.ylabel('Total Score (Average)')
    # TITLE_BOTTOM:{'Model Comparison by Total Score'}
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Model Comparison by Total Score', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'model_scores.png', dpi=150)
    plt.close()

    print(f"[STATS] グラフ保存: {output_dir}/model_scores.png")


def analyze_correlations(df, output_dir):
    """
    17項目間の相関分析
    """

    print(f"\n[CORR] 相関分析:")
    print(f"{'='*80}")

    # 数値列のみ抽出
    numeric_cols = ['ssim', 'psnr', 'lpips', 'ms_ssim', 'sharpness', 'contrast',
                    'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
                    'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
                    'histogram_corr', 'lab_L_mean', 'total_score']

    corr_matrix = df[numeric_cols].corr()

    # 相関行列をヒートマップ表示
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    # TITLE_BOTTOM:{'Correlation Matrix of 17 Metrics', fontsize=16, fontweight='bold'}
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150)
    plt.close()

    print(f"[STATS] 相関マトリックス保存: {output_dir}/correlation_matrix.png")

    # 高相関ペアを表示
    print(f"\n[HIGH] 高相関ペア（|r| > 0.7）:")
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
        print("   (なし)")

    print(f"{'='*80}\n")


def suggest_thresholds(df, output_dir):
    """
    Suggest evidence-based thresholds
    """

    print(f"\n[TIP] Recommended Threshold Suggestions:")
    print(f"{'='*80}")

    thresholds = {}

    # Determine thresholds from statistical values for each metric
    # Propose thresholds for all 17 metrics
    metrics_config = {
        'ssim': {'direction': 'high', 'percentile': 25, 'name': 'SSIM (Structural Similarity)'},
        'ms_ssim': {'direction': 'high', 'percentile': 25, 'name': 'MS-SSIM (Multi-Scale SSIM)'},
        'psnr': {'direction': 'high', 'percentile': 25, 'name': 'PSNR (Peak Signal-to-Noise Ratio)'},
        'lpips': {'direction': 'low', 'percentile': 75, 'name': 'LPIPS (Perceptual Similarity)'},
        'sharpness': {'direction': 'high', 'percentile': 25, 'name': 'Sharpness'},
        'contrast': {'direction': 'high', 'percentile': 25, 'name': 'Contrast'},
        'entropy': {'direction': 'high', 'percentile': 25, 'name': 'Entropy (Information Content)'},
        'noise': {'direction': 'low', 'percentile': 75, 'name': 'Noise Level'},
        'edge_density': {'direction': 'high', 'percentile': 25, 'name': 'Edge Density'},
        'artifact_total': {'direction': 'low', 'percentile': 75, 'name': 'Artifacts'},
        'delta_e': {'direction': 'low', 'percentile': 75, 'name': 'Color Difference (ΔE)'},
        'high_freq_ratio': {'direction': 'high', 'percentile': 25, 'name': 'High Frequency Ratio'},
        'texture_complexity': {'direction': 'high', 'percentile': 25, 'name': 'Texture Complexity'},
        'local_quality_mean': {'direction': 'high', 'percentile': 25, 'name': 'Local Quality Mean'},
        'histogram_corr': {'direction': 'high', 'percentile': 25, 'name': 'Histogram Correlation'},
        'lab_L_mean': {'direction': 'neutral', 'percentile': 50, 'name': 'LAB Lightness (Reference)'},
        'total_score': {'direction': 'high', 'percentile': 25, 'name': 'Total Score'},
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

    print(f"\n[DETECT] ハルシネーション検出ロジックの提案（26パターン）:")
    print(f"{'='*80}")

    # 検出カウント用
    detection_count = pd.Series(0, index=df.index)
    detected_patterns = {idx: [] for idx in df.index}
    pattern_stats = {}

    # ========== 組み合わせパターン（9つ） ==========
    print(f"\n【組み合わせパターン（複合異常検出）】")
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

    print(f"P1: SSIM高 × PSNR低（構造類似だがピクセル値相違）")
    print(f"    固定閾値 (SSIM>0.97 & PSNR<25): {len(pattern1a)}件")
    print(f"    動的閾値 (SSIM≥{ssim_high:.4f} & PSNR≤{psnr_low:.2f}): {len(pattern1b)}件")
    print(f"    統合後: {len(pattern1)}件 ({len(pattern1)/len(df)*100:.1f}%)")
    print(f"    リスク: 中～高（AIが構造を模倣した可能性）")

    # === パターン2: Sharpness高 × Noise高 ===
    sharp_high = df['sharpness'].quantile(0.75)
    noise_high = df['noise'].quantile(0.75)
    pattern2 = df[(df['sharpness'] > sharp_high) & (df['noise'] > noise_high)]
    detection_count[pattern2.index] += 1
    for idx in pattern2.index:
        detected_patterns[idx].append('P2:シャープ高×Noise高')
    pattern_stats['P2'] = {'count': len(pattern2), 'rate': len(pattern2)/len(df)*100}

    print(f"\nP2: Sharpness高 × Noise高（過剰処理）")
    print(f"    条件: シャープ>{sharp_high:.2f} & Noise>{noise_high:.2f}")
    print(f"    該当: {len(pattern2)}件 ({len(pattern2)/len(df)*100:.1f}%)")
    print(f"    リスク: 中（過度なシャープ化によるNoise増幅）")

    # === パターン3: Edge Density高 × 局所品質低 ===
    edge_90 = df['edge_density'].quantile(0.90)
    quality_25 = df['local_quality_mean'].quantile(0.25)
    pattern3 = df[(df['edge_density'] > edge_90) & (df['local_quality_mean'] < quality_25)]
    detection_count[pattern3.index] += 1
    for idx in pattern3.index:
        detected_patterns[idx].append('P3:エッジ高×品質低')
    pattern_stats['P3'] = {'count': len(pattern3), 'rate': len(pattern3)/len(df)*100}

    print(f"\nP3: Edge Density高 × 局所品質低（不自然なエッジ）")
    print(f"    条件: エッジ>{edge_90:.2f} & 局所品質<{quality_25:.4f}")
    print(f"    該当: {len(pattern3)}件 ({len(pattern3)/len(df)*100:.1f}%)")
    print(f"    リスク: 中～高（エッジ追加が不均一）")

    # === パターン4: Artifacts異常高 ===
    artifact_90 = df['artifact_total'].quantile(0.90)
    pattern4 = df[df['artifact_total'] > artifact_90]
    detection_count[pattern4.index] += 1
    for idx in pattern4.index:
        detected_patterns[idx].append('P4:Artifacts高')
    pattern_stats['P4'] = {'count': len(pattern4), 'rate': len(pattern4)/len(df)*100}

    print(f"\nP4: Artifacts異常高（GAN特有の歪み）")
    print(f"    条件: Artifacts>{artifact_90:.2f}")
    print(f"    該当: {len(pattern4)}件 ({len(pattern4)/len(df)*100:.1f}%)")
    print(f"    リスク: 高（リンギング・ブロックNoise）")

    # === パターン5: LPIPS高 × SSIM高 ===
    lpips_75 = df['lpips'].quantile(0.75)
    ssim_75 = df['ssim'].quantile(0.75)
    pattern5 = df[(df['lpips'] > lpips_75) & (df['ssim'] > ssim_75)]
    detection_count[pattern5.index] += 1
    for idx in pattern5.index:
        detected_patterns[idx].append('P5:LPIPS高×SSIM高')
    pattern_stats['P5'] = {'count': len(pattern5), 'rate': len(pattern5)/len(df)*100}

    print(f"\nP5: LPIPS高 × SSIM高（知覚と構造の矛盾）")
    print(f"    条件: LPIPS>{lpips_75:.4f} & SSIM>{ssim_75:.4f}")
    print(f"    該当: {len(pattern5)}件 ({len(pattern5)/len(df)*100:.1f}%)")
    print(f"    リスク: 中（構造は似ているが知覚的に異なる）")

    # === パターン6: 局所品質ばらつき大 ===
    if 'local_quality_std' in df.columns:
        quality_std_75 = df['local_quality_std'].quantile(0.75)
        pattern6 = df[df['local_quality_std'] > quality_std_75]
        detection_count[pattern6.index] += 1
        for idx in pattern6.index:
            detected_patterns[idx].append('P6:品質ばらつき大')
        pattern_stats['P6'] = {'count': len(pattern6), 'rate': len(pattern6)/len(df)*100}

        print(f"\nP6: 局所品質ばらつき大（不均一な処理）")
        print(f"    条件: 局所SSIM標準偏差>{quality_std_75:.4f}")
        print(f"    該当: {len(pattern6)}件 ({len(pattern6)/len(df)*100:.1f}%)")
        print(f"    リスク: 中～高（領域によって品質が異なる）")
    else:
        pattern_stats['P6'] = {'count': 0, 'rate': 0}
        print(f"\nP6: 局所品質ばらつき大 → データなし（スキップ）")

    # === パターン7: Entropy低 × High-Freq高 ===
    entropy_25 = df['entropy'].quantile(0.25)
    highfreq_75 = df['high_freq_ratio'].quantile(0.75)
    pattern7 = df[(df['entropy'] < entropy_25) & (df['high_freq_ratio'] > highfreq_75)]
    detection_count[pattern7.index] += 1
    for idx in pattern7.index:
        detected_patterns[idx].append('P7:Entropy低×高周波高')
    pattern_stats['P7'] = {'count': len(pattern7), 'rate': len(pattern7)/len(df)*100}

    print(f"\nP7: Entropy低 × 高周波高（反復パターン）")
    print(f"    条件: Entropy<{entropy_25:.3f} & 高周波>{highfreq_75:.4f}")
    print(f"    該当: {len(pattern7)}件 ({len(pattern7)/len(df)*100:.1f}%)")
    print(f"    リスク: 中（人工的な反復パターン）")

    # === パターン8: Contrast異常 × Histogram相関低 ===
    contrast_90 = df['contrast'].quantile(0.90)
    histcorr_25 = df['histogram_corr'].quantile(0.25)
    pattern8 = df[(df['contrast'] > contrast_90) & (df['histogram_corr'] < histcorr_25)]
    detection_count[pattern8.index] += 1
    for idx in pattern8.index:
        detected_patterns[idx].append('P8:Contrast異常×Hist相関低')
    pattern_stats['P8'] = {'count': len(pattern8), 'rate': len(pattern8)/len(df)*100}

    print(f"\nP8: Contrast異常 × Histogram相関低（濃度分布崩壊）")
    print(f"    条件: Contrast>{contrast_90:.2f} & Hist相関<{histcorr_25:.4f}")
    print(f"    該当: {len(pattern8)}件 ({len(pattern8)/len(df)*100:.1f}%)")
    print(f"    リスク: 中（Contrast強調で濃度分布が崩れている）")

    # === パターン9: MS-SSIM低 × Total Score低 ===
    msssim_25 = df['ms_ssim'].quantile(0.25)
    total_25 = df['total_score'].quantile(0.25)
    pattern9 = df[(df['ms_ssim'] < msssim_25) & (df['total_score'] < total_25)]
    detection_count[pattern9.index] += 1
    for idx in pattern9.index:
        detected_patterns[idx].append('P9:MS-SSIM低×総合低')
    pattern_stats['P9'] = {'count': len(pattern9), 'rate': len(pattern9)/len(df)*100}

    print(f"\nP9: MS-SSIM低 × Total Score低（総合的低品質）")
    print(f"    条件: MS-SSIM<{msssim_25:.4f} & 総合<{total_25:.2f}")
    print(f"    該当: {len(pattern9)}件 ({len(pattern9)/len(df)*100:.1f}%)")
    print(f"    リスク: 高（複数スケールで品質劣化）")

    # ========== 単独パターン（17項目） ==========
    print(f"\n{'='*80}")
    print(f"【単独閾値判定パターン（17項目）】")
    print(f"{'='*80}")

    single_pattern_count = 0

    # 高い方が良い指標（異常に低い = 下位10%）
    high_is_good = [
        ('ssim', 'SSIM低'), ('ms_ssim', 'MS-SSIM低'), ('psnr', 'PSNR低'),
        ('sharpness', 'Sharpness低'), ('contrast', 'Contrast低'), ('entropy', 'Entropy低'),
        ('edge_density', 'EdgeDensity低'), ('high_freq_ratio', 'HighFreq低'),
        ('texture_complexity', 'Texture低'), ('local_quality_mean', 'LocalQuality低'),
        ('histogram_corr', 'HistCorr低'), ('total_score', 'TotalScore低')
    ]

    print(f"\n高い方が良い指標（下位10%を異常検出）:")
    for col, name in high_is_good:
        if col in df.columns:
            threshold = df[col].quantile(0.10)
            detected = df[df[col] < threshold]
            detection_count[detected.index] += 1
            for idx in detected.index:
                detected_patterns[idx].append(f'単独:{name}')
            single_pattern_count += len(detected)
            print(f"  {name:20s}: <{threshold:8.4f} → {len(detected):4d}件 ({len(detected)/len(df)*100:5.1f}%)")

    # 低い方が良い指標（異常に高い = 上位10%）
    low_is_good = [
        ('lpips', 'LPIPS高'), ('noise', 'Noise高'),
        ('artifact_total', 'Artifacts高'), ('delta_e', 'DeltaE高')
    ]

    print(f"\n低い方が良い指標（上位10%を異常検出）:")
    for col, name in low_is_good:
        if col in df.columns:
            threshold = df[col].quantile(0.90)
            detected = df[df[col] > threshold]
            detection_count[detected.index] += 1
            for idx in detected.index:
                detected_patterns[idx].append(f'単独:{name}')
            single_pattern_count += len(detected)
            print(f"  {name:20s}: >{threshold:8.4f} → {len(detected):4d}件 ({len(detected)/len(df)*100:5.1f}%)")

    if 'local_quality_std' in df.columns:
        threshold = df['local_quality_std'].quantile(0.90)
        detected = df[df['local_quality_std'] > threshold]
        detection_count[detected.index] += 1
        for idx in detected.index:
            detected_patterns[idx].append(f'単独:LocalQualityStd高')
        single_pattern_count += len(detected)
        print(f"  {'LocalQualityStd高':20s}: >{threshold:8.4f} → {len(detected):4d}件 ({len(detected)/len(df)*100:5.1f}%)")

    print(f"\n単独パターン合計検出: {single_pattern_count}件（延べ数）")

    # ========== 総合リスクスコア計算 ==========
    print(f"\n{'='*80}")
    print(f"[STATS] 総合ハルシネーションリスクスコア（26パターン統合）")
    print(f"{'='*80}")

    # 信頼度分類（多数決）
    high_confidence = df[detection_count >= 5]  # 5パターン以上
    medium_confidence = df[(detection_count >= 3) & (detection_count < 5)]  # 3-4パターン
    low_confidence = df[(detection_count >= 1) & (detection_count < 3)]  # 1-2パターン
    no_detection = df[detection_count == 0]  # 検出なし（正常）

    print(f"\n信頼度別分類:")
    print(f"  高信頼度(5+パターン): {len(high_confidence):5d}件 ({len(high_confidence)/len(df)*100:5.1f}%)")
    print(f"  中信頼度(3-4パターン): {len(medium_confidence):5d}件 ({len(medium_confidence)/len(df)*100:5.1f}%)")
    print(f"  低信頼度(1-2パターン): {len(low_confidence):5d}件 ({len(low_confidence)/len(df)*100:5.1f}%)")
    print(f"  正常(検出0):          {len(no_detection):5d}件 ({len(no_detection)/len(df)*100:5.1f}%)")

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
    print(f"\n[SAVE] 26パターン検出結果保存: {output_csv}")

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
    print(f"[SAVE] パターン別サマリー保存: {summary_path}")

    print(f"{'='*80}\n")


def generate_research_plots(df, output_dir):
    """
    研究用プロット画像を生成（論文・発表用）
    """

    print(f"\n[STATS] 研究用プロット生成中:")
    print(f"{'='*80}")

    # 1. Sharpness vs PSNR 散布図（AI Modelの戦略を示す）
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['psnr'], model_data['sharpness'],
                   label=model, alpha=0.6, s=50)

    plt.xlabel('PSNR (Fidelity) [dB]', fontsize=14, fontweight='bold')
    plt.ylabel('Sharpness (Clarity)', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'AI Model Strategy Map: Fidelity vs Clarity', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 戦略領域の注釈
    plt.axhline(y=df['sharpness'].median(), color='red', linestyle='--', alpha=0.3, label='Median')
    plt.axvline(x=df['psnr'].median(), color='red', linestyle='--', alpha=0.3)

    # 領域ラベル
    max_psnr = df['psnr'].max()
    max_sharp = df['sharpness'].max()
    plt.text(max_psnr * 0.95, max_sharp * 0.95, 'Ideal Region\n(High Fidelity & Clarity)',
             fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.text(df['psnr'].min() * 1.05, max_sharp * 0.95, 'Over-processing\n(Low Fidelity)\nHallucination Risk',
             fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plot1_path = output_dir / 'strategy_map_sharpness_vs_psnr.png'
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'AI Model Strategy Map: Fidelity vs Clarity', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Sharpness vs PSNR 散布図: {plot1_path}")


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

    plt.ylabel('LPIPS (Perceptual Similarity)', fontsize=14, fontweight='bold')
    plt.xlabel('AI Model', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'LPIPS Distribution by Model (Stability)\nSmaller box = More stable', fontsize=16, fontweight='bold'}
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plot2_path = output_dir / 'stability_lpips_boxplot.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] LPIPS 箱ひげ図: {plot2_path}")


    # 3. SSIM vs PSNR 散布図（相関確認・異常検出）
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ssim'], model_data['psnr'],
                   label=model, alpha=0.6, s=50)

    plt.xlabel('SSIM (Structural Similarity)', fontsize=14, fontweight='bold')
    plt.ylabel('PSNR (Signal-to-Noise Ratio) [dB]', fontsize=14, fontweight='bold')
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
    print(f"[OK] SSIM vs PSNR 散布図: {plot3_path}")


    # 4. Noise vs Artifacts 散布図
    plt.figure(figsize=(12, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['noise'], model_data['artifact_total'],
                   label=model, alpha=0.6, s=50)

    plt.xlabel('Noise Level', fontsize=14, fontweight='bold')
    plt.ylabel('Artifacts (Blocking + Ringing)', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Noise vs Artifacts\nLower-left is ideal (both low)', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Ideal Regionの表示
    low_noise = df['noise'].quantile(0.25)
    low_artifact = df['artifact_total'].quantile(0.25)
    plt.axvline(x=low_noise, color='green', linestyle='--', alpha=0.3)
    plt.axhline(y=low_artifact, color='green', linestyle='--', alpha=0.3)
    plt.fill_between([0, low_noise], 0, low_artifact, alpha=0.1, color='green', label='Ideal Region')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plot4_path = output_dir / 'quality_noise_vs_artifact.png'
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Noise vs Artifacts\nLower-left is ideal (both low)', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Noise vs Artifacts: {plot4_path}")


    # 5. モデル別レーダーチャート（主要6指標）
    fig = plt.figure(figsize=(14, 10))

    categories = ['SSIM', 'PSNR', 'Sharpness', 'Edge Density', 'Noise\n(Inverted)', 'Artifacts\n(Inverted)']
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
    print(f"[OK] モデル別レーダーチャート: {plot5_path}")


    # 6. 17項目のバイオリンプロット（分布の可視化）
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    # TITLE_BOTTOM:{'Distribution of 17 Metrics (Violin Plot)', fontsize=20, fontweight='bold'}

    metrics_for_violin = [
        'ssim', 'ms_ssim', 'psnr', 'lpips', 'sharpness', 'contrast',
        'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
        'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
        'histogram_corr', 'lab_L_mean', 'total_score'
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
    print(f"[OK] 17項目バイオリンプロット: {plot6_path}")

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
                   label=f'Hallucination Suspected ({len(hallucination_candidates)} cases)', zorder=10)

    plt.axhline(y=psnr_low, color='orange', linestyle='--', alpha=0.5, label=f'PSNR Threshold ({psnr_low:.1f})')
    plt.axvline(x=ssim_high, color='orange', linestyle='--', alpha=0.5, label=f'SSIM Threshold ({ssim_high:.3f})')

    plt.xlabel('SSIM (Structural Similarity)', fontsize=14, fontweight='bold')
    plt.ylabel('PSNR [dB]', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Hallucination Detection: High SSIM & Low PSNR\nLower-right = Mimicked structure, low fidelity', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_ssim_high_psnr_low.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ハルシネーション検出①（SSIM×PSNR）")


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
                   label=f'Over-processing Suspected ({len(over_processed)} cases)', zorder=10)

    plt.axhline(y=noise_high, color='orange', linestyle='--', alpha=0.5)
    plt.axvline(x=sharp_high, color='orange', linestyle='--', alpha=0.5)

    plt.xlabel('Sharpness', fontsize=14, fontweight='bold')
    plt.ylabel('Noise Level', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Over-processing Detection: High Sharpness & Noise\nUpper-right = Noise amplified by sharpening', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_sharpness_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ハルシネーション検出②（Sharpness×Noise）")


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
                   label=f'Unnatural Edges Suspected ({len(unnatural_edges)} cases)', zorder=10)

    plt.xlabel('Edge Density', fontsize=14, fontweight='bold')
    plt.ylabel('Local Quality Std Dev', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Unnatural Edge Detection: High Edge & Quality Variance\nUpper-right = Uneven edge addition', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_edge_vs_local_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ハルシネーション検出③（エッジ×局所品質）")


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

    plt.xlabel('High Frequency Ratio', fontsize=14, fontweight='bold')
    plt.ylabel('Entropy (Information)', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Artificial Pattern Detection: High Freq vs Entropy\nOutliers = Repetitive pattern suspected', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_highfreq_vs_entropy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] ハルシネーション検出④（高周波×Entropy）")


    # ===== 品質トレードオフ系プロット =====

    # 11. SSIM × Noise
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ssim'], model_data['noise'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('SSIM (Structural Similarity)', fontsize=14, fontweight='bold')
    plt.ylabel('Noise Level', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Quality Tradeoff: Structural Similarity vs Noise', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tradeoff_ssim_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] トレードオフ①（SSIM×Noise）")


    # 12. PSNR × Contrast
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['psnr'], model_data['contrast'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('PSNR [dB]', fontsize=14, fontweight='bold')
    plt.ylabel('Contrast', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Quality Tradeoff: Fidelity vs Contrast Enhancement', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Quality Tradeoff: Structural Similarity vs Noise', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_psnr_vs_contrast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] トレードオフ②（PSNR×Contrast）")


    # 13. Sharpness × Artifacts
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['sharpness'], model_data['artifact_total'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('Sharpness', fontsize=14, fontweight='bold')
    plt.ylabel('Artifacts (Blocking + Ringing)', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Quality Tradeoff: Sharpening vs Distortion', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Quality Tradeoff: Fidelity vs Contrast Enhancement', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_sharpness_vs_artifact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] トレードオフ③（Sharpness×Artifacts）")


    # 14. LPIPS × MS-SSIM
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['lpips'], model_data['ms_ssim'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('LPIPS (Perceptual Similarity)', fontsize=14, fontweight='bold')
    plt.ylabel('MS-SSIM', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Perception vs Structure: LPIPS vs MS-SSIM\nNegative correlation expected', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Quality Tradeoff: Sharpening vs Distortion', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_lpips_vs_msssim.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] トレードオフ④（LPIPS×MS-SSIM）")


    # 15. テクスチャ × 高周波成分
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['texture_complexity'], model_data['high_freq_ratio'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('Texture Complexity', fontsize=14, fontweight='bold')
    plt.ylabel('High Frequency Ratio', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Texture vs Frequency Component Consistency', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Perception vs Structure: LPIPS vs MS-SSIM\nNegative correlation expected', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'tradeoff_texture_vs_highfreq.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] トレードオフ⑤（テクスチャ×高周波）")


    # ===== 医療画像特化系プロット =====

    # 16. Contrast × Histogram Correlation
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['contrast'], model_data['histogram_corr'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('Contrast', fontsize=14, fontweight='bold')
    plt.ylabel('Histogram Correlation', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Medical Image Quality: Contrast Enhancement vs Intensity Distribution', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Texture vs Frequency Component Consistency', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_contrast_vs_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 医療特化①（Contrast×ヒストグラム）")


    # 17. Edge Density × Local Quality Mean
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['edge_density'], model_data['local_quality_mean'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('Edge Density', fontsize=14, fontweight='bold')
    plt.ylabel('Local Quality Mean', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Medical Image Quality: Edge Preservation vs Local Quality', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'medical_edge_vs_local_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 医療特化②（エッジ×局所品質）")


    # 18. Noise × Local Quality Std Dev
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['noise'], model_data['local_quality_std'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('Noise Level', fontsize=14, fontweight='bold')
    plt.ylabel('Local Quality Std Dev', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Medical Image Quality: Local Noise Distribution', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Medical Image Quality: Contrast Enhancement vs Intensity Distribution', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_noise_vs_local_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 医療特化③（Noise×局所品質SD）")


    # 19. 色差ΔE × LAB Lightness
    plt.figure(figsize=(12, 8))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['delta_e'], model_data['lab_L_mean'],
                   label=model, alpha=0.6, s=50)
    plt.xlabel('Color Difference ΔE', fontsize=14, fontweight='bold')
    plt.ylabel('LAB Lightness', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Medical Image Quality: Color vs Lightness (Important for pathology)', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Medical Image Quality: Edge Preservation vs Local Quality', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'medical_deltae_vs_lab.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 医療特化④（色差×LAB Lightness）")


    # ===== 分布・PCA系プロット =====

    # 20. Total Scoreのヒストグラム（モデル別）
    plt.figure(figsize=(12, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]['total_score']
        plt.hist(model_data, bins=20, alpha=0.5, label=model, edgecolor='black')
    plt.xlabel('Total Score', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'Total Score Distribution by Model', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Medical Image Quality: Local Noise Distribution', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'distribution_total_score_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 分布①（Total Scoreヒストグラム）")


    # 21. 主成分分析（PCA）プロット
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # 17項目を標準化
    metrics_for_pca = ['ssim', 'ms_ssim', 'psnr', 'lpips', 'sharpness', 'contrast',
                       'entropy', 'noise', 'edge_density', 'artifact_total', 'delta_e',
                       'high_freq_ratio', 'texture_complexity', 'local_quality_mean',
                       'histogram_corr', 'lab_L_mean', 'total_score']

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

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14, fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{f'Principal Component Analysis (PCA): 17 Metrics to 2D\nCumulative variance: {sum(pca.explained_variance_ratio_)*100:.1f}%', fontsize=16, fontweight='bold'}
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Total Score Distribution by Model', fontsize=16, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'pca_2d_projection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 分布②（PCA 2次元プロット）")


    # 22. パーセンタイルバンドプロット（主要指標）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # TITLE_BOTTOM:{'Percentile Band (25%-75%) Plot', fontsize=18, fontweight='bold'}

    key_metrics = ['ssim', 'psnr', 'sharpness', 'noise', 'edge_density', 'total_score']
    metric_names = ['SSIM', 'PSNR [dB]', 'Sharpness', 'Noise', 'Edge Density', 'Total Score']

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
    fig.text(0.5, -0.05, f'Principal Component Analysis (PCA): 17 Metrics to 2D\nCumulative variance: {sum(pca.explained_variance_ratio_)*100:.1f}%', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'percentile_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 分布③（パーセンタイルバンド）")


    # 23. 寄与率グラフ（PCA）
    pca_full = PCA()
    pca_full.fit(X_scaled)

    plt.figure(figsize=(12, 6))
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum)+1), cumsum, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Line')
    plt.xlabel('Number of Components', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Variance Ratio', fontsize=14, fontweight='bold')
    # TITLE_BOTTOM:{'PCACumulative Variance Ratio：何次元で95%説明できるか', fontsize=16, fontweight='bold'}
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Place title at bottom for academic papers
    fig = plt.gcf()
    fig.text(0.5, -0.05, 'Percentile Band (25%-75%) Plot', fontsize=18, fontweight='bold', ha='center', va='bottom', transform=fig.transFigure)
    plt.savefig(output_dir / 'pca_cumulative_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 分布④（PCA寄与率）")

    print(f"{'='*80}")
    print(f"[OK] 全23種類の研究用プロット生成完了")
    print(f"   論文・発表資料にそのまま使用できます（300dpi高解像度）\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"\n使い方:")
        print(f"  python analyze_results.py results/batch_analysis.csv\n")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"[ERROR] エラー: CSVファイルが見つかりません: {csv_file}")
        sys.exit(1)

    analyze_batch_results(csv_file)
