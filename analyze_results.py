"""
統計分析スクリプト：バッチ処理結果から閾値を決定

使い方:
python analyze_results.py results/batch_analysis.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def analyze_batch_results(csv_file):
    """
    バッチ処理結果の統計分析
    """

    # CSV読み込み
    df = pd.read_csv(csv_file)

    print(f"\n{'='*80}")
    print(f"📊 統計分析レポート")
    print(f"{'='*80}")
    print(f"📄 データファイル: {csv_file}")
    print(f"📷 画像数: {df['image_id'].nunique()}")
    print(f"🤖 モデル数: {df['model'].nunique()}")
    print(f"📊 総データ数: {len(df)}")
    print(f"{'='*80}\n")

    # 出力ディレクトリ作成
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)

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

    print(f"\n✅ 分析完了！")
    print(f"📁 結果保存先: {output_dir}/")


def print_basic_statistics(df):
    """
    基本統計量の表示
    """

    print(f"\n📈 主要指標の基本統計量:")
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

    print(f"\n🏆 モデル別ランキング:")
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

    # 総合スコアでソート
    model_comparison = model_comparison.sort_values(('total_score', 'mean'), ascending=False)

    print(model_comparison.to_string())
    print(f"{'='*80}\n")

    # CSV保存
    model_comparison.to_csv(output_dir / 'model_comparison.csv', encoding='utf-8-sig')

    # 可視化：モデル別総合スコア
    plt.figure(figsize=(12, 6))
    model_scores = df.groupby('model')['total_score'].mean().sort_values(ascending=False)

    plt.bar(range(len(model_scores)), model_scores.values)
    plt.xticks(range(len(model_scores)), model_scores.index, rotation=45, ha='right')
    plt.ylabel('総合スコア（平均）')
    plt.title('モデル別総合スコア比較')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_scores.png', dpi=150)
    plt.close()

    print(f"📊 グラフ保存: {output_dir}/model_scores.png")


def analyze_correlations(df, output_dir):
    """
    17項目間の相関分析
    """

    print(f"\n🔗 相関分析:")
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
    plt.title('17項目の相関マトリックス', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150)
    plt.close()

    print(f"📊 相関マトリックス保存: {output_dir}/correlation_matrix.png")

    # 高相関ペアを表示
    print(f"\n🔥 高相関ペア（|r| > 0.7）:")
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
    根拠のある閾値を提案
    """

    print(f"\n💡 推奨閾値の提案:")
    print(f"{'='*80}")

    thresholds = {}

    # 各指標の統計値から閾値を決定
    # 17項目すべての閾値を提案
    metrics_config = {
        'ssim': {'direction': 'high', 'percentile': 25, 'name': 'SSIM（構造類似性）'},
        'ms_ssim': {'direction': 'high', 'percentile': 25, 'name': 'MS-SSIM（マルチスケールSSIM）'},
        'psnr': {'direction': 'high', 'percentile': 25, 'name': 'PSNR（信号対雑音比）'},
        'lpips': {'direction': 'low', 'percentile': 75, 'name': 'LPIPS（知覚的類似度）'},
        'sharpness': {'direction': 'high', 'percentile': 25, 'name': 'シャープネス'},
        'contrast': {'direction': 'high', 'percentile': 25, 'name': 'コントラスト'},
        'entropy': {'direction': 'high', 'percentile': 25, 'name': 'エントロピー（情報量）'},
        'noise': {'direction': 'low', 'percentile': 75, 'name': 'ノイズレベル'},
        'edge_density': {'direction': 'high', 'percentile': 25, 'name': 'エッジ密度'},
        'artifact_total': {'direction': 'low', 'percentile': 75, 'name': 'アーティファクト'},
        'delta_e': {'direction': 'low', 'percentile': 75, 'name': '色差（ΔE）'},
        'high_freq_ratio': {'direction': 'high', 'percentile': 25, 'name': '高周波成分比率'},
        'texture_complexity': {'direction': 'high', 'percentile': 25, 'name': 'テクスチャ複雑度'},
        'local_quality_mean': {'direction': 'high', 'percentile': 25, 'name': '局所品質平均'},
        'histogram_corr': {'direction': 'high', 'percentile': 25, 'name': 'ヒストグラム相関'},
        'lab_L_mean': {'direction': 'neutral', 'percentile': 50, 'name': 'LAB明度（参考値）'},
        'total_score': {'direction': 'high', 'percentile': 25, 'name': '総合スコア'},
    }

    for metric, config in metrics_config.items():
        data = df[metric].dropna()

        if config['direction'] == 'neutral':
            # 中立的な指標（明度など）：中央値を参考値として表示
            threshold = np.percentile(data, config['percentile'])
            condition = f"参考値: {threshold:.4f}"
        elif config['direction'] == 'high':
            # 高い方が良い指標：25パーセンタイル以上を推奨
            threshold = np.percentile(data, config['percentile'])
            condition = f">= {threshold:.4f}"
        else:
            # 低い方が良い指標：75パーセンタイル以下を推奨
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

        print(f"{config['name']:30s}: {condition:20s} (平均: {data.mean():.4f}, 標準偏差: {data.std():.4f})")

    print(f"{'='*80}")
    print(f"💡 解釈:")
    print(f"   - これらの閾値は、全データの統計分布に基づいています")
    print(f"   - 25パーセンタイル = 上位75%の品質を「合格」とする基準")
    print(f"   - 75パーセンタイル = 下位75%の品質を「合格」とする基準")
    print(f"{'='*80}\n")

    # JSON保存
    import json
    with open(output_dir / 'recommended_thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    print(f"💾 閾値保存: {output_dir}/recommended_thresholds.json\n")


def suggest_hallucination_logic(df, output_dir):
    """
    ハルシネーション検出ロジックの提案
    """

    print(f"\n🔍 ハルシネーション検出ロジックの提案:")
    print(f"{'='*80}")

    # パターン1: SSIM高いのにPSNR低い（構造は似てるがピクセル値が違う）
    ssim_high = df['ssim'].quantile(0.75)
    psnr_low = df['psnr'].quantile(0.25)

    pattern1 = df[(df['ssim'] >= ssim_high) & (df['psnr'] <= psnr_low)]
    pattern1_rate = len(pattern1) / len(df) * 100

    print(f"【パターン1】SSIM高 & PSNR低 (構造類似だがピクセル値相違)")
    print(f"   条件: SSIM >= {ssim_high:.4f} AND PSNR <= {psnr_low:.2f}")
    print(f"   該当率: {pattern1_rate:.1f}% ({len(pattern1)}/{len(df)}件)")
    print(f"   リスク: 中～高（AIが構造を模倣した可能性）")

    # パターン2: シャープネス高いがノイズも高い（過剰処理）
    sharp_high = df['sharpness'].quantile(0.75)
    noise_high = df['noise'].quantile(0.75)

    pattern2 = df[(df['sharpness'] >= sharp_high) & (df['noise'] >= noise_high)]
    pattern2_rate = len(pattern2) / len(df) * 100

    print(f"\n【パターン2】シャープネス高 & ノイズ高 (過剰処理)")
    print(f"   条件: シャープネス >= {sharp_high:.2f} AND ノイズ >= {noise_high:.2f}")
    print(f"   該当率: {pattern2_rate:.1f}% ({len(pattern2)}/{len(df)}件)")
    print(f"   リスク: 中（過度なシャープ化によるノイズ増幅）")

    # パターン3: アーティファクト高（GAN特有）
    artifact_high = df['artifact_total'].quantile(0.90)

    pattern3 = df[df['artifact_total'] >= artifact_high]
    pattern3_rate = len(pattern3) / len(df) * 100

    print(f"\n【パターン3】アーティファクト高 (GAN特有の歪み)")
    print(f"   条件: アーティファクト >= {artifact_high:.2f}")
    print(f"   該当率: {pattern3_rate:.1f}% ({len(pattern3)}/{len(df)}件)")
    print(f"   リスク: 高（リンギング・ブロックノイズによる診断阻害）")

    # パターン4: 局所品質のばらつき大
    local_std_high = df['local_quality_std'].quantile(0.75)

    pattern4 = df[df['local_quality_std'] >= local_std_high]
    pattern4_rate = len(pattern4) / len(df) * 100

    print(f"\n【パターン4】局所品質のばらつき大 (不均一な処理)")
    print(f"   条件: 局所SSIM標準偏差 >= {local_std_high:.4f}")
    print(f"   該当率: {pattern4_rate:.1f}% ({len(pattern4)}/{len(df)}件)")
    print(f"   リスク: 中～高（領域によって品質が異なる = 一部にハルシネーション）")

    print(f"{'='*80}\n")

    # 総合ハルシネーションリスクスコアの計算
    print(f"📊 総合ハルシネーションリスクスコアの提案:")
    print(f"{'='*80}")

    df['hallucination_risk_score'] = 0

    # パターン1該当: +25点
    df.loc[(df['ssim'] >= ssim_high) & (df['psnr'] <= psnr_low), 'hallucination_risk_score'] += 25

    # パターン2該当: +20点
    df.loc[(df['sharpness'] >= sharp_high) & (df['noise'] >= noise_high), 'hallucination_risk_score'] += 20

    # パターン3該当: +30点
    df.loc[df['artifact_total'] >= artifact_high, 'hallucination_risk_score'] += 30

    # パターン4該当: +25点
    df.loc[df['local_quality_std'] >= local_std_high, 'hallucination_risk_score'] += 25

    # リスクレベル分類
    df['risk_level'] = pd.cut(df['hallucination_risk_score'],
                               bins=[0, 10, 30, 50, 100],
                               labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'])

    # リスク分布
    risk_dist = df['risk_level'].value_counts().sort_index()
    print(f"\nハルシネーションリスク分布:")
    for level, count in risk_dist.items():
        pct = count / len(df) * 100
        print(f"   {level:10s}: {count:4d}件 ({pct:5.1f}%)")

    # リスク付きCSV保存
    output_csv = output_dir / 'results_with_risk_score.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 リスクスコア付き結果保存: {output_csv}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"\n使い方:")
        print(f"  python analyze_results.py results/batch_analysis.csv\n")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"❌ エラー: CSVファイルが見つかりません: {csv_file}")
        sys.exit(1)

    analyze_batch_results(csv_file)
