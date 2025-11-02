import tkinter as tk
from tkinter import filedialog, messagebox
import os

class DataExtractionMixin:
    """データ抽出機能を提供するMixinクラス"""

    def extract_hallucination_suspects(self):
        """ハルシネーション疑いデータ抽出"""
        csv_path = self.stats_csv_path.get()

        if not csv_path:
            messagebox.showerror("エラー", "CSVファイルを選択してください")
            return

        if not os.path.exists(csv_path):
            messagebox.showerror("エラー", f"CSVファイルが見つかりません:\n{csv_path}")
            return

        try:
            import pandas as pd
            from pathlib import Path

            # CSVを読み込み
            df = pd.read_csv(csv_path, encoding='utf-8-sig')

            # ハルシネーション検出ロジック（17項目すべてを活用）

            # 各データポイントの検出カウント用（多数決ロジック）
            detection_count = pd.Series(0, index=df.index)
            detected_patterns = {idx: [] for idx in df.index}

            # ========== 組み合わせパターン（矛盾・複合異常） ==========

            # === パターン1: SSIM高 × PSNR低（2方式統合） ===
            # 方法A: 固定閾値
            hallucination_1a_fixed = df[(df['ssim'] > 0.97) & (df['psnr'] < 25)]
            # 方法B: 動的閾値
            ssim_high = df['ssim'].quantile(0.75)
            psnr_low = df['psnr'].quantile(0.25)
            hallucination_1b_quantile = df[(df['ssim'] >= ssim_high) & (df['psnr'] <= psnr_low)]
            # 統合
            hallucination_1 = pd.concat([hallucination_1a_fixed, hallucination_1b_quantile]).drop_duplicates()
            detection_count[hallucination_1.index] += 1
            for idx in hallucination_1.index:
                detected_patterns[idx].append('P1:SSIM高×PSNR低')

            # === パターン2: シャープネス高 × ノイズ高 ===
            sharpness_75 = df['sharpness'].quantile(0.75)
            noise_75 = df['noise'].quantile(0.75)
            hallucination_2 = df[(df['sharpness'] > sharpness_75) & (df['noise'] > noise_75)]
            detection_count[hallucination_2.index] += 1
            for idx in hallucination_2.index:
                detected_patterns[idx].append('P2:シャープ高×ノイズ高')

            # === パターン3: エッジ密度高 × 局所品質低 ===
            edge_90 = df['edge_density'].quantile(0.90)
            quality_25 = df['local_quality_mean'].quantile(0.25)
            hallucination_3 = df[(df['edge_density'] > edge_90) & (df['local_quality_mean'] < quality_25)]
            detection_count[hallucination_3.index] += 1
            for idx in hallucination_3.index:
                detected_patterns[idx].append('P3:エッジ高×品質低')

            # === パターン4: Artifacts異常高（GAN特有の歪み） ===
            artifact_90 = df['artifact_total'].quantile(0.90)
            hallucination_4 = df[df['artifact_total'] > artifact_90]
            detection_count[hallucination_4.index] += 1
            for idx in hallucination_4.index:
                detected_patterns[idx].append('P4:Artifacts高')

            # === パターン5: LPIPS高 × SSIM高（知覚と構造の矛盾） ===
            lpips_75 = df['lpips'].quantile(0.75)
            ssim_75 = df['ssim'].quantile(0.75)
            hallucination_5 = df[(df['lpips'] > lpips_75) & (df['ssim'] > ssim_75)]
            detection_count[hallucination_5.index] += 1
            for idx in hallucination_5.index:
                detected_patterns[idx].append('P5:LPIPS高×SSIM高')

            # === パターン6: 局所品質ばらつき大 ===
            if 'local_quality_std' in df.columns:
                quality_std_75 = df['local_quality_std'].quantile(0.75)
                hallucination_6 = df[df['local_quality_std'] > quality_std_75]
                detection_count[hallucination_6.index] += 1
                for idx in hallucination_6.index:
                    detected_patterns[idx].append('P6:品質ばらつき大')
            else:
                hallucination_6 = pd.DataFrame()

            # === パターン7: Entropy低 × High-Freq高（反復パターン） ===
            entropy_25 = df['entropy'].quantile(0.25)
            highfreq_75 = df['high_freq_ratio'].quantile(0.75)
            hallucination_7 = df[(df['entropy'] < entropy_25) & (df['high_freq_ratio'] > highfreq_75)]
            detection_count[hallucination_7.index] += 1
            for idx in hallucination_7.index:
                detected_patterns[idx].append('P7:Entropy低×高周波高')

            # === パターン8: Contrast異常 × Histogram相関低 ===
            contrast_90 = df['contrast'].quantile(0.90)
            histcorr_25 = df['histogram_corr'].quantile(0.25)
            hallucination_8 = df[(df['contrast'] > contrast_90) & (df['histogram_corr'] < histcorr_25)]
            detection_count[hallucination_8.index] += 1
            for idx in hallucination_8.index:
                detected_patterns[idx].append('P8:Contrast異常×Hist相関低')

            # === パターン9: MS-SSIM低 × 総合スコア低 ===
            msssim_25 = df['ms_ssim'].quantile(0.25)
            total_25 = df['total_score'].quantile(0.25)
            hallucination_9 = df[(df['ms_ssim'] < msssim_25) & (df['total_score'] < total_25)]
            detection_count[hallucination_9.index] += 1
            for idx in hallucination_9.index:
                detected_patterns[idx].append('P9:MS-SSIM低×総合低')

            # ========== 単独パターン（各項目の異常値） ==========

            # 高い方が良い指標（異常に低い）
            for col, name in [
                ('ssim', 'SSIM低'), ('ms_ssim', 'MS-SSIM低'), ('psnr', 'PSNR低'),
                ('sharpness', 'Sharpness低'), ('contrast', 'Contrast低'), ('entropy', 'Entropy低'),
                ('edge_density', 'EdgeDensity低'), ('high_freq_ratio', 'HighFreq低'),
                ('texture_complexity', 'Texture低'), ('local_quality_mean', 'LocalQuality低'),
                ('histogram_corr', 'HistCorr低'), ('total_score', 'TotalScore低')
            ]:
                threshold = df[col].quantile(0.10)  # 下位10%
                detected = df[df[col] < threshold]
                detection_count[detected.index] += 1
                for idx in detected.index:
                    detected_patterns[idx].append(f'単独:{name}')

            # 低い方が良い指標（異常に高い）
            for col, name in [
                ('lpips', 'LPIPS高'), ('noise', 'Noise高'), ('artifact_total', 'Artifacts高'),
                ('delta_e', 'DeltaE高')
            ]:
                threshold = df[col].quantile(0.90)  # 上位10%
                detected = df[df[col] > threshold]
                detection_count[detected.index] += 1
                for idx in detected.index:
                    detected_patterns[idx].append(f'単独:{name}')

            # ========== 信頼度分類（多数決） ==========
            high_confidence = df[detection_count >= 5]  # 5パターン以上
            medium_confidence = df[(detection_count >= 3) & (detection_count < 5)]  # 3-4パターン
            low_confidence = df[(detection_count >= 1) & (detection_count < 3)]  # 1-2パターン

            # 全検出データ統合
            hallucination_all = df[detection_count >= 1].copy()
            hallucination_all['detection_count'] = detection_count[hallucination_all.index]
            hallucination_all['detected_patterns'] = hallucination_all.index.map(
                lambda idx: ', '.join(detected_patterns[idx])
            )

            # モデル別集計
            model_counts = hallucination_all['model'].value_counts()

            # パターン別集計
            pattern_counts = {
                'P1:SSIM×PSNR': len(hallucination_1),
                'P1a:固定閾値': len(hallucination_1a_fixed),
                'P1b:動的閾値': len(hallucination_1b_quantile),
                'P2:シャープ×ノイズ': len(hallucination_2),
                'P3:エッジ×品質': len(hallucination_3),
                'P4:Artifacts': len(hallucination_4),
                'P5:LPIPS×SSIM': len(hallucination_5),
                'P6:品質ばらつき': len(hallucination_6),
                'P7:Entropy×高周波': len(hallucination_7),
                'P8:Contrast×Hist': len(hallucination_8),
                'P9:MS-SSIM×総合': len(hallucination_9),
            }

            # 信頼度別集計
            confidence_stats = {
                '高信頼度(5+)': len(high_confidence),
                '中信頼度(3-4)': len(medium_confidence),
                '低信頼度(1-2)': len(low_confidence),
            }

            # 詳細統計
            summary_stats = hallucination_all.groupby('model').agg({
                'ssim': ['mean', 'std', 'min', 'max'],
                'psnr': ['mean', 'std', 'min', 'max'],
                'sharpness': ['mean', 'std'],
                'noise': ['mean', 'std'],
                'total_score': ['mean', 'std'],
                'detection_count': ['mean', 'max']
            }).round(3)

            # 結果表示
            result_text = f"={'='*70}\n"
            result_text += f"ハルシネーション疑いデータ分析結果（17項目全活用）\n"
            result_text += f"={'='*70}\n\n"

            result_text += f"総データ数: {len(df)}件\n"
            result_text += f"ハルシネーション疑い: {len(hallucination_all)}件 ({len(hallucination_all)/len(df)*100:.1f}%)\n\n"

            result_text += f"【信頼度別検出数】\n"
            for conf, count in confidence_stats.items():
                percentage = count / len(df) * 100 if len(df) > 0 else 0
                result_text += f"  {conf}: {count}件 ({percentage:.1f}%)\n"
            result_text += f"\n"

            result_text += f"【組み合わせパターン別検出数】\n"
            result_text += f"  P1 (SSIM高×PSNR低): {pattern_counts['P1:SSIM×PSNR']}件\n"
            result_text += f"    - 固定閾値: {pattern_counts['P1a:固定閾値']}件\n"
            result_text += f"    - 動的閾値: {pattern_counts['P1b:動的閾値']}件\n"
            result_text += f"  P2 (シャープ×ノイズ): {pattern_counts['P2:シャープ×ノイズ']}件\n"
            result_text += f"  P3 (エッジ×品質): {pattern_counts['P3:エッジ×品質']}件\n"
            result_text += f"  P4 (Artifacts高): {pattern_counts['P4:Artifacts']}件\n"
            result_text += f"  P5 (LPIPS×SSIM): {pattern_counts['P5:LPIPS×SSIM']}件\n"
            result_text += f"  P6 (品質ばらつき): {pattern_counts['P6:品質ばらつき']}件\n"
            result_text += f"  P7 (Entropy×高周波): {pattern_counts['P7:Entropy×高周波']}件\n"
            result_text += f"  P8 (Contrast×Hist): {pattern_counts['P8:Contrast×Hist']}件\n"
            result_text += f"  P9 (MS-SSIM×総合): {pattern_counts['P9:MS-SSIM×総合']}件\n"
            result_text += f"  ※単独パターン（17項目）も検出済み\n\n"

            result_text += f"【モデル別】\n"
            for model in sorted(model_counts.index):
                count = model_counts[model]
                percentage = count / len(df) * 100
                avg_detection = hallucination_all[hallucination_all['model'] == model]['detection_count'].mean()
                result_text += f"  {model}: {count}件 ({percentage:.1f}%) 平均検出数: {avg_detection:.1f}\n"

            result_text += f"\n{'='*70}\n"

            # CSV保存（疑いデータ）
            output_path = str(Path(csv_path).parent / f"hallucination_suspects_{Path(csv_path).stem}.csv")
            hallucination_all.to_csv(output_path, index=False, encoding='utf-8-sig')
            result_text += f"[OK] 疑いデータCSV: {output_path}\n"

            # サマリーCSV保存（モデル別統計）
            summary_path = str(Path(csv_path).parent / f"hallucination_summary_{Path(csv_path).stem}.csv")

            # サマリーデータ作成
            summary_data = []
            for model in df['model'].unique():
                model_all = df[df['model'] == model]
                model_hal = hallucination_all[hallucination_all['model'] == model]
                model_high = high_confidence[high_confidence['model'] == model]
                model_medium = medium_confidence[medium_confidence['model'] == model]
                model_low = low_confidence[low_confidence['model'] == model]

                summary_data.append({
                    'model': model,
                    'total_count': len(model_all),
                    'hallucination_count': len(model_hal),
                    'hallucination_rate_%': len(model_hal) / len(model_all) * 100 if len(model_all) > 0 else 0,
                    'high_confidence': len(model_high),
                    'medium_confidence': len(model_medium),
                    'low_confidence': len(model_low),
                    'avg_detection_count': model_hal['detection_count'].mean() if len(model_hal) > 0 else 0,
                    'avg_ssim': model_hal['ssim'].mean() if len(model_hal) > 0 else 0,
                    'avg_psnr': model_hal['psnr'].mean() if len(model_hal) > 0 else 0,
                    'avg_sharpness': model_hal['sharpness'].mean() if len(model_hal) > 0 else 0,
                    'avg_noise': model_hal['noise'].mean() if len(model_hal) > 0 else 0,
                    'avg_total_score': model_hal['total_score'].mean() if len(model_hal) > 0 else 0
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            result_text += f"[OK] サマリーCSV: {summary_path}\n"

            # 詳細統計レポート保存（テキスト）
            report_path = str(Path(csv_path).parent / f"hallucination_report_{Path(csv_path).stem}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
                f.write(f"\n{'='*60}\n")
                f.write("【モデル別詳細統計】\n")
                f.write(f"{'='*60}\n\n")
                f.write(summary_stats.to_string())
            result_text += f"[OK] 詳細レポート: {report_path}\n"

            # グラフ生成
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False

            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # 1. モデル別ハルシネーション発生率（棒グラフ）
            ax1 = fig.add_subplot(gs[0, :2])
            models = []
            rates = []
            for model in sorted(df['model'].unique()):
                model_total = len(df[df['model'] == model])
                model_hal = len(hallucination_all[hallucination_all['model'] == model])
                models.append(model)
                rates.append(model_hal / model_total * 100 if model_total > 0 else 0)

            bars = ax1.bar(models, rates, color=['#4CAF50', '#FFC107', '#F44336'])
            ax1.set_ylabel('ハルシネーション発生率 (%)', fontsize=12)
            ax1.set_title('モデル別ハルシネーション発生率', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)

            # 値をバーの上に表示
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

            # 2. 信頼度別分布（円グラフ）
            ax2 = fig.add_subplot(gs[0, 2])
            conf_labels = ['高\n(5+)', '中\n(3-4)', '低\n(1-2)']
            conf_counts = [len(high_confidence), len(medium_confidence), len(low_confidence)]
            conf_colors = ['#F44336', '#FFC107', '#4CAF50']

            # 0件のパターンを除外
            filtered_labels = []
            filtered_counts = []
            filtered_colors = []
            for label, count, color in zip(conf_labels, conf_counts, conf_colors):
                if count > 0:
                    filtered_labels.append(label)
                    filtered_counts.append(count)
                    filtered_colors.append(color)

            if len(filtered_counts) > 0:
                ax2.pie(filtered_counts, labels=filtered_labels, autopct='%1.1f%%',
                       colors=filtered_colors, startangle=90)
            ax2.set_title('信頼度別分布\n高=5+パターン\n中=3-4パターン\n低=1-2パターン', fontsize=11, fontweight='bold')

            # 3. SSIM vs PSNR散布図（疑いデータ）
            ax3 = fig.add_subplot(gs[1, 0])
            for model in hallucination_all['model'].unique():
                model_data = hallucination_all[hallucination_all['model'] == model]
                ax3.scatter(model_data['ssim'], model_data['psnr'], label=model, alpha=0.6, s=80)
            ax3.set_xlabel('SSIM', fontsize=11)
            ax3.set_ylabel('PSNR (dB)', fontsize=11)
            ax3.set_title('SSIM vs PSNR（疑いデータ）', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)

            # 4. シャープネス vs ノイズ散布図（疑いデータ）
            ax4 = fig.add_subplot(gs[1, 1])
            for model in hallucination_all['model'].unique():
                model_data = hallucination_all[hallucination_all['model'] == model]
                ax4.scatter(model_data['sharpness'], model_data['noise'], label=model, alpha=0.6, s=80)
            ax4.set_xlabel('シャープネス', fontsize=11)
            ax4.set_ylabel('ノイズ', fontsize=11)
            ax4.set_title('シャープネス vs ノイズ（疑いデータ）', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)

            # 5. エッジ密度 vs 局所品質散布図（疑いデータ）
            ax5 = fig.add_subplot(gs[1, 2])
            for model in hallucination_all['model'].unique():
                model_data = hallucination_all[hallucination_all['model'] == model]
                ax5.scatter(model_data['edge_density'], model_data['local_quality_mean'], label=model, alpha=0.6, s=80)
            ax5.set_xlabel('エッジ密度', fontsize=11)
            ax5.set_ylabel('局所品質', fontsize=11)
            ax5.set_title('エッジ密度 vs 局所品質（疑いデータ）', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)

            # 6. モデル別平均スコア比較（レーダーチャート）
            ax6 = fig.add_subplot(gs[2, :], projection='polar')

            categories = ['SSIM', 'PSNR/50', 'シャープネス\n(正規化)', 'ノイズ\n(反転)', '総合スコア/100']
            angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
            angles += angles[:1]

            for model in sorted(hallucination_all['model'].unique()):
                model_data = hallucination_all[hallucination_all['model'] == model]
                if len(model_data) > 0:
                    values = [
                        model_data['ssim'].mean(),
                        model_data['psnr'].mean() / 50,
                        min(model_data['sharpness'].mean() / 300, 1.0),
                        1.0 - min(model_data['noise'].mean() / 0.1, 1.0),
                        model_data['total_score'].mean() / 100
                    ]
                    values += values[:1]
                    ax6.plot(angles, values, 'o-', linewidth=2, label=model)
                    ax6.fill(angles, values, alpha=0.15)

            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(categories, fontsize=10)
            ax6.set_ylim(0, 1)
            ax6.set_title('モデル別平均スコア比較（ハルシネーション疑いデータ）', fontsize=14, fontweight='bold', pad=20)
            ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax6.grid(True)

            plt.suptitle(f'ハルシネーション疑いデータ分析（17項目全活用） (n={len(hallucination_all)})',
                        fontsize=16, fontweight='bold', y=0.98)

            # 保存
            graph_path = str(Path(csv_path).parent / f"hallucination_analysis_{Path(csv_path).stem}.png")
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()

            result_text += f"[OK] 分析グラフ: {graph_path}\n"
            result_text += f"{'='*60}\n"

            # 結果表示
            self.batch_result_text.delete("1.0", tk.END)
            self.batch_result_text.insert("1.0", result_text)

            self.batch_status_label.configure(
                text=f"[OK] ハルシネーション疑いデータ抽出完了（{len(hallucination_all)}件）",
                text_color="#ff4444"
            )

            messagebox.showinfo(
                "抽出完了",
                f"ハルシネーション疑いデータを抽出しました。\n\n"
                f"総データ数: {len(df)}件\n"
                f"疑いあり: {len(hallucination_all)}件 ({len(hallucination_all)/len(df)*100:.1f}%)\n\n"
                f"保存先:\n{output_path}\n\n"
                f"このCSVで再度統計分析を実行できます。"
            )

        except Exception as e:
            messagebox.showerror("エラー", f"ハルシネーション抽出中にエラーが発生しました:\n{str(e)}")

    def extract_clean_dataset(self):
        """正常データ（検出0）を抽出してクリーンデータセットを作成"""
        try:
            csv_path = filedialog.askopenfilename(
                title="バッチ分析CSVを選択",
                filetypes=[("CSV files", "*.csv")]
            )
            if not csv_path:
                return

            from pathlib import Path
            import pandas as pd
            import shutil
            from datetime import datetime

            self.batch_status_label.configure(
                text="⏳ 正常データ抽出中...",
                text_color="#ffaa00"
            )
            self.root.update()

            # CSVを読み込み
            df = pd.read_csv(csv_path, encoding='utf-8-sig')

            # ハルシネーション検出ロジック実行（detection_count計算）
            detection_count = pd.Series(0, index=df.index)
            detected_patterns = {idx: [] for idx in df.index}

            # === 組み合わせパターン ===
            # P1: SSIM高 × PSNR低
            hallucination_1a_fixed = df[(df['ssim'] > 0.97) & (df['psnr'] < 25)]
            ssim_high = df['ssim'].quantile(0.75)
            psnr_low = df['psnr'].quantile(0.25)
            hallucination_1b_quantile = df[(df['ssim'] >= ssim_high) & (df['psnr'] <= psnr_low)]
            hallucination_1 = pd.concat([hallucination_1a_fixed, hallucination_1b_quantile]).drop_duplicates()
            detection_count[hallucination_1.index] += 1
            for idx in hallucination_1.index:
                detected_patterns[idx].append('P1')

            # P2: シャープネス高 × ノイズ高
            sharpness_75 = df['sharpness'].quantile(0.75)
            noise_75 = df['noise'].quantile(0.75)
            hallucination_2 = df[(df['sharpness'] > sharpness_75) & (df['noise'] > noise_75)]
            detection_count[hallucination_2.index] += 1
            for idx in hallucination_2.index:
                detected_patterns[idx].append('P2')

            # P3: エッジ密度高 × 局所品質低
            edge_90 = df['edge_density'].quantile(0.90)
            quality_25 = df['local_quality_mean'].quantile(0.25)
            hallucination_3 = df[(df['edge_density'] > edge_90) & (df['local_quality_mean'] < quality_25)]
            detection_count[hallucination_3.index] += 1
            for idx in hallucination_3.index:
                detected_patterns[idx].append('P3')

            # P4: Artifacts異常高
            artifact_90 = df['artifact_total'].quantile(0.90)
            hallucination_4 = df[df['artifact_total'] > artifact_90]
            detection_count[hallucination_4.index] += 1
            for idx in hallucination_4.index:
                detected_patterns[idx].append('P4')

            # P5: LPIPS高 × SSIM高
            lpips_75 = df['lpips'].quantile(0.75)
            ssim_75 = df['ssim'].quantile(0.75)
            hallucination_5 = df[(df['lpips'] > lpips_75) & (df['ssim'] > ssim_75)]
            detection_count[hallucination_5.index] += 1
            for idx in hallucination_5.index:
                detected_patterns[idx].append('P5')

            # P6: 局所品質ばらつき大
            if 'local_quality_std' in df.columns:
                quality_std_75 = df['local_quality_std'].quantile(0.75)
                hallucination_6 = df[df['local_quality_std'] > quality_std_75]
                detection_count[hallucination_6.index] += 1
                for idx in hallucination_6.index:
                    detected_patterns[idx].append('P6')

            # P7-P9省略（必要に応じて追加）

            # === 単独パターン ===
            for col, name in [
                ('ssim', 'SSIM'), ('ms_ssim', 'MS-SSIM'), ('psnr', 'PSNR'),
                ('sharpness', 'Sharpness'), ('contrast', 'Contrast'), ('entropy', 'Entropy'),
                ('edge_density', 'EdgeDensity'), ('high_freq_ratio', 'HighFreq'),
                ('texture_complexity', 'Texture'), ('local_quality_mean', 'LocalQuality'),
                ('histogram_corr', 'HistCorr'), ('total_score', 'TotalScore')
            ]:
                threshold = df[col].quantile(0.10)
                detected = df[df[col] < threshold]
                detection_count[detected.index] += 1

            for col, name in [
                ('lpips', 'LPIPS'), ('noise', 'Noise'), ('artifact_total', 'Artifacts'),
                ('delta_e', 'DeltaE')
            ]:
                threshold = df[col].quantile(0.90)
                detected = df[df[col] > threshold]
                detection_count[detected.index] += 1

            # 正常データ抽出（detection_count == 0）
            normal_df = df[detection_count == 0].copy()

            # 出力ディレクトリ作成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(csv_path).parent / f"clean_dataset_{timestamp}"
            output_dir.mkdir(exist_ok=True)

            # モデル別フォルダ作成
            original_dir = output_dir / "original"
            original_dir.mkdir(exist_ok=True)

            model_dirs = {}
            for model in df['model'].unique():
                model_dir = output_dir / f"{model}_clean"
                model_dir.mkdir(exist_ok=True)
                model_dirs[model] = model_dir

            # ファイルコピー
            copied_files = []
            metadata = []

            for image_id in normal_df['image_id'].unique():
                # 元画像をコピー（1回のみ）
                image_rows = normal_df[normal_df['image_id'] == image_id]
                if len(image_rows) > 0:
                    first_row = image_rows.iloc[0]
                    original_path = first_row['original_path']

                    if os.path.exists(original_path):
                        dest_orig = original_dir / Path(original_path).name
                        if not dest_orig.exists():
                            shutil.copy2(original_path, dest_orig)
                            copied_files.append(str(dest_orig))

                # モデル別超解像画像をコピー
                model_status = {}
                for model in df['model'].unique():
                    model_row = image_rows[image_rows['model'] == model]
                    if len(model_row) > 0:
                        upscaled_path = model_row.iloc[0]['upscaled_path']
                        if os.path.exists(upscaled_path):
                            dest_upscaled = model_dirs[model] / Path(upscaled_path).name
                            shutil.copy2(upscaled_path, dest_upscaled)
                            copied_files.append(str(dest_upscaled))
                            model_status[model] = 'clean'
                        else:
                            model_status[model] = 'missing'
                    else:
                        model_status[model] = 'hallucination'

                # メタデータ作成
                metadata_row = {
                    'image_id': image_id,
                    'original_path': str(dest_orig) if 'dest_orig' in locals() else '',
                }
                for model in sorted(df['model'].unique()):
                    metadata_row[f'{model}_status'] = model_status.get(model, 'none')
                    model_row = normal_df[(normal_df['image_id'] == image_id) & (normal_df['model'] == model)]
                    if len(model_row) > 0:
                        metadata_row[f'{model}_ssim'] = model_row.iloc[0]['ssim']
                        metadata_row[f'{model}_psnr'] = model_row.iloc[0]['psnr']
                        metadata_row[f'{model}_total_score'] = model_row.iloc[0]['total_score']

                metadata.append(metadata_row)

            # metadata.csv保存
            metadata_df = pd.DataFrame(metadata)
            metadata_path = output_dir / "metadata.csv"
            metadata_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')

            # README作成
            readme_path = output_dir / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("クリーンデータセット（正常画像のみ）\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"元データ: {csv_path}\n\n")
                f.write(f"総画像数: {len(normal_df['image_id'].unique())}枚\n")
                for model in sorted(df['model'].unique()):
                    count = len(normal_df[normal_df['model'] == model])
                    f.write(f"  {model}: {count}枚\n")
                f.write("\n【フォルダ構成】\n")
                f.write("  original/      : 元画像\n")
                for model in sorted(df['model'].unique()):
                    f.write(f"  {model}_clean/ : {model}で正常な超解像画像\n")
                f.write("  metadata.csv   : 詳細情報（AI学習用）\n\n")
                f.write("【使い方】\n")
                f.write("1. AI学習データとして使用\n")
                f.write("2. 品質フィルタリング済みデータセット\n")
                f.write("3. ベンチマークデータ\n\n")
                f.write("※ ハルシネーション検出で問題なしと判定された画像のみを含みます\n")

            # 結果表示
            result_text = f"=" * 70 + "\n"
            result_text += "[OK] クリーンデータセット作成完了\n"
            result_text += "=" * 70 + "\n\n"
            result_text += f"[FOLDER] 出力先: {output_dir}\n\n"
            result_text += f"[STATS] 統計:\n"
            result_text += f"  総データ数: {len(df)}件\n"
            result_text += f"  正常データ: {len(normal_df)}件 ({len(normal_df)/len(df)*100:.1f}%)\n"
            result_text += f"  正常画像数: {len(normal_df['image_id'].unique())}枚\n\n"
            result_text += f"【モデル別正常データ】\n"
            for model in sorted(df['model'].unique()):
                count = len(normal_df[normal_df['model'] == model])
                total = len(df[df['model'] == model])
                result_text += f"  {model}: {count}/{total}件 ({count/total*100:.1f}%)\n"
            result_text += f"\n[FILE] ファイル:\n"
            result_text += f"  metadata.csv : メタデータ\n"
            result_text += f"  README.txt   : 説明書\n"
            result_text += f"  コピーファイル数: {len(copied_files)}個\n"

            self.batch_result_text.delete("1.0", tk.END)
            self.batch_result_text.insert("1.0", result_text)

            self.batch_status_label.configure(
                text=f"[OK] クリーンデータセット作成完了（{len(normal_df['image_id'].unique())}枚）",
                text_color="#44ff44"
            )

            messagebox.showinfo(
                "作成完了",
                f"クリーンデータセット（正常画像のみ）を作成しました。\n\n"
                f"正常画像数: {len(normal_df['image_id'].unique())}枚\n"
                f"出力先: {output_dir}\n\n"
                f"AI学習データとして使用できます。"
            )

        except Exception as e:
            import traceback
            messagebox.showerror("エラー", f"クリーンデータセット作成中にエラーが発生しました:\n{str(e)}\n\n{traceback.format_exc()}")

