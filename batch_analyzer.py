"""
バッチ処理スクリプト：大量の画像ペアを自動分析してCSV出力

使い方:
1. 設定ファイルを作成:
   python batch_analyzer.py --create-config

2. batch_config.json を編集してフォルダパスを設定

3. バッチ処理実行:
   python batch_analyzer.py batch_config.json
"""

import os
import csv
import json
import gc
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from advanced_image_analyzer import analyze_images

def process_single_pair(args):
    """
    単一の画像ペアを処理（並列処理用）

    Args:
        args: (orig_img_path, model_name, upscaled_dir, output_detail_dir, evaluation_mode)

    Returns:
        (success, result_or_error_message)
    """
    orig_img_path, model_name, upscaled_dir, output_detail_dir, evaluation_mode = args

    image_id = orig_img_path.stem

    try:
        # 超解像画像のパスを探す（PNG/JPG両対応、サフィックス対応）
        upscaled_path = None

        # まず完全一致を試す
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = upscaled_dir / f"{image_id}{ext}"
            if candidate.exists():
                upscaled_path = candidate
                break

        # 見つからない場合、サフィックス付きファイルを検索
        if upscaled_path is None:
            for ext in ['.png', '.jpg', '.jpeg']:
                # image_id で始まるファイルを検索
                pattern = f"{image_id}*{ext}"
                matches = list(upscaled_dir.glob(pattern))
                if matches:
                    upscaled_path = matches[0]  # 最初にマッチしたものを使用
                    break

        if upscaled_path is None:
            error_msg = f"⚠️  超解像画像が見つかりません: {model_name}/{image_id}"
            return (False, error_msg)

        # 分析実行
        output_subdir = output_detail_dir / model_name / image_id

        results = analyze_images(
            str(orig_img_path),
            str(upscaled_path),
            str(output_subdir),
            str(orig_img_path),
            evaluation_mode
        )

        # 17項目のスコアを抽出
        row = extract_metrics_for_csv(
            image_id,
            model_name,
            results,
            str(orig_img_path),
            str(upscaled_path)
        )

        # メモリ解放
        del results
        gc.collect()

        return (True, row)

    except Exception as e:
        error_msg = f"❌ エラー: {image_id} - {model_name}: {str(e)}"
        return (False, error_msg)

def batch_analyze(config_file, progress_callback=None):
    """
    バッチ処理の実行

    Args:
        config_file: 設定JSONファイルのパス
        progress_callback: 進捗通知用コールバック関数 (current, total, message)
    """

    # 設定読み込み
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    original_dir = Path(config['original_dir'])
    upscaled_dirs = {k: Path(v) for k, v in config['upscaled_dirs'].items()}
    output_csv = config['output_csv']
    output_detail_dir = Path(config.get('output_detail_dir', 'results/detailed/'))
    limit = config.get('limit', 0)  # 0 = 全て処理
    append_mode = config.get('append_mode', False)  # False = 上書き, True = 追加
    evaluation_mode = config.get('evaluation_mode', 'image')  # デフォルトは画像モード
    num_workers = config.get('num_workers', max(1, cpu_count() - 1))  # 並列処理数（デフォルト: CPU数-1）
    checkpoint_interval = config.get('checkpoint_interval', 1000)  # チェックポイント保存間隔
    checkpoint_file = Path(output_csv).parent / f"checkpoint_{Path(output_csv).stem}.csv"

    # 出力ディレクトリ作成
    output_detail_dir.mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # 元画像リスト取得（PNG推奨、JPGは警告）
    png_images = sorted(list(original_dir.glob('*.png')))
    jpg_images = sorted(list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.jpeg')))

    # JPG画像がある場合は警告
    if len(jpg_images) > 0:
        print(f"\n{'='*60}")
        print(f"⚠️  警告: JPGファイルが検出されました ({len(jpg_images)}枚)")
        print(f"{'='*60}")
        print(f"JPGは非可逆圧縮形式のため、すでに画質が劣化しています。")
        print(f"正確な評価のためには、元データからPNG形式で再出力することを強く推奨します。")
        print(f"")
        print(f"【JPGの問題点】")
        print(f"  - ブロックノイズ（8×8ピクセル単位の圧縮アーティファクト）")
        print(f"  - 高周波成分の損失（細かいディテールが消失）")
        print(f"  - 色情報の劣化（色滲み、バンディング）")
        print(f"  - AI超解像の劣化と区別不可能")
        print(f"")
        print(f"【推奨対応】")
        print(f"  1. 元のソフトウェア/カメラからPNG形式で再出力")
        print(f"  2. TIFF等の可逆形式から変換")
        print(f"  3. 医療用途・論文発表には必ずPNG形式を使用")
        print(f"{'='*60}\n")

    original_images = sorted(png_images + jpg_images)

    # 処理枚数制限
    if limit > 0 and limit < len(original_images):
        original_images = original_images[:limit]
        print(f"⚠️  分割実行モード: 最初の{limit}枚のみ処理します")

    if len(original_images) == 0:
        print(f"❌ エラー: 元画像が見つかりません: {original_dir}")
        return

    # 評価モード表示用辞書
    mode_names = {
        'image': '📸 画像モード（医療画像・写真など）',
        'document': '📄 文書モード（カルテ・契約書など）',
        'academic': '📚 学術評価モード（論文用・標準ベンチマーク互換）',
        'developer': '🔧 開発者モード（デバッグ用）'
    }

    print(f"\n{'='*60}")
    print(f"🚀 バッチ処理開始")
    print(f"{'='*60}")
    print(f"📁 元画像ディレクトリ: {original_dir}")
    print(f"🖼️  元画像数: {len(original_images)}枚")
    print(f"🤖 超解像モデル数: {len(upscaled_dirs)}種類")
    for model_name in upscaled_dirs.keys():
        print(f"   - {model_name}")
    print(f"💾 出力CSV: {output_csv}")
    print(f"⚙️  評価モード: {mode_names.get(evaluation_mode, evaluation_mode)}")
    print(f"⚡ 並列処理数: {num_workers}プロセス")
    print(f"💾 チェックポイント間隔: {checkpoint_interval}サンプルごと")
    print(f"{'='*60}\n")

    # 超解像モデルフォルダのJPG検出
    jpg_detected_models = []
    for model_name, upscaled_dir in upscaled_dirs.items():
        jpg_count = len(list(upscaled_dir.glob('*.jpg'))) + len(list(upscaled_dir.glob('*.jpeg')))
        if jpg_count > 0:
            jpg_detected_models.append((model_name, jpg_count))

    if len(jpg_detected_models) > 0:
        print(f"\n{'='*60}")
        print(f"⚠️  警告: 超解像結果にJPGファイルが検出されました")
        print(f"{'='*60}")
        for model_name, jpg_count in jpg_detected_models:
            print(f"  - {model_name}: {jpg_count}枚のJPGファイル")
        print(f"")
        print(f"JPGは非可逆圧縮のため、AI処理の品質を正確に評価できません。")
        print(f"元のAI超解像ツールでPNG形式で出力し直すことを強く推奨します。")
        print(f"{'='*60}\n")

    # 結果を格納するリスト
    all_results = []
    total_pairs = len(original_images) * len(upscaled_dirs)
    processed = 0
    errors = 0

    # 処理タスクのリストを作成（全ての画像×モデルの組み合わせ）
    tasks = []
    for orig_img_path in original_images:
        for model_name, upscaled_dir in upscaled_dirs.items():
            tasks.append((orig_img_path, model_name, upscaled_dir, output_detail_dir, evaluation_mode))

    print(f"📋 処理タスク数: {len(tasks)}")
    print(f"⏱️  推定処理時間: {len(tasks) * 15 / num_workers / 60:.1f}分 (1サンプル15秒想定)")
    print(f"{'='*60}\n")

    # 開始時刻記録
    start_time = time.time()

    # 並列処理で実行
    print(f"⚡ {num_workers}プロセスで並列処理開始...\n")

    with Pool(processes=num_workers) as pool:
        # imapを使って逐次的に結果を取得（メモリ効率化）
        results_iter = pool.imap(process_single_pair, tasks)

        # プログレスバー付きで処理
        for idx, (success, result) in enumerate(tqdm(results_iter, total=len(tasks), desc="バッチ処理中"), 1):
            if success:
                all_results.append(result)
                processed += 1

                # 進捗通知
                if progress_callback:
                    progress_callback(processed, total_pairs, f"完了: {result['image_id']} - {result['model']}")
            else:
                # エラーメッセージを表示
                print(f"\n{result}")
                errors += 1

                # 進捗通知
                if progress_callback:
                    progress_callback(processed + errors, total_pairs, result)

            # チェックポイント保存
            if idx % checkpoint_interval == 0 and len(all_results) > 0:
                elapsed_time = time.time() - start_time
                avg_time_per_sample = elapsed_time / idx
                eta_seconds = avg_time_per_sample * (len(tasks) - idx)

                print(f"\n{'='*60}")
                print(f"💾 チェックポイント保存中... ({idx}/{len(tasks)})")
                print(f"⏱️  経過時間: {elapsed_time/60:.1f}分")
                print(f"⏱️  残り時間: {eta_seconds/60:.1f}分")
                print(f"✔️  成功: {processed}, ❌ エラー: {errors}")
                print(f"{'='*60}\n")

                save_results_to_csv(all_results, str(checkpoint_file), append_mode=False)
                print(f"💾 チェックポイント保存完了: {checkpoint_file}\n")

    # 処理時間計算
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / len(tasks) if len(tasks) > 0 else 0

    # 結果をCSV保存
    if len(all_results) > 0:
        save_results_to_csv(all_results, output_csv, append_mode)

        # モデル別の処理件数を集計
        model_counts = {}
        for row in all_results:
            model = row['model']
            model_counts[model] = model_counts.get(model, 0) + 1

        print(f"\n{'='*60}")
        print(f"✅ バッチ処理完了！")
        print(f"{'='*60}")
        print(f"✔️  成功: {processed} / {total_pairs}")
        print(f"❌ エラー: {errors} / {total_pairs}")
        print(f"⏱️  総処理時間: {total_time/60:.1f}分 ({total_time/3600:.2f}時間)")
        print(f"⚡ 平均処理速度: {avg_time_per_sample:.2f}秒/サンプル")
        print(f"🚀 並列化効率: {num_workers}プロセス使用")
        print(f"\n📊 モデル別処理件数:")
        for model, count in model_counts.items():
            print(f"   {model}: {count}件")
        print(f"\n📄 結果CSV: {output_csv}")
        print(f"📊 詳細レポート: {output_detail_dir}")
        if checkpoint_file.exists():
            print(f"💾 チェックポイント: {checkpoint_file}")
        print(f"{'='*60}\n")

        # 簡易統計を表示
        display_summary_statistics(all_results)

        # チェックポイントファイルを削除（正常終了時）
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"\n💾 チェックポイントファイル削除済み（正常終了）")
    else:
        print(f"\n❌ 処理可能な画像がありませんでした")


def extract_metrics_for_csv(image_id, model_name, results, original_path, upscaled_path):
    """
    分析結果から17項目+メタ情報を抽出してCSV用の行データを作成
    """

    # SSIM/PSNR/色差は元画像ありの場合dict形式
    def safe_extract(data, key1, key2=None):
        """dict型とfloat型の両方に対応"""
        value = data.get(key1, 0)
        if isinstance(value, dict):
            if key2:
                return value.get(key2, 0)
            # dictの場合、img2（超解像）のスコアを返す
            return value.get('img2_vs_original', 0)
        return value if isinstance(value, (int, float)) else 0

    # 色差の取得
    delta_e_data = results.get('color_distribution', {}).get('delta_e', 0)
    if isinstance(delta_e_data, dict):
        delta_e_score = delta_e_data.get('img2_vs_original', 0)
    else:
        delta_e_score = delta_e_data if isinstance(delta_e_data, (int, float)) else 0

    row = {
        # メタ情報
        'image_id': image_id,
        'model': model_name,
        'original_path': original_path,
        'upscaled_path': upscaled_path,

        # 1. SSIM（構造類似性）- 超解像画像のスコア
        'ssim': safe_extract(results, 'ssim', 'img2_vs_original'),

        # 2. MS-SSIM
        'ms_ssim': results.get('ms_ssim', 0),

        # 3. PSNR（信号対雑音比）- 超解像画像のスコア
        'psnr': safe_extract(results, 'psnr', 'img2_vs_original'),

        # 4. LPIPS（知覚的類似度）
        'lpips': results.get('lpips', 0),

        # 5. シャープネス - 超解像画像のスコア
        'sharpness': results.get('sharpness', {}).get('img2', 0),
        'sharpness_diff_pct': results.get('sharpness', {}).get('difference_pct', 0),

        # 6. コントラスト - 超解像画像のスコア
        'contrast': results.get('contrast', {}).get('img2', 0),
        'contrast_diff_pct': results.get('contrast', {}).get('difference_pct', 0),

        # 7. エントロピー - 超解像画像のスコア
        'entropy': results.get('entropy', {}).get('img2', 0),

        # 8. ノイズレベル - 超解像画像のスコア（低いほど良い）
        'noise': results.get('noise', {}).get('img2', 0),

        # 9. エッジ保持率 - 超解像画像のスコア
        'edge_density': results.get('edges', {}).get('img2_density', 0),
        'edge_diff_pct': results.get('edges', {}).get('difference_pct', 0),

        # 10. アーティファクト - 超解像画像のスコア（低いほど良い）
        'artifact_block': results.get('artifacts', {}).get('img2_block_noise', 0),
        'artifact_ringing': results.get('artifacts', {}).get('img2_ringing', 0),
        'artifact_total': (results.get('artifacts', {}).get('img2_block_noise', 0) +
                          results.get('artifacts', {}).get('img2_ringing', 0)),

        # 11. 色差（ΔE）- 超解像画像のスコア（低いほど良い）
        'delta_e': delta_e_score,

        # 12. 高周波成分 - 超解像画像のスコア
        'high_freq_ratio': results.get('frequency_analysis', {}).get('img2', {}).get('high_freq_ratio', 0),

        # 13. テクスチャ - 超解像画像のスコア
        'texture_complexity': results.get('texture', {}).get('img2', {}).get('texture_complexity', 0),

        # 14. 局所品質
        'local_quality_mean': results.get('local_quality', {}).get('mean_ssim', 0),
        'local_quality_std': results.get('local_quality', {}).get('std_ssim', 0),
        'local_quality_min': results.get('local_quality', {}).get('min_ssim', 0),

        # 15. ヒストグラム相関
        'histogram_corr': results.get('histogram_correlation', 0),

        # 16. LAB明度 - 超解像画像のスコア
        'lab_L_mean': results.get('color_distribution', {}).get('img2', {}).get('LAB', {}).get('L_mean', 0),

        # 17. 総合スコア - 超解像画像のスコア
        'total_score': results.get('total_score', {}).get('img2', 0),
    }

    return row


def save_results_to_csv(all_results, output_csv, append_mode=False):
    """
    結果をCSVファイルに保存

    Args:
        all_results: 分析結果のリスト
        output_csv: 出力CSVファイルパス
        append_mode: True = 追加モード, False = 上書きモード
    """

    # DataFrameに変換
    df_new = pd.DataFrame(all_results)

    if append_mode and Path(output_csv).exists():
        # 追加モード: 既存CSVを読み込んで結合
        print(f"\n📊 追加モードで保存中...")
        df_existing = pd.read_csv(output_csv, encoding='utf-8-sig')

        # 重複チェック（同じimage_id + modelの場合は新データで上書き）
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['image_id', 'model'], keep='last')

        df_combined.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"   既存データ: {len(df_existing)}行")
        print(f"   新規データ: {len(df_new)}行")
        print(f"   結合後: {len(df_combined)}行")
        print(f"   ファイル: {output_csv}")
        print(f"   画像数: {df_combined['image_id'].nunique()}")
        print(f"   モデル数: {df_combined['model'].nunique()}")
    else:
        # 上書きモード
        if append_mode:
            print(f"\n📊 追加モードですが既存CSVがないため新規作成します")
        else:
            print(f"\n📊 上書きモードで保存中...")

        df_new.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"   ファイル: {output_csv}")
        print(f"   画像数: {df_new['image_id'].nunique()}")
        print(f"   モデル数: {df_new['model'].nunique()}")
        print(f"   総行数: {len(df_new)}")


def display_summary_statistics(all_results):
    """
    簡易統計を表示
    """

    df = pd.DataFrame(all_results)

    print(f"\n📈 モデル別平均スコア:")
    print(f"{'='*80}")

    # 主要指標でグループ化
    grouped = df.groupby('model').agg({
        'ssim': 'mean',
        'psnr': 'mean',
        'lpips': 'mean',
        'total_score': 'mean',
        'noise': 'mean',
        'artifact_total': 'mean'
    }).round(4)

    # 列名を整形
    grouped.columns = ['SSIM', 'PSNR', 'LPIPS', '総合スコア', 'ノイズ', 'アーティファクト']

    # ソート（総合スコア降順）
    grouped = grouped.sort_values('総合スコア', ascending=False)

    print(grouped.to_string())
    print(f"{'='*80}\n")

    # ランキング
    print(f"🏆 総合スコアランキング:")
    for i, (model, score) in enumerate(grouped['総合スコア'].items(), 1):
        print(f"   {i}位: {model:20s} - {score:.2f}点")


def create_config_template():
    """
    設定ファイルのテンプレートを作成
    """

    # CPU数を取得
    num_cpus = cpu_count()
    recommended_workers = max(1, num_cpus - 1)

    template = {
        "original_dir": "dataset/original/",
        "upscaled_dirs": {
            "upscayl_model1": "dataset/upscayl_model1/",
            "upscayl_model2": "dataset/upscayl_model2/",
            "upscayl_model3": "dataset/upscayl_model3/",
            "chainer_combo1": "dataset/chainer_combo1/",
            "chainer_combo2": "dataset/chainer_combo2/"
        },
        "output_csv": "results/batch_analysis.csv",
        "output_detail_dir": "results/detailed/",
        "num_workers": recommended_workers,
        "checkpoint_interval": 1000,
        "evaluation_mode": "academic"
    }

    config_path = 'batch_config.json'

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 設定ファイルテンプレート作成完了: {config_path}")
    print(f"\n📝 次のステップ:")
    print(f"   1. {config_path} を編集してフォルダパスを設定")
    print(f"   2. python batch_analyzer.py {config_path} を実行")
    print(f"\n💡 ヒント:")
    print(f"   - original_dir: 元画像（1000px）のフォルダ")
    print(f"   - upscaled_dirs: 各モデルの超解像結果フォルダ")
    print(f"   - num_workers: 並列処理数（現在のCPU: {num_cpus}コア、推奨: {recommended_workers}）")
    print(f"   - checkpoint_interval: チェックポイント保存間隔（デフォルト: 1000サンプル）")
    print(f"   - evaluation_mode: 評価モード（image/document/academic/developer）")
    print(f"   - 同じファイル名（image001.png等）で対応付けされます")
    print(f"\n⚡ 15000サンプル処理の場合:")
    print(f"   - 推定時間: 約{15000 * 15 / recommended_workers / 3600:.1f}時間 (1サンプル15秒想定)")
    print(f"   - チェックポイントで中断・再開可能\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(f"\n{'='*60}")
        print(f"バッチ処理スクリプト - 使い方")
        print(f"{'='*60}")
        print(f"\n📋 設定ファイルテンプレートを作成:")
        print(f"   python batch_analyzer.py --create-config")
        print(f"\n🚀 バッチ処理を実行:")
        print(f"   python batch_analyzer.py batch_config.json")
        print(f"\n{'='*60}\n")
        sys.exit(1)

    if sys.argv[1] == '--create-config':
        create_config_template()
    else:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"❌ エラー: 設定ファイルが見つかりません: {config_file}")
            sys.exit(1)
        batch_analyze(config_file)
