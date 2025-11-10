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
from i18n import I18n

def process_single_pair(args):
    """
    単一の画像ペアを処理（並列処理用）

    Args:
        args: (orig_img_path, model_name, upscaled_dir, output_detail_dir, evaluation_mode, patch_size)

    Returns:
        (success, result_or_error_message)
    """
    orig_img_path, model_name, upscaled_dir, output_detail_dir, evaluation_mode, patch_size = args

    image_id = orig_img_path.stem

    try:
        # 超解像画像のパスを探す（PNG/JPG両対応、サフィックス対応）
        upscaled_path, match_method = find_upscaled_image(orig_img_path, upscaled_dir)

        if upscaled_path is None:
            error_msg = f"[WARNING] 超解像画像が見つかりません: {model_name}/{image_id}"
            return (False, error_msg)

        # 分析実行
        output_subdir = output_detail_dir / model_name / image_id

        results = analyze_images(
            str(orig_img_path),
            str(upscaled_path),
            str(output_subdir),
            str(orig_img_path),
            evaluation_mode,
            comparison_mode='evaluation',
            patch_size=patch_size
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
        error_msg = f"[ERROR] {image_id} - {model_name}: {str(e)}"
        return (False, error_msg)


def find_upscaled_image(original_path, upscaled_dir):
    """
    元画像に対応する超解像画像を検索

    Args:
        original_path: 元画像のPath
        upscaled_dir: 超解像画像ディレクトリのPath

    Returns:
        tuple: (upscaled_path, match_method) またはNoneならマッチなし
               match_method: 'exact_match' | 'suffix_match' | None
    """
    image_id = original_path.stem

    # まず完全一致を試す
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = upscaled_dir / f"{image_id}{ext}"
        if candidate.exists():
            return (candidate, 'exact_match')

    # 見つからない場合、サフィックス付きファイルを検索
    for ext in ['.png', '.jpg', '.jpeg']:
        pattern = f"{image_id}*{ext}"
        matches = list(upscaled_dir.glob(pattern))
        if matches:
            return (matches[0], 'suffix_match')

    return (None, None)


def generate_mapping_csv(original_images, upscaled_dirs, output_dir):
    """
    画像ペアの対応表CSVを生成

    Args:
        original_images: 元画像のPathリスト
        upscaled_dirs: {model_name: upscaled_dir_path}の辞書
        output_dir: 出力ディレクトリ

    Returns:
        Path: 生成されたmapping CSVのパス
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping_csv_path = output_dir / f"mapping_{timestamp}.csv"

    print(f"\n{'='*60}")
    print(f"[INFO] 画像ペア対応表を生成中...")
    print(f"{'='*60}")

    mapping_data = []
    pair_id = 1
    total_original = len(original_images)
    matched_count = 0
    unmatched_count = 0

    for orig_path in original_images:
        for model_name, upscaled_dir in upscaled_dirs.items():
            upscaled_path, match_method = find_upscaled_image(orig_path, upscaled_dir)

            if upscaled_path:
                mapping_data.append({
                    'pair_id': pair_id,
                    'original_name': orig_path.name,
                    'original_path': str(orig_path.resolve()),
                    'upscaled_name': upscaled_path.name,
                    'upscaled_path': str(upscaled_path.resolve()),
                    'model': model_name,
                    'match_method': match_method,
                    'verified': 'false'
                })
                matched_count += 1
            else:
                # マッチしない場合も記録（手動修正用）
                mapping_data.append({
                    'pair_id': pair_id,
                    'original_name': orig_path.name,
                    'original_path': str(orig_path.resolve()),
                    'upscaled_name': 'NOT_FOUND',
                    'upscaled_path': '',
                    'model': model_name,
                    'match_method': 'none',
                    'verified': 'false'
                })
                unmatched_count += 1

            pair_id += 1

    # CSVに書き出し
    df = pd.DataFrame(mapping_data)
    df.to_csv(mapping_csv_path, index=False, encoding='utf-8-sig')

    print(f"[OK] 対応表CSV生成完了: {mapping_csv_path.name}")
    print(f"  総ペア数: {len(mapping_data)}")
    print(f"  マッチ成功: {matched_count}")
    print(f"  マッチ失敗: {unmatched_count}")
    if unmatched_count > 0:
        print(f"  [WARNING] マッチしない画像があります。CSVを確認して手動修正してください。")
    print(f"{'='*60}\n")

    return mapping_csv_path, matched_count, unmatched_count


def batch_analyze(config_file, progress_callback=None, mapping_confirmation_callback=None):
    """
    バッチ処理の実行

    Args:
        config_file: 設定JSONファイルのパス
        progress_callback: 進捗通知用コールバック関数 (current, total, message)
        mapping_confirmation_callback: 対応表CSV確認用コールバック
            Args: (mapping_csv_path, matched_count, unmatched_count)
            Returns: True (続行) or False (中止)
    """

    # 設定読み込み
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # i18n初期化（設定ファイルから言語を取得、デフォルトは日本語）
    language = config.get('language', 'ja')
    i18n = I18n(language)

    # タイムスタンプ生成（CSVとdetailed_*で共通使用）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_dir = Path(config['original_dir'])
    upscaled_dirs = {k: Path(v) for k, v in config['upscaled_dirs'].items()}

    # CSVファイルパスを構築（タイムスタンプ付き）
    base_output_csv = config['output_csv']
    csv_path = Path(base_output_csv)
    output_csv = str(csv_path.parent / f"{csv_path.stem}_{timestamp}{csv_path.suffix}")

    # detailed_*ディレクトリをCSVと同じ場所に出力（タイムスタンプで対応付け）
    output_detail_dir = csv_path.parent / f"detailed_{timestamp}"

    limit = config.get('limit', 0)  # 0 = 全て処理
    append_mode = config.get('append_mode', False)  # False = 上書き, True = 追加
    evaluation_mode = config.get('evaluation_mode', 'image')  # デフォルトは画像モード
    patch_size = config.get('patch_size', 16)  # P6ヒートマップのパッチサイズ（デフォルト: 16×16、論文標準）
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
        print(i18n.t('batch_analyzer.warning_jpg_detected').format(count=len(jpg_images)))
        print(f"{'='*60}")
        print(i18n.t('batch_analyzer.jpg_lossy_warning'))
        print(i18n.t('batch_analyzer.jpg_recommend_png'))
        print(f"")
        print(i18n.t('batch_analyzer.jpg_problems_title'))
        print(i18n.t('batch_analyzer.jpg_problem_1'))
        print(i18n.t('batch_analyzer.jpg_problem_2'))
        print(i18n.t('batch_analyzer.jpg_problem_3'))
        print(i18n.t('batch_analyzer.jpg_problem_4'))
        print(f"")
        print(i18n.t('batch_analyzer.jpg_recommended_actions'))
        print(i18n.t('batch_analyzer.jpg_action_1'))
        print(i18n.t('batch_analyzer.jpg_action_2'))
        print(i18n.t('batch_analyzer.jpg_action_3'))
        print(f"{'='*60}\n")

    original_images = sorted(png_images + jpg_images)

    # 処理枚数制限
    if limit > 0 and limit < len(original_images):
        original_images = original_images[:limit]
        print(i18n.t('batch_analyzer.info_limit_mode').format(limit=limit))

    if len(original_images) == 0:
        print(i18n.t('batch_analyzer.error_original_not_found').format(path=original_dir))
        return

    # 評価モード表示用辞書
    mode_names = {
        'image': '画像モード（医療画像・写真など）',
        'document': '文書モード（カルテ・契約書など）',
        'academic': '学術評価モード（論文用・標準ベンチマーク互換）',
        'developer': '開発者モード（デバッグ用）'
    }

    print(f"\n{'='*60}")
    print(i18n.t('batch_analyzer.batch_start'))
    print(f"{'='*60}")
    print(i18n.t('batch_analyzer.original_dir').format(path=original_dir))
    print(i18n.t('batch_analyzer.original_count').format(count=len(original_images)))
    print(i18n.t('batch_analyzer.model_count').format(count=len(upscaled_dirs)))
    for model_name in upscaled_dirs.keys():
        print(f"   - {model_name}")
    print(i18n.t('batch_analyzer.output_csv').format(path=output_csv))
    print(i18n.t('batch_analyzer.evaluation_mode').format(mode=mode_names.get(evaluation_mode, evaluation_mode)))
    print(i18n.t('batch_analyzer.parallel_workers').format(workers=num_workers))
    print(i18n.t('batch_analyzer.checkpoint_interval').format(interval=checkpoint_interval))
    print(f"{'='*60}\n")

    # 超解像モデルフォルダのJPG検出
    jpg_detected_models = []
    for model_name, upscaled_dir in upscaled_dirs.items():
        jpg_count = len(list(upscaled_dir.glob('*.jpg'))) + len(list(upscaled_dir.glob('*.jpeg')))
        if jpg_count > 0:
            jpg_detected_models.append((model_name, jpg_count))

    if len(jpg_detected_models) > 0:
        print(f"\n{'='*60}")
        print(f"[WARNING] 超解像結果にJPGファイルが検出されました")
        print(f"{'='*60}")
        for model_name, jpg_count in jpg_detected_models:
            print(f"  - {model_name}: {jpg_count}枚のJPGファイル")
        print(f"")
        print(f"JPGは非可逆圧縮のため、AI処理の品質を正確に評価できません。")
        print(f"元のAI超解像ツールでPNG形式で出力し直すことを強く推奨します。")
        print(f"{'='*60}\n")

    # ===== 画像ペア対応表の生成 =====
    # 既存のmapping.csvがあれば優先、なければ自動生成
    output_dir = Path(output_csv).parent
    manual_mapping_path = output_dir / 'mapping.csv'

    if manual_mapping_path.exists():
        print(f"\n{'='*60}")
        print(f"[INFO] 既存の対応表CSVを使用します")
        print(f"  {manual_mapping_path}")
        print(f"{'='*60}\n")
        mapping_csv_path = manual_mapping_path
    else:
        # 自動生成
        mapping_csv_path, matched_count, unmatched_count = generate_mapping_csv(
            original_images, upscaled_dirs, output_dir
        )

        # ユーザーに確認を促す（GUI用コールバック）
        if mapping_confirmation_callback:
            # GUIモード: ダイアログで確認
            proceed = mapping_confirmation_callback(mapping_csv_path, matched_count, unmatched_count)
            if not proceed:
                print(f"\n[INFO] ユーザーによりバッチ処理がキャンセルされました。")
                return
        else:
            # CLIモード: コンソールに情報表示
            print(f"[INFO] 対応表CSVが生成されました。")
            print(f"  確認したい場合は、以下のファイルを開いてください:")
            print(f"  {mapping_csv_path}")
            print(f"  マッチング結果に問題がなければ、このまま処理を続行します。\n")

    # 結果を格納するリスト
    all_results = []
    total_pairs = len(original_images) * len(upscaled_dirs)
    processed = 0
    errors = 0

    # 処理タスクのリストを作成（全ての画像×モデルの組み合わせ）
    tasks = []
    for orig_img_path in original_images:
        for model_name, upscaled_dir in upscaled_dirs.items():
            tasks.append((orig_img_path, model_name, upscaled_dir, output_detail_dir, evaluation_mode, patch_size))

    print(f"処理タスク数: {len(tasks)}")
    print(f"推定処理時間: {len(tasks) * 15 / num_workers / 60:.1f}分 (1サンプル15秒想定)")
    print(f"{'='*60}\n")

    # 開始時刻記録
    start_time = time.time()

    # 並列処理で実行
    print(f"{num_workers}プロセスで並列処理開始...\n")

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
                print(f"[INFO] チェックポイント保存中... ({idx}/{len(tasks)})")
                print(f"  経過時間: {elapsed_time/60:.1f}分")
                print(f"  残り時間: {eta_seconds/60:.1f}分")
                print(f"  成功: {processed}, エラー: {errors}")
                print(f"{'='*60}\n")

                save_results_to_csv(all_results, str(checkpoint_file), append_mode=False)
                print(f"[INFO] チェックポイント保存完了: {checkpoint_file}\n")

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
        print(i18n.t('batch_analyzer.batch_complete'))
        print(f"{'='*60}")
        print(i18n.t('batch_analyzer.success_total').format(success=processed, total=total_pairs))
        print(i18n.t('batch_analyzer.error_total').format(errors=errors, total=total_pairs))
        print(i18n.t('batch_analyzer.total_time').format(minutes=total_time/60, hours=total_time/3600))
        print(i18n.t('batch_analyzer.avg_time').format(seconds=avg_time_per_sample))
        print(i18n.t('batch_analyzer.parallel_efficiency').format(workers=num_workers))
        print(i18n.t('batch_analyzer.model_counts'))
        for model, count in model_counts.items():
            print(i18n.t('batch_analyzer.model_count_item').format(model=model, count=count))
        print(i18n.t('batch_analyzer.result_csv').format(path=output_csv))
        print(i18n.t('batch_analyzer.detail_report').format(path=output_detail_dir))
        if checkpoint_file.exists():
            print(i18n.t('batch_analyzer.checkpoint_file').format(path=checkpoint_file))
        print(f"{'='*60}\n")

        # 簡易統計を表示
        display_summary_statistics(all_results)

        # チェックポイントファイルを削除（正常終了時）
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"\n[INFO] チェックポイントファイル削除済み（正常終了）")
    else:
        print(f"\n[ERROR] 処理可能な画像がありませんでした")


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

    # 画像サイズ情報の取得
    basic_info = results.get('basic_info', {})
    img1_size = basic_info.get('img1_size', [0, 0])  # [width, height]
    img2_size = basic_info.get('img2_size', [0, 0])  # [width, height]

    # ファイル形式の取得
    from pathlib import Path
    original_format = Path(original_path).suffix.lstrip('.').lower()  # "png", "jpg"
    upscaled_format = Path(upscaled_path).suffix.lstrip('.').lower()  # "png", "jpg"

    row = {
        # メタ情報
        'image_id': image_id,
        'model': model_name,
        'original_path': original_path,
        'upscaled_path': upscaled_path,

        # 画像サイズ・フォーマット情報（DB用）
        'original_width': img1_size[0],
        'original_height': img1_size[1],
        'original_format': original_format,
        'upscaled_width': img2_size[0],
        'upscaled_height': img2_size[1],
        'upscaled_format': upscaled_format,

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
        print(f"\n[INFO] 追加モードで保存中...")
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
            print(f"\n[INFO] 追加モードですが既存CSVがないため新規作成します")
        else:
            print(f"\n[INFO] 上書きモードで保存中...")

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

    print(f"\nモデル別平均スコア:")
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
    print(f"総合スコアランキング:")
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
        "_comment_detailed": "detailed_YYYYMMDD_HHMMSS/ is auto-generated in the same directory as output_csv",
        "num_workers": recommended_workers,
        "checkpoint_interval": 1000,
        "evaluation_mode": "academic",
        "patch_size": 16
    }

    config_path = 'batch_config.json'

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 設定ファイルテンプレート作成完了: {config_path}")
    print(f"\n次のステップ:")
    print(f"   1. {config_path} を編集してフォルダパスを設定")
    print(f"   2. python batch_analyzer.py {config_path} を実行")
    print(f"\nヒント:")
    print(f"   - original_dir: 元画像（1000px）のフォルダ")
    print(f"   - upscaled_dirs: 各モデルの超解像結果フォルダ")
    print(f"   - num_workers: 並列処理数（現在のCPU: {num_cpus}コア、推奨: {recommended_workers}）")
    print(f"   - checkpoint_interval: チェックポイント保存間隔（デフォルト: 1000サンプル）")
    print(f"   - evaluation_mode: 評価モード（image/document/academic/developer）")
    print(f"   - patch_size: P6ヒートマップのパッチサイズ（8/16/32/64、デフォルト: 16）")
    print(f"   - 同じファイル名（image001.png等）で対応付けされます")
    print(f"\n15000サンプル処理の場合:")
    print(f"   - 推定時間: 約{15000 * 15 / recommended_workers / 3600:.1f}時間 (1サンプル15秒想定)")
    print(f"   - チェックポイントで中断・再開可能\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(f"\n{'='*60}")
        print(f"バッチ処理スクリプト - 使い方")
        print(f"{'='*60}")
        print(f"\n設定ファイルテンプレートを作成:")
        print(f"   python batch_analyzer.py --create-config")
        print(f"\nバッチ処理を実行:")
        print(f"   python batch_analyzer.py batch_config.json")
        print(f"\n{'='*60}\n")
        sys.exit(1)

    if sys.argv[1] == '--create-config':
        create_config_template()
    else:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"[ERROR] 設定ファイルが見つかりません: {config_file}")
            sys.exit(1)
        batch_analyze(config_file)
