"""
分析結果の解釈と評価を行うモジュール
"""

def interpret_results(results):
    """
    分析結果を解釈して、どちらの画像が優れているかを判定

    Returns:
        dict: 各項目の評価と総合判定
    """

    interpretation = {
        'items': [],
        'summary': {},
        'winner': None,
        'winner_count': {'img1': 0, 'img2': 0, 'draw': 0}
    }

    # 1. SSIM（構造類似性）
    ssim_val = results['ssim']
    if ssim_val >= 0.95:
        ssim_eval = "ほぼ同一の画像"
        winner = 'draw'
    elif ssim_val >= 0.80:
        ssim_eval = "非常に似ている"
        winner = 'draw'
    elif ssim_val >= 0.50:
        ssim_eval = "やや似ている"
        winner = 'draw'
    else:
        ssim_eval = "大きく異なる"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'SSIM (構造類似性)',
        'value': f"{ssim_val:.4f}",
        'explanation': '画像の構造的な類似度 (1.0=完全一致)',
        'evaluation': ssim_eval,
        'winner': winner
    })

    # 1.5. MS-SSIM（Multi-Scale SSIM）
    if results.get('ms_ssim') is not None:
        ms_ssim_val = results['ms_ssim']
        if ms_ssim_val >= 0.99:
            ms_ssim_eval = "ほぼ完全に一致（マルチスケールでも同一）"
        elif ms_ssim_val >= 0.95:
            ms_ssim_eval = "非常に類似（複数スケールで高類似）"
        elif ms_ssim_val >= 0.90:
            ms_ssim_eval = "類似（複数スケールで良好）"
        elif ms_ssim_val >= 0.80:
            ms_ssim_eval = "やや類似"
        else:
            ms_ssim_eval = "異なる"

        interpretation['items'].append({
            'name': 'MS-SSIM (マルチスケールSSIM)',
            'value': f"{ms_ssim_val:.4f}",
            'explanation': '複数スケールでの構造類似度 (SSIMの改良版、1.0=完全一致)',
            'evaluation': ms_ssim_eval,
            'winner': 'draw'
        })

    # 2. PSNR（信号対雑音比）
    psnr_val = results['psnr']
    if psnr_val >= 40:
        psnr_eval = "品質差なし（ほぼ同一）"
        winner = 'draw'
    elif psnr_val >= 30:
        psnr_eval = "許容範囲の差"
        winner = 'draw'
    elif psnr_val >= 20:
        psnr_eval = "明確な品質差あり"
        winner = 'draw'
    else:
        psnr_eval = "大幅に異なる画像"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'PSNR (信号対雑音比)',
        'value': f"{psnr_val:.2f} dB",
        'explanation': '画像の劣化度合い (高いほど類似)',
        'evaluation': psnr_eval,
        'winner': winner
    })

    # 2.5. LPIPS（知覚的類似度）
    if results.get('lpips') is not None:
        lpips_val = results['lpips']
        if lpips_val < 0.1:
            lpips_eval = "知覚的にほぼ同一（人間の目では区別困難）"
        elif lpips_val < 0.3:
            lpips_eval = "知覚的に類似"
        elif lpips_val < 0.5:
            lpips_eval = "知覚的にやや異なる"
        else:
            lpips_eval = "知覚的に大きく異なる"

        interpretation['items'].append({
            'name': 'LPIPS (知覚的類似度)',
            'value': f"{lpips_val:.4f}",
            'explanation': 'AI/深層学習ベースの知覚的類似度 (0に近いほど類似)',
            'evaluation': lpips_eval,
            'winner': 'draw'
        })

    # 3. シャープネス（鮮鋭度）
    sharp1 = results['sharpness']['img1']
    sharp2 = results['sharpness']['img2']
    sharp_diff = results['sharpness']['difference_pct']

    if sharp2 > sharp1:
        sharp_eval = f"画像2の方が鮮明 ({sharp_diff:+.1f}%)"
        winner = 'img2'
    elif sharp1 > sharp2:
        sharp_eval = f"画像1の方が鮮明 ({-sharp_diff:+.1f}%)"
        winner = 'img1'
    else:
        sharp_eval = "同等の鮮明さ"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'シャープネス (鮮鋭度)',
        'value': f"画像1: {sharp1:.1f} | 画像2: {sharp2:.1f}",
        'explanation': 'エッジの鮮明さ (高いほど鮮明)',
        'evaluation': sharp_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 4. コントラスト
    contrast1 = results['contrast']['img1']
    contrast2 = results['contrast']['img2']
    contrast_diff = results['contrast']['difference_pct']

    if contrast2 > contrast1:
        contrast_eval = f"画像2の方が高コントラスト ({contrast_diff:+.1f}%)"
        winner = 'img2'
    elif contrast1 > contrast2:
        contrast_eval = f"画像1の方が高コントラスト ({-contrast_diff:+.1f}%)"
        winner = 'img1'
    else:
        contrast_eval = "同等のコントラスト"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'コントラスト',
        'value': f"画像1: {contrast1:.1f} | 画像2: {contrast2:.1f}",
        'explanation': '明暗の差 (高いほどメリハリがある)',
        'evaluation': contrast_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 5. ノイズレベル
    noise1 = results['noise']['img1']
    noise2 = results['noise']['img2']

    if noise2 < noise1:
        noise_eval = f"画像2の方がノイズが少ない (差: {noise1-noise2:.1f})"
        winner = 'img2'
    elif noise1 < noise2:
        noise_eval = f"画像1の方がノイズが少ない (差: {noise2-noise1:.1f})"
        winner = 'img1'
    else:
        noise_eval = "同等のノイズレベル"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'ノイズレベル',
        'value': f"画像1: {noise1:.1f} | 画像2: {noise2:.1f}",
        'explanation': 'ノイズの量 (低いほど綺麗)',
        'evaluation': noise_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 6. エッジ保持率
    edge1 = results['edges']['img1_density']
    edge2 = results['edges']['img2_density']
    edge_diff = results['edges']['difference_pct']

    if edge2 > edge1:
        edge_eval = f"画像2の方が細部を保持 ({edge_diff:+.1f}%)"
        winner = 'img2'
    elif edge1 > edge2:
        edge_eval = f"画像1の方が細部を保持 ({-edge_diff:+.1f}%)"
        winner = 'img1'
    else:
        edge_eval = "同等の細部保持"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'エッジ保持率',
        'value': f"画像1: {edge1:,} | 画像2: {edge2:,}",
        'explanation': '細部・輪郭の保持度 (多いほど詳細)',
        'evaluation': edge_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 7. アーティファクト（歪み）
    artifact1 = results['artifacts']['img1_block_noise'] + results['artifacts']['img1_ringing']
    artifact2 = results['artifacts']['img2_block_noise'] + results['artifacts']['img2_ringing']

    if artifact2 < artifact1:
        artifact_eval = f"画像2の方が歪みが少ない (差: {artifact1-artifact2:.1f})"
        winner = 'img2'
    elif artifact1 < artifact2:
        artifact_eval = f"画像1の方が歪みが少ない (差: {artifact2-artifact1:.1f})"
        winner = 'img1'
    else:
        artifact_eval = "同等の歪みレベル"
        winner = 'draw'

    interpretation['items'].append({
        'name': 'アーティファクト',
        'value': f"画像1: {artifact1:.1f} | 画像2: {artifact2:.1f}",
        'explanation': '圧縮歪み・ブロックノイズ (低いほど良い)',
        'evaluation': artifact_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 8. 色差（ΔE）
    if 'delta_e' in results.get('color_distribution', {}):
        delta_e = results['color_distribution']['delta_e']

        if delta_e < 1:
            color_eval = "色の違いは人間の目では識別不可能"
        elif delta_e < 5:
            color_eval = "許容範囲の色差（ほぼ同じ）"
        elif delta_e < 10:
            color_eval = "明確な色の違いあり"
        else:
            color_eval = "大きく異なる色"

        interpretation['items'].append({
            'name': '色差 (ΔE)',
            'value': f"{delta_e:.2f}",
            'explanation': '知覚的な色の違い (低いほど類似)',
            'evaluation': color_eval,
            'winner': 'draw'
        })

    # 9. 周波数分析
    freq1_high = results['frequency_analysis']['img1']['high_freq_ratio'] * 100
    freq2_high = results['frequency_analysis']['img2']['high_freq_ratio'] * 100

    if abs(freq1_high - freq2_high) < 5:
        freq_eval = "同等の周波数分布"
        winner = 'draw'
    elif freq2_high > freq1_high:
        freq_eval = f"画像2の方が高周波成分が多い（細部が豊富）"
        winner = 'img2'
    else:
        freq_eval = f"画像1の方が高周波成分が多い（細部が豊富）"
        winner = 'img1'

    interpretation['items'].append({
        'name': '高周波成分',
        'value': f"画像1: {freq1_high:.1f}% | 画像2: {freq2_high:.1f}%",
        'explanation': '細かい模様・テクスチャの量',
        'evaluation': freq_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 10. エントロピー（情報量）
    entropy1 = results['entropy']['img1']
    entropy2 = results['entropy']['img2']

    if abs(entropy1 - entropy2) < 0.1:
        entropy_eval = "同等の情報量"
        winner = 'draw'
    elif entropy2 > entropy1:
        entropy_eval = f"画像2の方が情報量が多い（より複雑）"
        winner = 'img2'
    else:
        entropy_eval = f"画像1の方が情報量が多い（より複雑）"
        winner = 'img1'

    interpretation['items'].append({
        'name': 'エントロピー',
        'value': f"画像1: {entropy1:.3f} | 画像2: {entropy2:.3f}",
        'explanation': '画像の情報量・複雑さ (高いほど複雑)',
        'evaluation': entropy_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 11. テクスチャ複雑度
    texture1 = results['texture']['img1']['texture_complexity']
    texture2 = results['texture']['img2']['texture_complexity']

    if abs(texture1 - texture2) < 5:
        texture_eval = "同等のテクスチャ複雑度"
        winner = 'draw'
    elif texture2 > texture1:
        texture_eval = f"画像2の方がテクスチャが豊富"
        winner = 'img2'
    else:
        texture_eval = f"画像1の方がテクスチャが豊富"
        winner = 'img1'

    interpretation['items'].append({
        'name': 'テクスチャ複雑度',
        'value': f"画像1: {texture1:.1f} | 画像2: {texture2:.1f}",
        'explanation': 'テクスチャの複雑さ (高いほど詳細)',
        'evaluation': texture_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 12. 局所品質（パッチSSIM）
    local_ssim_mean = results['local_quality']['mean_ssim']
    local_ssim_std = results['local_quality']['std_ssim']

    if local_ssim_mean >= 0.9:
        local_eval = "局所的にも非常に類似（品質均一）"
    elif local_ssim_mean >= 0.7:
        local_eval = "局所的に良好な類似性"
    elif local_ssim_mean >= 0.5:
        local_eval = "局所的にやや差異あり"
    else:
        local_eval = "局所的に大きな差異あり"

    interpretation['items'].append({
        'name': '局所品質（均一性）',
        'value': f"平均: {local_ssim_mean:.3f}, 標準偏差: {local_ssim_std:.3f}",
        'explanation': 'パッチ単位での品質のばらつき (標準偏差が低いほど均一)',
        'evaluation': local_eval,
        'winner': 'draw'
    })

    # 13. ヒストグラム相関
    hist_corr = results['histogram_correlation']

    if hist_corr >= 0.95:
        hist_eval = "ヒストグラムがほぼ一致"
    elif hist_corr >= 0.80:
        hist_eval = "ヒストグラムが類似"
    elif hist_corr >= 0.50:
        hist_eval = "ヒストグラムにやや差あり"
    else:
        hist_eval = "ヒストグラムが大きく異なる"

    interpretation['items'].append({
        'name': 'ヒストグラム相関',
        'value': f"{hist_corr:.4f}",
        'explanation': '輝度分布の類似度 (1.0=完全一致)',
        'evaluation': hist_eval,
        'winner': 'draw'
    })

    # 14. LAB色空間分析（明度）
    if 'LAB' in results['color_distribution']['img1']:
        lab1_L = results['color_distribution']['img1']['LAB']['L_mean']
        lab2_L = results['color_distribution']['img2']['LAB']['L_mean']

        if abs(lab1_L - lab2_L) < 5:
            lab_eval = "明度がほぼ同等"
            winner = 'draw'
        elif lab2_L > lab1_L:
            lab_eval = f"画像2の方が明るい (差: {lab2_L - lab1_L:.1f})"
            winner = 'draw'
        else:
            lab_eval = f"画像1の方が明るい (差: {lab1_L - lab2_L:.1f})"
            winner = 'draw'

        interpretation['items'].append({
            'name': 'LAB明度',
            'value': f"画像1: {lab1_L:.1f} | 画像2: {lab2_L:.1f}",
            'explanation': '知覚的な明るさ (高いほど明るい)',
            'evaluation': lab_eval,
            'winner': winner
        })

    # 15. 総合スコア比較
    score1 = results['total_score']['img1']
    score2 = results['total_score']['img2']

    if abs(score1 - score2) < 5:
        score_eval = "総合スコアがほぼ同等"
        winner = 'draw'
    elif score2 > score1:
        score_eval = f"画像2の方が総合スコアが高い (差: {score2 - score1:.1f}点)"
        winner = 'img2'
    else:
        score_eval = f"画像1の方が総合スコアが高い (差: {score1 - score2:.1f}点)"
        winner = 'img1'

    interpretation['items'].append({
        'name': '総合スコア',
        'value': f"画像1: {score1:.1f} | 画像2: {score2:.1f}",
        'explanation': '5項目の総合評価 (100点満点)',
        'evaluation': score_eval,
        'winner': winner
    })
    interpretation['winner_count'][winner] += 1

    # 総合判定
    img1_wins = interpretation['winner_count']['img1']
    img2_wins = interpretation['winner_count']['img2']
    draws = interpretation['winner_count']['draw']

    if img1_wins > img2_wins:
        overall_winner = 'img1'
        overall_msg = f"画像1の方が全体的に高品質（{img1_wins}項目で優位）"
    elif img2_wins > img1_wins:
        overall_winner = 'img2'
        overall_msg = f"画像2の方が全体的に高品質（{img2_wins}項目で優位）"
    else:
        overall_winner = 'draw'
        overall_msg = "両画像は同等の品質"

    interpretation['winner'] = overall_winner
    interpretation['summary'] = {
        'img1_wins': img1_wins,
        'img2_wins': img2_wins,
        'draws': draws,
        'message': overall_msg,
        'total_score_img1': results['total_score']['img1'],
        'total_score_img2': results['total_score']['img2']
    }

    return interpretation

def format_interpretation_text(interpretation):
    """解釈結果をテキスト形式で整形"""

    lines = []
    lines.append("=" * 80)
    lines.append("📊 分析結果の解釈（わかりやすい説明）")
    lines.append("=" * 80)
    lines.append("")

    for i, item in enumerate(interpretation['items'], 1):
        lines.append(f"【{i}. {item['name']}】")
        lines.append(f"  数値: {item['value']}")
        lines.append(f"  意味: {item['explanation']}")
        lines.append(f"  評価: {item['evaluation']}")

        # 勝者を表示
        if item['winner'] == 'img1':
            lines.append(f"  ✅ 画像1が優位")
        elif item['winner'] == 'img2':
            lines.append(f"  ✅ 画像2が優位")
        else:
            lines.append(f"  ➖ 同等")
        lines.append("")

    lines.append("=" * 80)
    lines.append("🏆 総合判定")
    lines.append("=" * 80)
    lines.append(f"画像1が優位: {interpretation['summary']['img1_wins']}項目")
    lines.append(f"画像2が優位: {interpretation['summary']['img2_wins']}項目")
    lines.append(f"同等: {interpretation['summary']['draws']}項目")
    lines.append("")
    lines.append(f"総合スコア: 画像1={interpretation['summary']['total_score_img1']:.1f}点 | "
                 f"画像2={interpretation['summary']['total_score_img2']:.1f}点")
    lines.append("")
    lines.append(f"💡 結論: {interpretation['summary']['message']}")
    lines.append("=" * 80)

    return "\n".join(lines)
