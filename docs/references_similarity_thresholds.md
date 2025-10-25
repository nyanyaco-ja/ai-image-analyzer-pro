# 画像超解像品質評価の閾値に関する参考文献

## 本プロジェクトで使用した閾値の根拠

### SSIM（Structural Similarity Index）閾値

**採用した基準値:**
- **エラー**: SSIM < 0.50 （元画像と全く異なる）
- **警告**: SSIM 0.50-0.70 （ハルシネーションの可能性）
- **許容範囲**: SSIM 0.70-0.85 （適度に高解像度化）
- **良好**: SSIM > 0.85 （高品質な超解像）

**根拠となる研究:**
- 医療画像超解像（2024年）: SRGAN SSIM = 0.8423、改良モデル SSIM = 0.8803
- 医療画像応用における典型的な範囲: SSIM 0.6578 - 0.7812
- 競争力のある性能: SSIM > 0.80

### PSNR（Peak Signal-to-Noise Ratio）閾値

**採用した基準値:**
- **エラー**: PSNR < 20 dB （元画像と全く異なる）
- **警告**: PSNR 20-27 dB （品質低下の可能性）
- **許容範囲**: PSNR 27-30 dB
- **良好**: PSNR > 30 dB

**根拠となる研究:**
- NTIRE 2024 Challenge 最先端性能: PSNR > 31.1 dB
- 医療画像超解像（2024年）: SRGAN PSNR = 28.45 dB、改良モデル PSNR = 28.87 dB
- 医療画像応用における典型的な範囲: PSNR 26.98 - 28.00 dB
- 競争力のある性能: PSNR > 28 dB

## 参考文献リスト

### 1. NTIRE 2024 Challenge on Image Super-Resolution
- **タイトル**: NTIRE 2024 Challenge on Image Super-Resolution (×4): Methods and Results
- **会議**: CVPR 2024 Workshop
- **URL**: https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_NTIRE_2024_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2024_paper.pdf
- **主要な知見**: トップ6チームのPSNRが31.1 dBを超える（2024年最先端性能）

### 2. 医療画像における軽量超解像技術
- **タイトル**: Lightweight Super-Resolution Techniques in Medical Imaging: Bridging Quality and Computational Efficiency
- **出版**: PMC (PubMed Central)
- **年**: 2024
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11673497/
- **主要な知見**:
  - SRGAN: PSNR 28.45 dB, SSIM 0.8423
  - 提案モデル: PSNR 28.87 dB, SSIM 0.8803

### 3. PSNR/SSIM/VMAF品質メトリクスの解説
- **タイトル**: Making Sense of PSNR, SSIM, VMAF
- **出版元**: Visionular
- **URL**: https://visionular.ai/vmaf-ssim-psnr-quality-metrics/
- **主要な知見**: SSIMの値域は[0,1]、値が大きいほど歪みが小さい

### 4. AI画像品質メトリクス実践ガイド（2025年版）
- **タイトル**: AI Image Quality Metrics LPIPS & SSIM Practical Guide 2025
- **出版元**: Unified Image Tools
- **URL**: https://unifiedimagetools.com/en/articles/ai-image-quality-metrics-lpips-ssim-2025
- **主要な知見**: LPIPS等の追加メトリクスの重要性

### 5. Single-Image Super-Resolution ベンチマーク
- **タイトル**: Single-Image Super-Resolution: A Benchmark
- **会議**: ECCV 2014
- **著者**: M.H. Yang et al.
- **URL**: https://faculty.ucmerced.edu/mhyang/papers/eccv14_super.pdf
- **主要な知見**: 超解像評価の基礎的ベンチマーク

### 6. 画像超解像メトリクスの包括的レビュー
- **タイトル**: A comprehensive review of image super-resolution metrics
- **出版**: ACTA IMEKO
- **URL**: https://acta.imeko.org/index.php/acta-imeko/article/view/1679/2939
- **主要な知見**: 主観評価とPSNR/SSIMの相関に関する批判的分析

### 7. PSNR/SSIM: 適用領域と批判
- **タイトル**: PSNR and SSIM: application areas and criticism
- **出版元**: Video Processing Lab
- **URL**: https://videoprocessing.ai/metrics/ways-of-cheating-on-popular-objective-metrics.html
- **主要な知見**:
  - PSNRとSSIMは超解像品質評価において負の相関を示す場合がある
  - 視覚品質との乖離に注意が必要

### 8. NExpR: 医療画像超解像の高速任意スケール処理
- **タイトル**: NExpR: Neural Explicit Representation for Fast Arbitrary-scale Medical Image Super-resolution
- **出版**: PMC
- **年**: 2024
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11663104/

### 9. エッジ情報を活用した画像超解像
- **タイトル**: Image Super-Resolution Improved by Edge Information
- **会議**: IEEE Conference
- **URL**: https://ieeexplore.ieee.org/document/8914550/

### 10. SSIM/PSNR比較テーブル（ResearchGate）
- **タイトル**: SSIM and PSNR comparisons of image super-resolution results
- **URL**: https://www.researchgate.net/figure/SSIM-and-PSNR-comparisons-of-image-super-resolution-results_tbl2_257879060

## 重要な注意事項

1. **メトリクスの限界**: PSNR/SSIMは視覚品質と必ずしも相関しない場合がある
2. **追加メトリクスの推奨**: LPIPS（知覚的類似度）の併用を推奨
3. **ドメイン依存性**: 医療画像と自然画像では適切な閾値が異なる可能性
4. **データ駆動型調整**: 本プロジェクトでは15,000枚の実データで閾値を再調整予定

## 実装日

- **初期実装**: 2025年10月25日
- **論文ベース閾値採用**: NTIRE 2024, 医療画像超解像研究（2024年）を参考
- **予定調整**: 15,000枚分析後、実データに基づく閾値調整

## 今後の改善計画

1. v2.0データセット（15,000枚）の分析
2. 実データに基づく閾値の統計的決定（パーセンタイル法）
3. ドメイン別閾値の検討（X線画像等）
4. LPIPS等の追加メトリクスの統合
5. 深層学習ベースの幻覚検出器との統合
