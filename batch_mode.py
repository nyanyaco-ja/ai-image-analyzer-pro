import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
import sys
import json
from datetime import datetime
from ui_components import AccordionSection, get_timestamp_filename
import customtkinter as ctk

class BatchModeMixin:
    """バッチモード機能を提供するMixinクラス"""

    def create_batch_mode_ui(self):
        """バッチ処理モードのUI作成（左パネル）"""

        # 説明セクション
        info_frame = ctk.CTkFrame(self.batch_mode_frame, fg_color="#2d3748", corner_radius=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))

        self.batch_info_title = ctk.CTkLabel(
            info_frame,
            text=self.i18n.t('batch.title'),
            font=("Arial", 18, "bold"),
            text_color="#4A90E2"
        )
        self.batch_info_title.pack(anchor="w", padx=15, pady=(15, 5))

        self.batch_info_text = ctk.CTkLabel(
            info_frame,
            text=self.i18n.t('batch.description'),
            font=("Arial", 13),
            text_color="#cccccc",
            justify="left"
        )
        self.batch_info_text.pack(anchor="w", padx=15, pady=(0, 15))

        # === アコーディオン: 評価モード選択 ===
        self.batch_evaluation_mode = tk.StringVar(value="image")

        self.eval_accordion = AccordionSection(
            self.batch_mode_frame,
            self.i18n.t('sections.evaluation_settings'),
            bg_color="#1e2740",
            title_color="#4A90E2",
            font_size=18
        )

        # 評価モード選択フレーム
        mode_select_frame = ctk.CTkFrame(self.eval_accordion.content_frame, fg_color="transparent")
        mode_select_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 画像モード（翻訳対応）
        self.batch_mode_image = ctk.CTkRadioButton(
            mode_select_frame,
            text=self.i18n.t('modes.image'),
            variable=self.batch_evaluation_mode,
            value="image",
            font=("Arial", 14),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        self.batch_mode_image.pack(anchor="w", padx=30, pady=(0, 8))

        self.batch_mode_image_desc = ctk.CTkLabel(
            mode_select_frame,
            text=f"  {self.i18n.t('modes.image_desc')}",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.batch_mode_image_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # 文書モード（翻訳対応）
        self.batch_mode_document = ctk.CTkRadioButton(
            mode_select_frame,
            text=self.i18n.t('modes.document'),
            variable=self.batch_evaluation_mode,
            value="document",
            font=("Arial", 14),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        self.batch_mode_document.pack(anchor="w", padx=30, pady=(0, 8))

        self.batch_mode_document_desc = ctk.CTkLabel(
            mode_select_frame,
            text=f"  {self.i18n.t('modes.document_desc')}",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.batch_mode_document_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # 開発者モード（翻訳対応）
        self.batch_mode_developer = ctk.CTkRadioButton(
            mode_select_frame,
            text=self.i18n.t('modes.developer'),
            variable=self.batch_evaluation_mode,
            value="developer",
            font=("Arial", 14),
            text_color="#ffffff",
            fg_color="#ffa500",
            hover_color="#cc8400"
        )
        self.batch_mode_developer.pack(anchor="w", padx=30, pady=(0, 8))

        self.batch_mode_developer_desc = ctk.CTkLabel(
            mode_select_frame,
            text=f"  {self.i18n.t('modes.developer_desc')}",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.batch_mode_developer_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # === アコーディオン: フォルダ設定 ===
        self.folder_accordion = AccordionSection(
            self.batch_mode_frame,
            self.i18n.t('sections.folder_settings'),
            bg_color="#1e2740",
            title_color="#4A90E2",
            font_size=18
        )

        # PNG推奨の注意書き
        self.batch_png_warning = ctk.CTkLabel(
            self.folder_accordion.content_frame,
            text=self.i18n.t('warnings.png_recommended'),
            font=("Arial", 12, "bold"),
            text_color="#ff6b6b"
        )
        self.batch_png_warning.pack(anchor="w", padx=15, pady=(10, 5))

        # 元画像フォルダ
        self.batch_original_dir = tk.StringVar()
        self.batch_original_label = ctk.CTkLabel(
            self.folder_accordion.content_frame,
            text=self.i18n.t('batch.original_folder_label'),
            font=("Arial", 14, "bold"),
            text_color="#00ff88"
        )
        self.batch_original_label.pack(anchor="w", padx=15, pady=(5, 5))

        original_frame = ctk.CTkFrame(self.folder_accordion.content_frame, fg_color="transparent")
        original_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        original_entry = ctk.CTkEntry(
            original_frame,
            textvariable=self.batch_original_dir,
            placeholder_text="dataset/original/",
            height=45,
            font=("Arial", 13)
        )
        original_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.batch_original_browse_btn = ctk.CTkButton(
            original_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_batch_original,
            width=80,
            height=45,
            font=("Arial", 14),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.batch_original_browse_btn.pack(side=tk.RIGHT)

        # 超解像モデルフォルダ（複数）
        self.batch_upscaled_label = ctk.CTkLabel(
            self.folder_accordion.content_frame,
            text=self.i18n.t('batch.model_folder_label'),
            font=("Arial", 14, "bold"),
            text_color="#ffffff"
        )
        self.batch_upscaled_label.pack(anchor="w", padx=15, pady=(10, 5))

        # モデルフォルダ入力欄（5個）
        self.batch_model_vars = []
        self.batch_model_name_vars = []

        for i in range(5):
            model_frame = ctk.CTkFrame(self.folder_accordion.content_frame, fg_color="transparent")
            model_frame.pack(fill=tk.X, padx=15, pady=5)

            model_name_var = tk.StringVar(value=f"model{i+1}")
            model_path_var = tk.StringVar()

            self.batch_model_name_vars.append(model_name_var)
            self.batch_model_vars.append(model_path_var)

            # モデル名入力
            name_entry = ctk.CTkEntry(
                model_frame,
                textvariable=model_name_var,
                placeholder_text=self.i18n.t('batch.model_name_placeholder').format(num=i+1),
                width=140,
                height=40,
                font=("Arial", 12)
            )
            name_entry.pack(side=tk.LEFT, padx=(0, 5))

            # パス入力
            path_entry = ctk.CTkEntry(
                model_frame,
                textvariable=model_path_var,
                placeholder_text=f"dataset/upscayl_model{i+1}/",
                height=40,
                font=("Arial", 12)
            )
            path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

            # 参照ボタン
            browse_btn = ctk.CTkButton(
                model_frame,
                text="[FOLDER]",
                command=lambda idx=i: self.browse_batch_model(idx),
                width=50,
                height=40,
                font=("Arial", 13),
                fg_color="#555555",
                hover_color="#777777"
            )
            browse_btn.pack(side=tk.RIGHT)

        # === アコーディオン: 出力設定 ===
        self.output_csv_accordion = AccordionSection(
            self.batch_mode_frame,
            self.i18n.t('sections.csv_settings'),
            bg_color="#1e2740",
            title_color="#4A90E2",
            font_size=18
        )

        self.batch_output_csv = tk.StringVar(value=f"results/{get_timestamp_filename('batch_analysis', '.csv')}")
        self.batch_output_detail = tk.StringVar(value=f"results/detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}/")
        self.batch_limit = tk.IntVar(value=0)  # 0 = 全て
        self.batch_append_mode = tk.BooleanVar(value=True)  # True = 追加（デフォルト）, False = 上書き

        csv_frame = ctk.CTkFrame(self.output_csv_accordion.content_frame, fg_color="transparent")
        csv_frame.pack(fill=tk.X, padx=15, pady=5)

        csv_label = ctk.CTkLabel(csv_frame, text="CSV:", width=80, anchor="w", font=("Arial", 12))
        csv_label.pack(side=tk.LEFT)

        csv_entry = ctk.CTkEntry(
            csv_frame,
            textvariable=self.batch_output_csv,
            height=40,
            font=("Arial", 12)
        )
        csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        csv_browse_btn = ctk.CTkButton(
            csv_frame,
            text="[FOLDER]",
            command=self.browse_batch_csv_output,
            width=50,
            height=40,
            font=("Arial", 13),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        csv_browse_btn.pack(side=tk.RIGHT)

        detail_frame = ctk.CTkFrame(self.output_csv_accordion.content_frame, fg_color="transparent")
        detail_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        detail_label = ctk.CTkLabel(detail_frame, text=self.i18n.t('batch.detail_label'), width=80, anchor="w", font=("Arial", 12))
        detail_label.pack(side=tk.LEFT)

        detail_entry = ctk.CTkEntry(
            detail_frame,
            textvariable=self.batch_output_detail,
            height=40,
            font=("Arial", 12)
        )
        detail_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        detail_browse_btn = ctk.CTkButton(
            detail_frame,
            text="[FOLDER]",
            command=self.browse_batch_detail_output,
            width=50,
            height=40,
            font=("Arial", 13),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        detail_browse_btn.pack(side=tk.RIGHT)

        # 追加モード選択チェックボックス
        append_mode_frame = ctk.CTkFrame(self.output_csv_accordion.content_frame, fg_color="transparent")
        append_mode_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        append_checkbox = ctk.CTkCheckBox(
            append_mode_frame,
            text=self.i18n.t('batch.append_mode_label'),
            variable=self.batch_append_mode,
            font=("Arial", 13),
            text_color="#4A90E2",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        append_checkbox.pack(anchor="w")

        # === アコーディオン: 詳細設定 ===
        self.detail_accordion = AccordionSection(
            self.batch_mode_frame,
            self.i18n.t('sections.detail_output'),
            bg_color="#1e2740",
            title_color="#4A90E2",
            font_size=18
        )

        limit_info = ctk.CTkLabel(
            self.detail_accordion.content_frame,
            text=self.i18n.t('batch.limit_note'),
            font=("Arial", 11),
            text_color="#888888",
            justify="left"
        )
        limit_info.pack(anchor="w", padx=15, pady=(10, 5))

        # 処理枚数制限フレーム（縦に2段構成）
        limit_container = ctk.CTkFrame(self.detail_accordion.content_frame, fg_color="transparent")
        limit_container.pack(fill=tk.X, padx=15, pady=(0, 15))

        # タイトル
        limit_title = ctk.CTkLabel(
            limit_container,
            text=self.i18n.t('batch.limit_label'),
            font=("Arial", 13, "bold"),
            text_color="#4A90E2",
            anchor="w"
        )
        limit_title.pack(fill=tk.X, pady=(0, 8))

        # 第1段：スライダー + 現在値表示
        slider_frame = ctk.CTkFrame(limit_container, fg_color="transparent")
        slider_frame.pack(fill=tk.X, pady=(0, 8))

        self.limit_slider = ctk.CTkSlider(
            slider_frame,
            from_=0,
            to=500,
            number_of_steps=50,
            width=280,
            command=self.on_slider_change,
            fg_color="#2d3748",
            progress_color="#00ffff",
            button_color="#00ffff",
            button_hover_color="#00cccc"
        )
        self.limit_slider.pack(side=tk.LEFT, padx=(0, 15))
        self.limit_slider.set(0)

        self.limit_value_label = ctk.CTkLabel(
            slider_frame,
            text=self.i18n.t('batch.limit_all'),
            font=("Arial", 15, "bold"),
            text_color="#00ff88",
            width=90
        )
        self.limit_value_label.pack(side=tk.LEFT)

        # 第2段：直接入力
        entry_frame = ctk.CTkFrame(limit_container, fg_color="transparent")
        entry_frame.pack(fill=tk.X)

        entry_label = ctk.CTkLabel(
            entry_frame,
            text=self.i18n.t('batch.limit_direct_input'),
            font=("Arial", 12),
            text_color="#888888",
            anchor="w"
        )
        entry_label.pack(side=tk.LEFT, padx=(0, 10))

        self.limit_entry = ctk.CTkEntry(
            entry_frame,
            width=140,
            height=40,
            font=("Arial", 14),
            placeholder_text=self.i18n.t('batch.limit_placeholder'),
            fg_color="#1e2740",
            border_color="#00ffff",
            text_color="#ffffff"
        )
        self.limit_entry.pack(side=tk.LEFT)
        self.limit_entry.insert(0, "0")
        self.limit_entry.bind("<Return>", self.on_entry_change)
        self.limit_entry.bind("<FocusOut>", self.on_entry_change)
        self.limit_entry.bind("<KeyRelease>", self.on_entry_typing)

        # 値変更時のコールバック
        self.batch_limit.trace_add("write", self.update_limit_label)

        # === 並列処理設定 ===
        parallel_info = ctk.CTkLabel(
            self.detail_accordion.content_frame,
            text=self.i18n.t('batch.parallel_info'),
            font=("Arial", 11),
            text_color="#888888",
            justify="left"
        )
        parallel_info.pack(anchor="w", padx=15, pady=(20, 5))

        # 並列処理フレーム
        parallel_frame = ctk.CTkFrame(self.detail_accordion.content_frame, fg_color="transparent")
        parallel_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 並列処理ON/OFFチェックボックス
        self.use_parallel = tk.BooleanVar(value=False)  # デフォルトOFF
        parallel_checkbox = ctk.CTkCheckBox(
            parallel_frame,
            text=self.i18n.t('batch.use_parallel'),
            variable=self.use_parallel,
            command=self.toggle_parallel_settings,
            font=("Arial", 13),
            text_color="#4A90E2",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        parallel_checkbox.pack(anchor="w", pady=(0, 10))

        # プロセス数設定フレーム
        workers_frame = ctk.CTkFrame(parallel_frame, fg_color="transparent")
        workers_frame.pack(fill=tk.X)

        workers_label = ctk.CTkLabel(
            workers_frame,
            text=self.i18n.t('batch.num_workers'),
            font=("Arial", 12),
            text_color="#888888",
            anchor="w"
        )
        workers_label.pack(side=tk.LEFT, padx=(20, 10))

        from multiprocessing import cpu_count
        max_workers = max(1, cpu_count())
        self.num_workers = tk.IntVar(value=max(1, cpu_count() - 1))

        self.workers_spinbox = ctk.CTkEntry(
            workers_frame,
            width=80,
            height=35,
            font=("Arial", 13),
            textvariable=self.num_workers,
            state='disabled',  # デフォルトで無効
            fg_color="#2d3748",
            border_color="#4A90E2",
            text_color="#ffffff"
        )
        self.workers_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        workers_info = ctk.CTkLabel(
            workers_frame,
            text=self.i18n.t('batch.workers_hint').format(recommended=max(1, cpu_count() - 1), max=max_workers),
            font=("Arial", 11),
            text_color="#666666"
        )
        workers_info.pack(side=tk.LEFT)

        # P6パッチサイズ選択
        self.batch_patch_info = ctk.CTkLabel(
            self.detail_accordion.content_frame,
            text=self.i18n.t('batch.patch_size_title'),
            font=("Arial", 13, "bold"),
            text_color="#4A90E2",
            justify="left"
        )
        self.batch_patch_info.pack(anchor="w", padx=15, pady=(20, 5))

        self.batch_patch_desc = ctk.CTkLabel(
            self.detail_accordion.content_frame,
            text=self.i18n.t('batch.patch_size_desc'),
            font=("Arial", 11),
            text_color="#888888",
            justify="left"
        )
        self.batch_patch_desc.pack(anchor="w", padx=15, pady=(0, 5))

        # パッチサイズ変数（デフォルト16）
        self.batch_patch_size = tk.IntVar(value=16)

        patch_frame = ctk.CTkFrame(self.detail_accordion.content_frame, fg_color="transparent")
        patch_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        # 8×8オプション
        self.batch_patch_8 = ctk.CTkRadioButton(
            patch_frame,
            text=self.i18n.t('batch.patch_8x8'),
            variable=self.batch_patch_size,
            value=8,
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#ff6b6b",
            hover_color="#ee5555"
        )
        self.batch_patch_8.pack(anchor="w", pady=(0, 5))

        # 16×16オプション（推奨）
        self.batch_patch_16 = ctk.CTkRadioButton(
            patch_frame,
            text=self.i18n.t('batch.patch_16x16'),
            variable=self.batch_patch_size,
            value=16,
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        self.batch_patch_16.pack(anchor="w", pady=(0, 5))

        # 32×32オプション
        self.batch_patch_32 = ctk.CTkRadioButton(
            patch_frame,
            text=self.i18n.t('batch.patch_32x32'),
            variable=self.batch_patch_size,
            value=32,
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#4ecdc4",
            hover_color="#3db8af"
        )
        self.batch_patch_32.pack(anchor="w")

        # === 通常のバッチ処理セクション ===
        # 実行ボタン（翻訳対応）
        self.batch_analyze_btn = ctk.CTkButton(
            self.batch_mode_frame,
            text=f"[RUN] {self.i18n.t('buttons.analyze_batch')}",
            command=self.start_batch_analysis,
            height=60,
            corner_radius=10,
            font=("Arial", 18, "bold"),
            fg_color="#00ff88",
            text_color="#000000",
            hover_color="#00dd77"
        )
        self.batch_analyze_btn.pack(fill=tk.X, pady=(0, 15))

        # === アコーディオン: 統計分析 ===
        self.stats_accordion = AccordionSection(
            self.batch_mode_frame,
            self.i18n.t('sections.stats_analysis'),
            bg_color="#1e2740",
            title_color="#4A90E2",
            font_size=18
        )

        stats_info = ctk.CTkLabel(
            self.stats_accordion.content_frame,
            text=self.i18n.t('batch.stats_desc'),
            font=("Arial", 13),
            text_color="#cccccc",
            justify="left"
        )
        stats_info.pack(anchor="w", padx=15, pady=(10, 10))

        # CSV選択
        self.stats_csv_path = tk.StringVar()

        csv_select_frame = ctk.CTkFrame(self.stats_accordion.content_frame, fg_color="transparent")
        csv_select_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        csv_select_entry = ctk.CTkEntry(
            csv_select_frame,
            textvariable=self.stats_csv_path,
            placeholder_text=self.i18n.t('batch.csv_placeholder'),
            height=45,
            font=("Arial", 13)
        )
        csv_select_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        csv_select_btn = ctk.CTkButton(
            csv_select_frame,
            text=self.i18n.t('batch.csv_select_button'),
            command=self.browse_stats_csv,
            width=120,
            height=45,
            font=("Arial", 14),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        csv_select_btn.pack(side=tk.RIGHT)

        # ボタン配置用フレーム
        button_frame = ctk.CTkFrame(self.stats_accordion.content_frame, fg_color="transparent")
        button_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 統計分析実行ボタン
        self.stats_analyze_btn = ctk.CTkButton(
            button_frame,
            text=f"[ANALYZE] {self.i18n.t('buttons.analyze_stats')}",
            command=self.start_stats_analysis,
            height=55,
            corner_radius=10,
            font=("Arial", 15, "bold"),
            fg_color="#ffa500",
            text_color="#000000",
            hover_color="#cc8400"
        )
        self.stats_analyze_btn.pack(fill=tk.X, pady=(0, 10))

        # ログエクスポートボタン
        self.export_log_btn = ctk.CTkButton(
            button_frame,
            text="[EXPORT] ログをファイル保存",
            command=self.export_current_log,
            height=40,
            corner_radius=10,
            font=("Arial", 13, "bold"),
            fg_color="#2d7d46",
            text_color="#ffffff",
            hover_color="#236138"
        )
        self.export_log_btn.pack(fill=tk.X, pady=(0, 10))

        # ハルシネーション抽出ボタン
        self.hallucination_extract_btn = ctk.CTkButton(
            button_frame,
            text=self.i18n.t('batch.hallucination_warning'),
            command=self.extract_hallucination_suspects,
            height=45,
            corner_radius=10,
            font=("Arial", 13, "bold"),
            fg_color="#ff4444",
            text_color="#ffffff",
            hover_color="#cc3333"
        )
        self.hallucination_extract_btn.pack(fill=tk.X, pady=(0, 5))

        # クリーンデータセット抽出ボタン（NEW in v1.5）
        self.clean_dataset_btn = ctk.CTkButton(
            button_frame,
            text=self.i18n.t('batch.normal_data_extract'),
            command=self.extract_clean_dataset,
            height=45,
            corner_radius=10,
            font=("Arial", 13, "bold"),
            fg_color="#44ff44",
            text_color="#000000",
            hover_color="#33cc33"
        )
        self.clean_dataset_btn.pack(fill=tk.X)

        # 結果表示エリア
        self.batch_result_text = ctk.CTkTextbox(
            self.batch_mode_frame,
            font=("Meiryo", 12),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10,
            height=200
        )
        self.batch_result_text.pack(fill=tk.BOTH, expand=True)


    def browse_batch_original(self):
        dirname = filedialog.askdirectory(title=self.i18n.t('batch.select_original_folder'))
        if dirname:
            self.batch_original_dir.set(dirname)

    def browse_batch_model(self, index):
        dirname = filedialog.askdirectory(title=self.i18n.t('batch.select_model_folder').format(num=index+1))
        if dirname:
            self.batch_model_vars[index].set(dirname)

    def browse_batch_csv_output(self):
        """CSV出力先選択"""
        filename = filedialog.asksaveasfilename(
            title=self.i18n.t('batch.select_csv_output'),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("すべてのファイル", "*.*")],
            initialfile=get_timestamp_filename("batch_analysis", ".csv")
        )
        if filename:
            self.batch_output_csv.set(filename)

    def browse_batch_detail_output(self):
        """詳細レポート出力先フォルダ選択"""
        dirname = filedialog.askdirectory(title=self.i18n.t('batch.select_detail_folder'))
        if dirname:
            self.batch_output_detail.set(dirname)


    def run_batch_bicubic_downscale(self):
        """バッチBicubic縮小を実行"""
        import cv2
        import os
        from tkinter import messagebox
        import glob

        # 入力フォルダ確認
        input_dir = self.academic_input_dir.get()
        if not input_dir:
            messagebox.showerror("エラー", "入力フォルダを選択してください")
            return

        if not os.path.exists(input_dir):
            messagebox.showerror("エラー", f"入力フォルダが見つかりません:\n{input_dir}")
            return

        # 出力フォルダ確認
        output_dir = self.academic_output_dir.get()
        if not output_dir:
            messagebox.showerror("エラー", "出力フォルダを選択してください")
            return

        # 出力フォルダ作成
        os.makedirs(output_dir, exist_ok=True)

        # 縮小倍率確認
        try:
            scale = float(self.academic_scale.get())
            if scale <= 0 or scale >= 1:
                messagebox.showerror("エラー", "縮小倍率は0より大きく1未満の値を指定してください")
                return
        except ValueError:
            messagebox.showerror("エラー", "縮小倍率は数値で指定してください")
            return

        # 画像ファイル一覧を取得
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if len(image_files) == 0:
            messagebox.showerror("エラー", f"入力フォルダに画像ファイルが見つかりません:\n{input_dir}")
            return

        # 確認ダイアログ
        result = messagebox.askyesno(
            "確認",
            f"以下の設定でバッチBicubic縮小を実行します:\n\n"
            f"入力フォルダ: {input_dir}\n"
            f"出力フォルダ: {output_dir}\n"
            f"画像数: {len(image_files)}枚\n"
            f"縮小倍率: ×{scale} (例: 1000px → {int(1000*scale)}px)\n\n"
            f"実行しますか？"
        )

        if not result:
            return

        # バッチ処理実行
        success_count = 0
        error_count = 0
        error_files = []

        try:
            for i, img_path in enumerate(image_files, 1):
                try:
                    # 画像読み込み（ビット深度・カラー形式を保持）
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        error_count += 1
                        error_files.append(os.path.basename(img_path))
                        continue

                    h, w = img.shape[:2]

                    # Bicubic縮小
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img_lr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                    # 出力パス生成
                    base_name = os.path.basename(img_path)
                    name, ext = os.path.splitext(base_name)
                    output_path = os.path.join(output_dir, f"{name}_LR_bicubic_x{scale:.2f}.png")

                    # 保存
                    cv2.imwrite(output_path, img_lr)
                    success_count += 1

                    # 進捗表示（10枚ごと）
                    if i % 10 == 0 or i == len(image_files):
                        print(f"進捗: {i}/{len(image_files)} 枚完了")

                except Exception as e:
                    error_count += 1
                    error_files.append(f"{os.path.basename(img_path)}: {str(e)}")
                    continue

            # 完了メッセージ
            message = f"[OK] バッチBicubic縮小が完了しました\n\n"
            message += f"成功: {success_count}枚\n"
            if error_count > 0:
                message += f"エラー: {error_count}枚\n\n"
                message += "エラーファイル:\n"
                for err_file in error_files[:10]:  # 最大10件表示
                    message += f"  - {err_file}\n"
                if len(error_files) > 10:
                    message += f"  ... 他{len(error_files)-10}件\n"

            message += f"\n出力先:\n{output_dir}\n\n"
            message += f"次のステップ:\n"
            message += f"1. この低解像度画像を外部ツールでAI超解像\n"
            message += f"2. 超解像結果をバッチ処理で評価\n"
            message += f"3. 評価モードを「学術評価モード」に設定"

            messagebox.showinfo("完了", message)

        except Exception as e:
            messagebox.showerror("エラー", f"バッチ処理中にエラーが発生しました:\n{str(e)}")


    def toggle_parallel_settings(self):
        """並列処理ON/OFF時の処理（バッチ処理タブ）"""
        if self.use_parallel.get():
            # 並列処理ON: プロセス数入力を有効化
            self.workers_spinbox.configure(state='normal')
        else:
            # 並列処理OFF: プロセス数入力を無効化
            self.workers_spinbox.configure(state='disabled')


    def start_batch_analysis(self):
        """バッチ処理開始"""
        # バリデーション：元画像フォルダ（必須）
        if not self.batch_original_dir.get():
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('batch.error_no_original_folder')
            )
            return

        if not os.path.exists(self.batch_original_dir.get()):
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('batch.error_original_folder_not_found').format(path=self.batch_original_dir.get())
            )
            return

        # バリデーション：有効なモデルフォルダをカウント
        valid_models = {}
        for i in range(5):
            model_name = self.batch_model_name_vars[i].get().strip()
            model_path = self.batch_model_vars[i].get().strip()

            if model_path:
                # モデル名が空の場合
                if not model_name:
                    messagebox.showerror(
                        self.i18n.t('messages.error'),
                        self.i18n.t('batch.error_no_model_name').format(num=i+1)
                    )
                    return
                # フォルダが存在しない場合
                if not os.path.exists(model_path):
                    messagebox.showerror(
                        self.i18n.t('messages.error'),
                        self.i18n.t('batch.error_model_not_found').format(num=i+1, path=model_path)
                    )
                    return
                valid_models[model_name] = model_path

        # 最低1つは必須（画像1に相当）
        if len(valid_models) == 0:
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('batch.error_no_models')
            )
            return

        # 設定ファイル作成
        from multiprocessing import cpu_count

        # 並列処理数の決定
        if self.use_parallel.get():
            # 並列処理ON: ユーザー指定のプロセス数を使用
            num_workers = max(1, min(self.num_workers.get(), cpu_count()))
        else:
            # 並列処理OFF: 1プロセス（並列なし）
            num_workers = 1

        config = {
            "original_dir": self.batch_original_dir.get(),
            "upscaled_dirs": valid_models,
            "output_csv": self.batch_output_csv.get(),
            "output_detail_dir": self.batch_output_detail.get(),
            "limit": self.batch_limit.get(),  # 処理枚数制限
            "append_mode": self.batch_append_mode.get(),  # 追加モード
            "evaluation_mode": self.batch_evaluation_mode.get(),  # 評価モード（バッチ処理タブの設定）
            "num_workers": num_workers,  # 並列処理数（ユーザー設定）
            "checkpoint_interval": 1000,  # チェックポイント間隔（1000サンプルごと）
            "patch_size": self.batch_patch_size.get()  # P6ヒートマップのパッチサイズ
        }

        # UIを無効化
        self.batch_analyze_btn.configure(state='disabled')
        self.batch_progress.set(0)
        self.batch_status_label.configure(text=self.i18n.t('batch.status_starting'), text_color="#00ffff")
        self.batch_result_text.delete("1.0", tk.END)

        # 別スレッドで実行
        thread = threading.Thread(target=self.run_batch_analysis, args=(config,))
        thread.daemon = True
        thread.start()

    def update_batch_progress(self, current, total, message):
        """バッチ処理進捗更新（別スレッドから呼ばれる）"""
        progress = current / total if total > 0 else 0
        self.root.after(0, lambda: self.batch_progress.set(progress))
        progress_text = self.i18n.t('batch.status_processing').format(current=current, total=total, message=message)
        self.root.after(0, lambda: self.batch_status_label.configure(
            text=progress_text,
            text_color="#4A90E2"
        ))
        self.root.after(0, lambda: self.batch_result_text.insert(tk.END, f"{message}\n"))
        self.root.after(0, lambda: self.batch_result_text.see(tk.END))

    def run_batch_analysis(self, config):
        """バッチ処理実行"""
        try:
            import sys
            from io import StringIO
            from batch_analyzer import batch_analyze
            from pathlib import Path

            # 一時設定ファイル作成
            temp_config_path = "temp_batch_config.json"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # バッチ処理実行（進捗コールバック付き）
            batch_analyze(temp_config_path, progress_callback=self.update_batch_progress)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # 一時ファイル削除
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            self.root.after(0, self.display_batch_results, output, True, config['output_csv'])

        except Exception as e:
            sys.stdout = old_stdout
            self.root.after(0, self.display_batch_results, str(e), False, None)

    def display_batch_results(self, output, success, csv_path):
        """バッチ処理結果表示"""
        self.batch_analyze_btn.configure(state='normal')
        self.batch_progress.set(1 if success else 0)

        self.batch_result_text.insert("1.0", output)

        if success:
            self.batch_status_label.configure(
                text=self.i18n.t('batch.status_complete').format(path=csv_path),
                text_color="#00ff88"
            )

            # CSVパスを統計分析欄に自動入力
            if csv_path:
                self.stats_csv_path.set(csv_path)

            messagebox.showinfo(
                self.i18n.t('messages.completed'),
                self.i18n.t('batch.status_complete').format(path=csv_path)
            )
        else:
            self.batch_status_label.configure(
                text=self.i18n.t('batch.status_error'),
                text_color="#ff4444"
            )
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('batch.error_batch_failed').format(output=output)
            )


    def on_slider_change(self, value):
        """スライダー変更時のコールバック"""
        int_value = int(value)
        self.batch_limit.set(int_value)

        # 数値入力フィールドも更新
        self.limit_entry.delete(0, tk.END)
        self.limit_entry.insert(0, str(int_value))

        # ラベル更新
        if int_value == 0:
            self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_all'), text_color="#00ff88")
        else:
            self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_count').format(count=int_value), text_color="#00ffff")

    def on_entry_typing(self, event=None):
        """入力中のリアルタイムフィードバック"""
        try:
            value = self.limit_entry.get().strip()
            if value == "" or value == "0":
                self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_all'), text_color="#00ff88")
            else:
                int_value = int(value)
                if int_value > 0:
                    self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_count').format(count=int_value), text_color="#00ffff")
        except ValueError:
            pass  # 入力中は無視

    def on_entry_change(self, event=None):
        """数値入力フィールド確定時のコールバック（Enter or フォーカスアウト）"""
        try:
            value = self.limit_entry.get().strip()
            if value == "":
                int_value = 0
            else:
                int_value = int(value)

            # 負の値は0にする
            if int_value < 0:
                int_value = 0
                self.limit_entry.delete(0, tk.END)
                self.limit_entry.insert(0, "0")

            self.batch_limit.set(int_value)

            # スライダーも更新（500以下の場合のみ）
            if int_value <= 500:
                self.limit_slider.set(int_value)

            # ラベル更新
            if int_value == 0:
                self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_all'), text_color="#00ff88")
            else:
                self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_count').format(count=int_value), text_color="#00ffff")

        except ValueError:
            # 無効な入力の場合は0にリセット
            self.limit_entry.delete(0, tk.END)
            self.limit_entry.insert(0, "0")
            self.batch_limit.set(0)
            self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_all'), text_color="#00ff88")

    def update_limit_label(self, *args):
        """処理枚数ラベル更新（trace用）"""
        limit = self.batch_limit.get()
        if limit == 0:
            self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_all'), text_color="#00ff88")
        else:
            self.limit_value_label.configure(text=self.i18n.t('batch.limit_display_count').format(count=limit), text_color="#00ffff")
