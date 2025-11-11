import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
import sys
import json
from datetime import datetime
from ui_components import AccordionSection, get_timestamp_filename
import customtkinter as ctk
from PIL import Image

class AcademicModeMixin:
    """学術モード機能を提供するMixinクラス"""

    def create_academic_mode_ui(self):
        """論文用ベンチマーク評価モードのUI作成（左パネル）"""

        # 説明セクション
        info_frame = ctk.CTkFrame(self.academic_mode_frame, fg_color="#2d1b4e", corner_radius=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))

        self.academic_info_title = ctk.CTkLabel(
            info_frame,
            text=self.i18n.t('academic.title'),
            font=("Arial", 18, "bold"),
            text_color="#9b59b6"
        )
        self.academic_info_title.pack(anchor="w", padx=15, pady=(15, 5))

        self.academic_info_text = ctk.CTkLabel(
            info_frame,
            text=self.i18n.t('academic.description'),
            font=("Arial", 13),
            text_color="#cccccc",
            justify="left"
        )
        self.academic_info_text.pack(anchor="w", padx=15, pady=(0, 15))

        # === アコーディオン: 処理フロー ===
        self.workflow_accordion = AccordionSection(
            self.academic_mode_frame,
            self.i18n.t('help.workflow_title'),
            bg_color="#1e2740",
            title_color="#4A90E2",
            font_size=18
        )

        self.workflow_text = ctk.CTkLabel(
            self.workflow_accordion.content_frame,
            text=self.i18n.t('academic.workflow_steps'),
            font=("Arial", 13),
            text_color="#cccccc",
            justify="left"
        )
        self.workflow_text.pack(anchor="w", padx=15, pady=(10, 15))

        # === アコーディオン: Step 0（デフォルト閉） ===
        self.bicubic_accordion = AccordionSection(
            self.academic_mode_frame,
            self.i18n.t('sections.bicubic_info'),
            bg_color="#2d1b3d",
            title_color="#9b59b6",
            font_size=18
        )
        self.bicubic_accordion.is_open = False
        self.bicubic_accordion.content_frame.pack_forget()
        self.bicubic_accordion.header_btn.configure(text=f"▶ {self.bicubic_accordion.title}")

        bicubic_desc = ctk.CTkLabel(
            self.bicubic_accordion.content_frame,
            text=self.i18n.t('academic.bicubic_desc'),
            font=("Arial", 12),
            text_color="#888888",
            justify="left"
        )
        bicubic_desc.pack(anchor="w", padx=15, pady=(10, 10))

        # 入力フォルダ
        input_folder_label = ctk.CTkLabel(
            self.bicubic_accordion.content_frame,
            text=self.i18n.t('academic.input_folder_label'),
            font=("Arial", 13),
            text_color="#cccccc"
        )
        input_folder_label.pack(anchor="w", padx=15, pady=(5, 5))

        input_folder_frame = ctk.CTkFrame(self.bicubic_accordion.content_frame, fg_color="transparent")
        input_folder_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.academic_input_dir = tk.StringVar()
        input_entry = ctk.CTkEntry(
            input_folder_frame,
            textvariable=self.academic_input_dir,
            placeholder_text=self.i18n.t('academic.placeholder_gt_folder'),
            height=45,
            font=("Arial", 13)
        )
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        input_btn = ctk.CTkButton(
            input_folder_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_academic_input,
            width=90,
            height=45,
            font=("Arial", 14),
            fg_color="#9b59b6",
            hover_color="#7d3c98"
        )
        input_btn.pack(side=tk.RIGHT)

        # 出力フォルダ
        output_folder_label = ctk.CTkLabel(
            self.bicubic_accordion.content_frame,
            text=self.i18n.t('academic.output_folder_label'),
            font=("Arial", 13),
            text_color="#cccccc"
        )
        output_folder_label.pack(anchor="w", padx=15, pady=(5, 5))

        output_folder_frame = ctk.CTkFrame(self.bicubic_accordion.content_frame, fg_color="transparent")
        output_folder_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.academic_output_dir = tk.StringVar()
        output_entry = ctk.CTkEntry(
            output_folder_frame,
            textvariable=self.academic_output_dir,
            placeholder_text=self.i18n.t('academic.placeholder_lr_folder'),
            height=45,
            font=("Arial", 13)
        )
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        output_btn = ctk.CTkButton(
            output_folder_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_academic_output,
            width=90,
            height=45,
            font=("Arial", 14),
            fg_color="#9b59b6",
            hover_color="#7d3c98"
        )
        output_btn.pack(side=tk.RIGHT)

        # 縮小倍率
        scale_label = ctk.CTkLabel(
            self.bicubic_accordion.content_frame,
            text=self.i18n.t('academic.scale_label'),
            font=("Arial", 13),
            text_color="#cccccc"
        )
        scale_label.pack(anchor="w", padx=15, pady=(5, 5))

        scale_frame = ctk.CTkFrame(self.bicubic_accordion.content_frame, fg_color="transparent")
        scale_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.academic_scale = tk.StringVar(value="0.5")
        scale_entry = ctk.CTkEntry(
            scale_frame,
            textvariable=self.academic_scale,
            width=120,
            height=40,
            font=("Arial", 13)
        )
        scale_entry.pack(side=tk.LEFT, padx=(0, 10))

        scale_note = ctk.CTkLabel(
            scale_frame,
            text=self.i18n.t('academic.scale_note'),
            font=("Arial", 11),
            text_color="#888888"
        )
        scale_note.pack(side=tk.LEFT)

        # 実行ボタン
        bicubic_btn = ctk.CTkButton(
            self.bicubic_accordion.content_frame,
            text=self.i18n.t('academic.run_bicubic'),
            command=self.run_batch_bicubic_downscale,
            height=50,
            corner_radius=10,
            font=("Arial", 15, "bold"),
            fg_color="#9b59b6",
            text_color="#ffffff",
            hover_color="#7d3c98"
        )
        bicubic_btn.pack(fill=tk.X, padx=15, pady=(5, 15))

        # === アコーディオン: 評価設定 ===
        self.config_accordion = AccordionSection(
            self.academic_mode_frame,
            self.i18n.t('sections.evaluation_settings'),
            bg_color="#1e2740",
            title_color="#9b59b6",
            font_size=18
        )

        # 評価モード固定表示
        self.mode_info = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.mode_info'),
            font=("Arial", 14, "bold"),
            text_color="#9b59b6"
        )
        self.mode_info.pack(anchor="w", padx=15, pady=(10, 15))

        # PNG推奨の注意書き
        self.academic_png_warning = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('warnings.png_recommended'),
            font=("Arial", 12, "bold"),
            text_color="#ff6b6b"
        )
        self.academic_png_warning.pack(anchor="w", padx=15, pady=(0, 10))

        # 元画像フォルダ
        self.academic_original_dir = tk.StringVar()
        original_label = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.original_folder_required'),
            font=("Arial", 14, "bold"),
            text_color="#00ff88"
        )
        original_label.pack(anchor="w", padx=15, pady=(5, 5))

        original_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        original_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        original_entry = ctk.CTkEntry(
            original_frame,
            textvariable=self.academic_original_dir,
            placeholder_text=self.i18n.t('academic.placeholder_original'),
            height=45,
            font=("Arial", 13)
        )
        original_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        original_btn = ctk.CTkButton(
            original_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_academic_original,
            width=90,
            height=45,
            font=("Arial", 14),
            fg_color="#9b59b6",
            text_color="#FFFFFF",
            hover_color="#7d3c98"
        )
        original_btn.pack(side=tk.RIGHT)

        # 超解像モデルフォルダ（最大5つ）
        models_label = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.model_folder_label'),
            font=("Arial", 14, "bold"),
            text_color="#ffffff"
        )
        models_label.pack(anchor="w", padx=15, pady=(10, 5))

        self.academic_model_vars = []
        self.academic_model_name_vars = []

        for i in range(5):
            model_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
            model_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

            name_var = tk.StringVar()
            self.academic_model_name_vars.append(name_var)

            name_entry = ctk.CTkEntry(
                model_frame,
                textvariable=name_var,
                placeholder_text=self.i18n.t('academic.model_name_placeholder').format(num=i+1),
                width=140,
                height=40,
                font=("Arial", 12)
            )
            name_entry.pack(side=tk.LEFT, padx=(0, 10))

            path_var = tk.StringVar()
            self.academic_model_vars.append(path_var)

            path_entry = ctk.CTkEntry(
                model_frame,
                textvariable=path_var,
                placeholder_text=self.i18n.t('academic.model_path_placeholder').format(num=i+1),
                height=40,
                font=("Arial", 12)
            )
            path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

            browse_btn = ctk.CTkButton(
                model_frame,
                text=self.i18n.t('buttons.browse'),
                command=lambda idx=i: self.browse_academic_model(idx),
                width=90,
                height=40,
                font=("Arial", 13),
                fg_color="#9b59b6",
                text_color="#FFFFFF",
                hover_color="#7d3c98"
            )
            browse_btn.pack(side=tk.RIGHT)

        # 出力設定
        self.output_label = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.save_settings'),
            font=("Arial", 14, "bold"),
            text_color="#ffffff"
        )
        self.output_label.pack(anchor="w", padx=15, pady=(15, 5))

        # CSV出力パス
        csv_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        csv_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        csv_label = ctk.CTkLabel(csv_frame, text=self.i18n.t('academic.csv_label'), width=80, anchor="w", font=("Arial", 13))
        csv_label.pack(side=tk.LEFT, padx=(0, 10))

        self.academic_output_csv = tk.StringVar(value=get_timestamp_filename("batch_results_academic", ".csv"))
        csv_entry = ctk.CTkEntry(
            csv_frame,
            textvariable=self.academic_output_csv,
            height=40,
            font=("Arial", 12)
        )
        csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        csv_btn = ctk.CTkButton(
            csv_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_academic_csv_output,
            width=90,
            height=40,
            font=("Arial", 13),
            fg_color="#9b59b6",
            text_color="#FFFFFF",
            hover_color="#7d3c98"
        )
        csv_btn.pack(side=tk.RIGHT)

        # 詳細出力フォルダ
        detail_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        detail_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        detail_label = ctk.CTkLabel(detail_frame, text=self.i18n.t('academic.detail_label'), width=80, anchor="w", font=("Arial", 13))
        detail_label.pack(side=tk.LEFT, padx=(0, 10))

        self.academic_output_detail = tk.StringVar(value=f"batch_results_detail_academic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        detail_entry = ctk.CTkEntry(
            detail_frame,
            textvariable=self.academic_output_detail,
            height=40,
            font=("Arial", 12)
        )
        detail_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        detail_btn = ctk.CTkButton(
            detail_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_academic_detail_output,
            width=90,
            height=40,
            font=("Arial", 13),
            fg_color="#9b59b6",
            text_color="#FFFFFF",
            hover_color="#7d3c98"
        )
        detail_btn.pack(side=tk.RIGHT)

        # 処理枚数制限
        limit_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        limit_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.limit_label = ctk.CTkLabel(
            limit_frame,
            text=self.i18n.t('academic.processing_limit'),
            width=100,
            anchor="w",
            font=("Arial", 13)
        )
        self.limit_label.pack(side=tk.LEFT, padx=(0, 10))

        self.academic_limit = tk.IntVar(value=0)
        limit_entry = ctk.CTkEntry(
            limit_frame,
            textvariable=self.academic_limit,
            width=120,
            height=40,
            font=("Arial", 13)
        )
        limit_entry.pack(side=tk.LEFT)

        limit_hint = ctk.CTkLabel(
            limit_frame,
            text=self.i18n.t('academic.limit_hint'),
            font=("Arial", 12),
            text_color="#888888"
        )
        limit_hint.pack(side=tk.LEFT, padx=(10, 0))

        # 追加モード
        append_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        append_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        self.academic_append_mode = tk.BooleanVar(value=False)
        append_check = ctk.CTkCheckBox(
            append_frame,
            text=self.i18n.t('academic.append_mode_label'),
            variable=self.academic_append_mode,
            font=("Arial", 13),
            text_color="#ffffff",
            fg_color="#9b59b6",
            hover_color="#7d3c98"
        )
        append_check.pack(anchor="w")

        # === 並列処理設定（論文用） ===
        parallel_info = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.parallel_info'),
            font=("Arial", 11),
            text_color="#888888",
            justify="left"
        )
        parallel_info.pack(anchor="w", padx=15, pady=(20, 5))

        parallel_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        parallel_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 並列処理ON/OFF
        self.academic_use_parallel = tk.BooleanVar(value=False)  # デフォルトOFF
        academic_parallel_checkbox = ctk.CTkCheckBox(
            parallel_frame,
            text=self.i18n.t('academic.use_parallel'),
            variable=self.academic_use_parallel,
            command=self.toggle_academic_parallel_settings,
            font=("Arial", 13),
            text_color="#9b59b6",
            fg_color="#9b59b6",
            hover_color="#7d3c98"
        )
        academic_parallel_checkbox.pack(anchor="w", pady=(0, 10))

        # プロセス数設定
        academic_workers_frame = ctk.CTkFrame(parallel_frame, fg_color="transparent")
        academic_workers_frame.pack(fill=tk.X)

        academic_workers_label = ctk.CTkLabel(
            academic_workers_frame,
            text=self.i18n.t('academic.num_workers'),
            font=("Arial", 12),
            text_color="#888888",
            anchor="w"
        )
        academic_workers_label.pack(side=tk.LEFT, padx=(20, 10))

        from multiprocessing import cpu_count
        max_workers = max(1, cpu_count())
        self.academic_num_workers = tk.IntVar(value=max(1, cpu_count() - 1))

        self.academic_workers_spinbox = ctk.CTkEntry(
            academic_workers_frame,
            width=80,
            height=35,
            font=("Arial", 13),
            textvariable=self.academic_num_workers,
            state='disabled',  # デフォルトで無効
            fg_color="#2d3748",
            border_color="#9b59b6",
            text_color="#ffffff"
        )
        self.academic_workers_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        academic_workers_info = ctk.CTkLabel(
            academic_workers_frame,
            text=self.i18n.t('academic.workers_hint').format(
                recommended=max(1, cpu_count() - 1),
                max=max_workers
            ),
            font=("Arial", 11),
            text_color="#666666"
        )
        academic_workers_info.pack(side=tk.LEFT)

        # P6パッチサイズ選択
        self.academic_patch_info = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.patch_size_title'),
            font=("Arial", 13, "bold"),
            text_color="#9b59b6",
            justify="left"
        )
        self.academic_patch_info.pack(anchor="w", padx=15, pady=(20, 5))

        self.academic_patch_desc = ctk.CTkLabel(
            self.config_accordion.content_frame,
            text=self.i18n.t('academic.patch_size_desc'),
            font=("Arial", 11),
            text_color="#888888",
            justify="left"
        )
        self.academic_patch_desc.pack(anchor="w", padx=15, pady=(0, 5))

        # パッチサイズ変数（デフォルト16）
        self.academic_patch_size = tk.IntVar(value=16)

        patch_frame = ctk.CTkFrame(self.config_accordion.content_frame, fg_color="transparent")
        patch_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        # 8×8オプション
        self.academic_patch_8 = ctk.CTkRadioButton(
            patch_frame,
            text=self.i18n.t('academic.patch_8x8'),
            variable=self.academic_patch_size,
            value=8,
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#ff6b6b",
            hover_color="#ee5555"
        )
        self.academic_patch_8.pack(anchor="w", pady=(0, 5))

        # 16×16オプション（推奨）
        self.academic_patch_16 = ctk.CTkRadioButton(
            patch_frame,
            text=self.i18n.t('academic.patch_16x16'),
            variable=self.academic_patch_size,
            value=16,
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#9b59b6",
            hover_color="#7d3c98"
        )
        self.academic_patch_16.pack(anchor="w", pady=(0, 5))

        # 32×32オプション
        self.academic_patch_32 = ctk.CTkRadioButton(
            patch_frame,
            text=self.i18n.t('academic.patch_32x32'),
            variable=self.academic_patch_size,
            value=32,
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#4ecdc4",
            hover_color="#3db8af"
        )
        self.academic_patch_32.pack(anchor="w")

        # 実行ボタン
        self.academic_analyze_btn = ctk.CTkButton(
            self.academic_mode_frame,
            text=f"[RUN] {self.i18n.t('buttons.analyze_academic')}",
            command=self.start_academic_analysis,
            height=60,
            corner_radius=10,
            font=("Arial", 18, "bold"),
            fg_color="#9b59b6",
            text_color="#FFFFFF",
            hover_color="#7d3c98"
        )
        self.academic_analyze_btn.pack(fill=tk.X, pady=(0, 15))

        # === アコーディオン: 統計分析 ===
        self.academic_stats_accordion = AccordionSection(
            self.academic_mode_frame,
            self.i18n.t('sections.stats_analysis'),
            bg_color="#1e2740",
            title_color="#ff6b6b",
            font_size=18
        )

        self.stats_info_academic = ctk.CTkLabel(
            self.academic_stats_accordion.content_frame,
            text=self.i18n.t('academic.stats_warning'),
            font=("Arial", 13),
            text_color="#ffcc00",
            justify="left"
        )
        self.stats_info_academic.pack(anchor="w", padx=15, pady=(10, 10))

        stats_csv_frame = ctk.CTkFrame(self.academic_stats_accordion.content_frame, fg_color="transparent")
        stats_csv_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        stats_csv_label = ctk.CTkLabel(
            stats_csv_frame,
            text=self.i18n.t('academic.csv_label'),
            width=80,
            anchor="w",
            font=("Arial", 13)
        )
        stats_csv_label.pack(side=tk.LEFT, padx=(0, 10))

        self.academic_stats_csv_path = tk.StringVar()
        stats_csv_entry = ctk.CTkEntry(
            stats_csv_frame,
            textvariable=self.academic_stats_csv_path,
            placeholder_text=self.i18n.t('academic.placeholder_stats_csv'),
            height=45,
            font=("Arial", 13)
        )
        stats_csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        stats_csv_btn = ctk.CTkButton(
            stats_csv_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_academic_stats_csv,
            width=90,
            height=45,
            font=("Arial", 14),
            fg_color="#9b59b6",
            text_color="#FFFFFF",
            hover_color="#7d3c98"
        )
        stats_csv_btn.pack(side=tk.RIGHT)

        self.academic_stats_analyze_btn = ctk.CTkButton(
            self.academic_stats_accordion.content_frame,
            text=f"[ANALYZE] {self.i18n.t('buttons.analyze_stats')}",
            command=self.start_academic_stats_analysis,
            height=55,
            corner_radius=10,
            font=("Arial", 16, "bold"),
            fg_color="#ff6b6b",
            text_color="#FFFFFF",
            hover_color="#ff4444"
        )
        self.academic_stats_analyze_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

    def browse_academic_input(self):
        """学術評価用：入力フォルダ選択"""
        dirname = filedialog.askdirectory(title=self.i18n.t('academic.select_gt_folder'))
        if dirname:
            self.academic_input_dir.set(dirname)

    def browse_academic_output(self):
        """学術評価用：出力フォルダ選択"""
        dirname = filedialog.askdirectory(title=self.i18n.t('academic.select_lr_folder'))
        if dirname:
            self.academic_output_dir.set(dirname)


    def browse_academic_original(self):
        dirname = filedialog.askdirectory(title=self.i18n.t('academic.select_original_folder'))
        if dirname:
            self.academic_original_dir.set(dirname)

    def browse_academic_model(self, index):
        dirname = filedialog.askdirectory(title=self.i18n.t('academic.select_model_folder').format(num=index+1))
        if dirname:
            self.academic_model_vars[index].set(dirname)

    def browse_academic_csv_output(self):
        """論文用：CSV出力先選択"""
        filename = filedialog.asksaveasfilename(
            title=self.i18n.t('academic.select_csv_output'),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), (self.i18n.t('messages.select_folder'), "*.*")],
            initialfile=get_timestamp_filename("batch_results_academic", ".csv")
        )
        if filename:
            self.academic_output_csv.set(filename)

    def browse_academic_detail_output(self):
        """論文用：詳細レポート出力先フォルダ選択"""
        dirname = filedialog.askdirectory(title=self.i18n.t('academic.select_detail_folder'))
        if dirname:
            self.academic_output_detail.set(dirname)

    def browse_academic_stats_csv(self):
        """論文用：統計分析用CSV選択"""
        filename = filedialog.askopenfilename(
            title=self.i18n.t('academic.select_stats_csv'),
            filetypes=[("CSV", "*.csv"), (self.i18n.t('messages.select_folder'), "*.*")]
        )
        if filename:
            self.academic_stats_csv_path.set(filename)


    def start_academic_analysis(self):
        """論文用ベンチマーク評価開始"""
        # バリデーション：元画像フォルダ（必須）
        if not self.academic_original_dir.get():
            messagebox.showerror(self.i18n.t('messages.error'), self.i18n.t('academic.error_no_original'))
            return

        if not os.path.exists(self.academic_original_dir.get()):
            messagebox.showerror(self.i18n.t('messages.error'),
                self.i18n.t('academic.error_original_not_found').format(path=self.academic_original_dir.get()))
            return

        # バリデーション：有効なモデルフォルダをカウント
        valid_models = {}
        for i in range(5):
            model_name = self.academic_model_name_vars[i].get().strip()
            model_path = self.academic_model_vars[i].get().strip()

            if model_path:
                # モデル名が空の場合
                if not model_name:
                    messagebox.showerror(self.i18n.t('messages.error'),
                        self.i18n.t('academic.error_no_model_name').format(num=i+1))
                    return
                # フォルダが存在しない場合
                if not os.path.exists(model_path):
                    messagebox.showerror(self.i18n.t('messages.error'),
                        self.i18n.t('academic.error_model_not_found').format(num=i+1, path=model_path))
                    return
                valid_models[model_name] = model_path

        # 最低1つは必須（画像1に相当）
        if len(valid_models) == 0:
            messagebox.showerror(self.i18n.t('messages.error'), self.i18n.t('academic.error_no_models'))
            return

        # 設定ファイル作成（評価モード固定：academic）
        from multiprocessing import cpu_count

        # 並列処理数の決定
        if self.academic_use_parallel.get():
            # 並列処理ON: ユーザー指定のプロセス数を使用
            num_workers = max(1, min(self.academic_num_workers.get(), cpu_count()))
        else:
            # 並列処理OFF: 1プロセス（並列なし）
            num_workers = 1

        config = {
            "original_dir": self.academic_original_dir.get(),
            "upscaled_dirs": valid_models,
            "output_csv": self.academic_output_csv.get(),
            "output_detail_dir": self.academic_output_detail.get(),
            "limit": self.academic_limit.get(),
            "append_mode": self.academic_append_mode.get(),
            "evaluation_mode": "academic",  # 学術評価モード固定
            "num_workers": num_workers,  # 並列処理数（ユーザー設定）
            "checkpoint_interval": 1000,  # チェックポイント間隔（1000サンプルごと）
            "patch_size": self.academic_patch_size.get()  # P6ヒートマップのパッチサイズ
        }

        # UIを無効化
        self.academic_analyze_btn.configure(state='disabled')
        self.academic_progress.set(0)
        self.academic_status_label.configure(
            text=self.i18n.t('academic.starting_evaluation'),
            text_color="#00ffff"
        )
        self.academic_result_text.delete("1.0", tk.END)

        # 別スレッドで実行
        thread = threading.Thread(target=self.run_academic_analysis, args=(config,))
        thread.daemon = True
        thread.start()

    def run_academic_analysis(self, config):
        """論文用ベンチマーク評価実行"""
        try:
            import sys
            from io import StringIO
            from batch_analyzer import batch_analyze
            from pathlib import Path

            # 一時設定ファイル作成
            temp_config_path = "temp_academic_config.json"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # バッチ処理実行（進捗コールバック付き）
            batch_analyze(temp_config_path, progress_callback=self.update_academic_progress)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # 一時ファイル削除
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            self.root.after(0, self.display_academic_results, output, True, config['output_csv'])

        except Exception as e:
            sys.stdout = old_stdout
            self.root.after(0, self.display_academic_results, str(e), False, None)

    def update_academic_progress(self, current, total, message):
        """論文用評価進捗更新（別スレッドから呼ばれる）"""
        progress = current / total if total > 0 else 0
        self.root.after(0, lambda: self.academic_progress.set(progress))
        progress_text = self.i18n.t('messages.processing').format(current=current, total=total)
        self.root.after(0, lambda: self.academic_status_label.configure(
            text=f"{progress_text} - {message}",
            text_color="#9b59b6"
        ))
        self.root.after(0, lambda: self.academic_result_text.insert(tk.END, f"{message}\n"))
        self.root.after(0, lambda: self.academic_result_text.see(tk.END))

    def display_academic_results(self, output, success, csv_path):
        """論文用評価結果表示"""
        self.academic_analyze_btn.configure(state='normal')
        self.academic_progress.set(1 if success else 0)

        self.academic_result_text.insert("1.0", output)

        if success:
            self.academic_status_label.configure(
                text=self.i18n.t('academic.evaluation_complete'),
                text_color="#00ff88"
            )

            # CSVパスを統計分析欄に自動入力
            if csv_path:
                self.academic_stats_csv_path.set(csv_path)

            messagebox.showinfo(
                self.i18n.t('messages.completed'),
                self.i18n.t('academic.evaluation_complete_detail').format(csv_path=csv_path)
            )
        else:
            self.academic_status_label.configure(
                text=self.i18n.t('academic.evaluation_error'),
                text_color="#ff4444"
            )
            messagebox.showerror(self.i18n.t('messages.error'),
                self.i18n.t('errors.analysis_failed').format(error=output))


    def start_academic_stats_analysis(self):
        """論文用統計分析開始"""
        csv_path = self.academic_stats_csv_path.get()

        if not csv_path:
            messagebox.showerror(self.i18n.t('messages.error'), self.i18n.t('academic.error_no_stats_csv'))
            return

        if not os.path.exists(csv_path):
            messagebox.showerror(self.i18n.t('messages.error'),
                self.i18n.t('academic.error_stats_csv_not_found').format(path=csv_path))
            return

        # UIを無効化
        self.academic_stats_analyze_btn.configure(state='disabled')
        self.academic_status_label.configure(
            text=self.i18n.t('academic.stats_running'),
            text_color="#ffa500"
        )

        # 別スレッドで実行
        thread = threading.Thread(target=self.run_academic_stats_analysis, args=(csv_path,))
        thread.daemon = True
        thread.start()

    def run_academic_stats_analysis(self, csv_path):
        """論文用統計分析実行"""
        try:
            import sys
            from io import StringIO
            from analyze_results import analyze_batch_results

            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # 統計分析実行（出力ディレクトリのパスを取得）
            output_dir = analyze_batch_results(csv_path)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            self.root.after(0, self.display_academic_stats_results, output, True, output_dir)

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            sys.stdout = old_stdout
            self.root.after(0, self.display_academic_stats_results, error_detail, False, None)

    def display_academic_stats_results(self, output, success, output_dir=None):
        """論文用統計分析結果表示"""
        self.academic_stats_analyze_btn.configure(state='normal')

        self.academic_result_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.academic_result_text.insert(tk.END, output)
        self.academic_result_text.see(tk.END)

        if success:
            self.academic_status_label.configure(
                text=f"[OK] 統計分析完了！25種類のプロットが {output_dir}/ に保存されました",
                text_color="#00ff88"
            )

            messagebox.showinfo(
                "完了",
                f"統計分析が完了しました。\n\n"
                f"25種類の研究用プロット（300dpi）が\n"
                f"{output_dir}/ フォルダに保存されました。\n\n"
                f"・ハルシネーション検出（4種類）\n"
                f"・品質トレードオフ（5種類）\n"
                f"・医療画像特化（4種類）\n"
                f"・分布・PCA分析（4種類）\n"
                f"・その他（6種類）"
            )

            # フォルダを開くか確認
            result = messagebox.askyesno(
                "フォルダを開く",
                f"{output_dir} フォルダを開きますか？"
            )
            if result:
                if output_dir and os.path.exists(output_dir):
                    os.startfile(output_dir)
        else:
            self.academic_status_label.configure(
                text=self.i18n.t('academic.stats_error'),
                text_color="#ff4444"
            )
            messagebox.showerror(self.i18n.t('messages.error'),
                self.i18n.t('errors.analysis_failed').format(error=output))


    def toggle_academic_parallel_settings(self):
        """並列処理ON/OFF時の処理（論文用タブ）"""
        if self.academic_use_parallel.get():
            # 並列処理ON: プロセス数入力を有効化
            self.academic_workers_spinbox.configure(state='normal')
        else:
            # 並列処理OFF: プロセス数入力を無効化
            self.academic_workers_spinbox.configure(state='disabled')


    def generate_lowres_academic(self):
        """学術評価用の低解像度画像を生成（Bicubic縮小 ×0.5）"""
        import cv2
        import os
        from tkinter import messagebox

        # 元画像パスを確認
        original_path = self.original_path.get()
        if not original_path:
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('academic.error_select_original_first')
            )
            return

        if not os.path.exists(original_path):
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('academic.error_original_file_not_found').format(path=original_path)
            )
            return

        try:
            # 画像読み込み（ビット深度・カラー形式を保持）
            img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                messagebox.showerror(
                    self.i18n.t('messages.error'),
                    self.i18n.t('academic.error_image_load_failed')
                )
                return

            h, w = img.shape[:2]

            # Bicubic縮小（×0.5）
            img_lr = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_CUBIC)

            # 出力パスを生成
            base_dir = os.path.dirname(original_path)
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_LR_bicubic_x05.png")

            # 保存
            cv2.imwrite(output_path, img_lr)

            # 成功メッセージ
            messagebox.showinfo(
                self.i18n.t('academic.bicubic_complete_title'),
                self.i18n.t('academic.bicubic_complete_message').format(
                    w=w,
                    h=h,
                    w_lr=w//2,
                    h_lr=h//2,
                    output_path=output_path
                )
            )

        except Exception as e:
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('academic.error_bicubic_failed').format(error=str(e))
            )

