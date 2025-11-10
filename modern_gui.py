import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
from advanced_image_analyzer import analyze_images
import json
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw
import customtkinter as ctk
from i18n import get_i18n

# CPU/GPUモニタリング
try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# CustomTkinter設定
ctk.set_appearance_mode("dark")  # ダークモード
ctk.set_default_color_theme("blue")  # カラーテーマ


# 分離したモジュールのインポート
from ui_components import AccordionSection, get_timestamp_filename
from system_monitor import SystemMonitorMixin
from batch_mode import BatchModeMixin
from academic_mode import AcademicModeMixin
from stats_analysis import StatsAnalysisMixin
from data_extraction import DataExtractionMixin

class ModernImageAnalyzerGUI(
    SystemMonitorMixin,
    BatchModeMixin,
    AcademicModeMixin,
    StatsAnalysisMixin,
    DataExtractionMixin
):
    def __init__(self, root):
        self.root = root

        # 多言語対応の初期化（最初に実行）
        self.i18n = get_i18n(default_language='ja')  # デフォルト日本語
        self.current_language = 'ja'

        self.root.title(self.i18n.t('app.title'))
        self.root.geometry("1200x800")

        # 背景色設定（サイバーパンク風）
        self.bg_color = "#0a0e27"
        self.accent_color = "#00ffff"
        self.secondary_color = "#1e2740"

        # 変数
        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()
        self.img3_path = tk.StringVar()
        self.img4_path = tk.StringVar()
        self.img5_path = tk.StringVar()
        self.original_path = tk.StringVar()  # 元画像（GT画像・必須）
        self.output_dir = tk.StringVar(value="analysis_results")
        self.analysis_results = None
        self.current_step = ""

        # モニタリング用
        self.monitoring_active = False
        self.cpu_usage = 0
        self.gpu_usage = 0
        self.ram_usage = 0

        self.create_modern_ui()

        # リアルタイムモニタリング開始
        if MONITORING_AVAILABLE:
            self.start_monitoring()

    def create_modern_ui(self):
        # メインコンテナ
        main_container = ctk.CTkFrame(self.root, fg_color="#0a0e27")
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ヘッダー部分
        header_frame = ctk.CTkFrame(main_container, fg_color="#1e2740", height=120, corner_radius=0)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        # ロゴ画像を読み込み（左側に配置）
        try:
            logo_image = Image.open("images/maou.jpg")
            logo_image = logo_image.resize((80, 80), Image.Resampling.LANCZOS)
            # 円形にクロップ
            mask = Image.new('L', (80, 80), 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 80, 80), fill=255)
            logo_image.putalpha(mask)
            self.logo_photo = ImageTk.PhotoImage(logo_image)

            logo_label = tk.Label(header_frame, image=self.logo_photo, bg="#1e2740")
            logo_label.place(x=30, y=20)
        except:
            pass

        # タイトル（翻訳対応）
        self.title_label = ctk.CTkLabel(
            header_frame,
            text=self.i18n.t('app.title'),
            font=("Arial", 32, "bold"),
            text_color="#4A90E2"
        )
        self.title_label.place(x=130, y=25)

        # サブタイトル（翻訳対応）
        self.subtitle_label = ctk.CTkLabel(
            header_frame,
            text=self.i18n.t('app.subtitle'),
            font=("Arial", 14),
            text_color="#888888"
        )
        self.subtitle_label.place(x=130, y=70)

        # 右上コンテナ（言語選択とシステムモニターを配置）
        right_header_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_header_container.place(relx=1.0, x=-20, y=25, anchor="ne")

        # システムモニター（CPU/GPU/RAM）を一番右に配置
        if MONITORING_AVAILABLE:
            monitor_frame = ctk.CTkFrame(right_header_container, fg_color="transparent")
            monitor_frame.pack(side=tk.RIGHT, padx=(20, 0))

            # CPUメーター
            self.cpu_label = ctk.CTkLabel(
                monitor_frame,
                text="CPU",
                font=("Arial", 10, "bold"),
                text_color="#4A90E2"
            )
            self.cpu_label.grid(row=0, column=0, padx=10)

            self.cpu_canvas = tk.Canvas(
                monitor_frame,
                width=60,
                height=60,
                bg="#1e2740",
                highlightthickness=0
            )
            self.cpu_canvas.grid(row=1, column=0, padx=10)

            # GPUメーター
            self.gpu_label = ctk.CTkLabel(
                monitor_frame,
                text="GPU",
                font=("Arial", 10, "bold"),
                text_color="#00ff88"
            )
            self.gpu_label.grid(row=0, column=1, padx=10)

            self.gpu_canvas = tk.Canvas(
                monitor_frame,
                width=60,
                height=60,
                bg="#1e2740",
                highlightthickness=0
            )
            self.gpu_canvas.grid(row=1, column=1, padx=10)

            # RAMメーター
            self.ram_label = ctk.CTkLabel(
                monitor_frame,
                text="RAM",
                font=("Arial", 10, "bold"),
                text_color="#ffaa00"
            )
            self.ram_label.grid(row=0, column=2, padx=10)

            self.ram_canvas = tk.Canvas(
                monitor_frame,
                width=60,
                height=60,
                bg="#1e2740",
                highlightthickness=0
            )
            self.ram_canvas.grid(row=1, column=2, padx=10)

        # 言語切り替えボタン（システムモニターの左）
        lang_frame = ctk.CTkFrame(right_header_container, fg_color="transparent")
        lang_frame.pack(side=tk.RIGHT, padx=(0, 20))

        lang_label = ctk.CTkLabel(
            lang_frame,
            text="LANG",
            font=("Arial", 20),
            text_color="#4A90E2"
        )
        lang_label.pack(side=tk.LEFT, padx=(0, 10))

        self.lang_button = ctk.CTkButton(
            lang_frame,
            text=self.i18n.t('gui.lang_japanese'),
            command=self.toggle_language,
            width=120,
            height=35,
            corner_radius=8,
            font=("Arial", 12, "bold"),
            fg_color="#2d3748",
            hover_color="#4A90E2"
        )
        self.lang_button.pack(side=tk.LEFT)

        # モード切り替えボタンエリア
        mode_frame = ctk.CTkFrame(main_container, fg_color="#1e2740", height=60, corner_radius=0)
        mode_frame.pack(fill=tk.X, padx=0, pady=0)
        mode_frame.pack_propagate(False)

        # モード切り替えボタン
        button_container = ctk.CTkFrame(mode_frame, fg_color="transparent")
        button_container.place(relx=0.5, rely=0.5, anchor="center")

        self.single_mode_btn = ctk.CTkButton(
            button_container,
            text=f"[IMG] {self.i18n.t('tabs.single_analysis')}",
            command=self.switch_to_single_mode,
            height=40,
            width=180,
            corner_radius=10,
            font=("Arial", 13, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.single_mode_btn.pack(side=tk.LEFT, padx=5)

        self.batch_mode_btn = ctk.CTkButton(
            button_container,
            text=f"[BATCH] {self.i18n.t('tabs.batch_processing')}",
            command=self.switch_to_batch_mode,
            height=40,
            width=180,
            corner_radius=10,
            font=("Arial", 13, "bold"),
            fg_color="#4a5568",
            text_color="#ffffff",
            hover_color="#2d3748"
        )
        self.batch_mode_btn.pack(side=tk.LEFT, padx=5)

        self.academic_mode_btn = ctk.CTkButton(
            button_container,
            text=f"[ACAD] {self.i18n.t('tabs.academic_benchmark')}",
            command=self.switch_to_academic_mode,
            height=40,
            width=220,
            corner_radius=10,
            font=("Arial", 13, "bold"),
            fg_color="#4a5568",
            text_color="#ffffff",
            hover_color="#2d3748"
        )
        self.academic_mode_btn.pack(side=tk.LEFT, padx=5)

        # コンテンツエリア（2カラムレイアウト - リサイズ可能）
        content_frame = ctk.CTkFrame(main_container, fg_color="#0a0e27")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # PanedWindowで左右調整可能に（標準tkinter・最適化版）
        self.paned_window = tk.PanedWindow(
            content_frame,
            orient=tk.HORIZONTAL,
            bg="#0a0e27",
            sashwidth=8,
            sashrelief=tk.RAISED,
            bd=0,
            handlesize=10,
            handlepad=30,
            showhandle=True,
            sashpad=2,
            relief=tk.FLAT,
            opaqueresize=False  # リサイズ中は枠線のみ表示（パフォーマンス向上）
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # 左側パネル（入力エリア）
        self.left_panel = ctk.CTkFrame(self.paned_window, fg_color="#1e2740", corner_radius=15)
        self.paned_window.add(self.left_panel, width=550, minsize=400, stretch="never")

        # スクロール可能なフレーム（単一モード用）
        self.single_mode_frame = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self.single_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self._improve_scroll_speed(self.single_mode_frame)

        # バッチモード用フレーム（後で作成）
        self.batch_mode_frame = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self._improve_scroll_speed(self.batch_mode_frame)

        # 論文用ベンチマーク評価モード用フレーム（後で作成）
        self.academic_mode_frame = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self._improve_scroll_speed(self.academic_mode_frame)

        # 右側パネル（画像比較・結果表示エリア）
        self.right_panel = ctk.CTkFrame(self.paned_window, fg_color="#1e2740", corner_radius=15)
        self.paned_window.add(self.right_panel, minsize=300, stretch="always")

        # === 単一画像比較モードのUI ===
        # 画像選択セクション
        input_section = ctk.CTkFrame(self.single_mode_frame, fg_color="transparent")
        input_section.pack(fill=tk.X)

        # 評価モード選択（アコーディオン）
        self.mode_accordion = AccordionSection(input_section, self.i18n.t('sections.evaluation_settings'), font_size=18)
        mode_frame = self.mode_accordion.get_content_frame()

        # 評価モード変数
        self.evaluation_mode = tk.StringVar(value="image")

        # 画像モード（翻訳対応）
        self.mode_image = ctk.CTkRadioButton(
            mode_frame,
            text=self.i18n.t('modes.image'),
            variable=self.evaluation_mode,
            value="image",
            font=("Arial", 14),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        self.mode_image.pack(anchor="w", padx=30, pady=(15, 8))

        self.mode_image_desc = ctk.CTkLabel(
            mode_frame,
            text=f"  {self.i18n.t('modes.image_desc')}",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.mode_image_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # 文書モード（翻訳対応）
        self.mode_document = ctk.CTkRadioButton(
            mode_frame,
            text=self.i18n.t('modes.document'),
            variable=self.evaluation_mode,
            value="document",
            font=("Arial", 14),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        self.mode_document.pack(anchor="w", padx=30, pady=(0, 8))

        self.mode_document_desc = ctk.CTkLabel(
            mode_frame,
            text=f"  {self.i18n.t('modes.document_desc')}",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.mode_document_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # 開発者モード（翻訳対応）
        self.mode_developer = ctk.CTkRadioButton(
            mode_frame,
            text=self.i18n.t('modes.developer'),
            variable=self.evaluation_mode,
            value="developer",
            font=("Arial", 14),
            text_color="#ffffff",
            fg_color="#ffa500",
            hover_color="#cc8400"
        )
        self.mode_developer.pack(anchor="w", padx=30, pady=(0, 8))

        self.mode_developer_desc = ctk.CTkLabel(
            mode_frame,
            text=f"  {self.i18n.t('modes.developer_desc')}",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.mode_developer_desc.pack(anchor="w", padx=30, pady=(0, 15))

        # P6パッチサイズ選択
        self.patch_label = ctk.CTkLabel(
            mode_frame,
            text=self.i18n.t('gui.patch_size_title'),
            font=("Arial", 14, "bold"),
            text_color="#00ffff"
        )
        self.patch_label.pack(anchor="w", padx=30, pady=(15, 8))

        # パッチサイズ変数（デフォルト16）
        self.patch_size = tk.IntVar(value=16)

        # 8×8オプション
        self.patch_8 = ctk.CTkRadioButton(
            mode_frame,
            text=self.i18n.t('gui.patch_8x8'),
            variable=self.patch_size,
            value=8,
            font=("Arial", 13),
            text_color="#ffffff",
            fg_color="#ff6b6b",
            hover_color="#ee5555"
        )
        self.patch_8.pack(anchor="w", padx=30, pady=(0, 5))

        self.patch_8_desc = ctk.CTkLabel(
            mode_frame,
            text=self.i18n.t('gui.patch_8x8_detail'),
            font=("Arial", 11),
            text_color="#888888"
        )
        self.patch_8_desc.pack(anchor="w", padx=30, pady=(0, 8))

        # 16×16オプション（推奨）
        self.patch_16 = ctk.CTkRadioButton(
            mode_frame,
            text=self.i18n.t('gui.patch_16x16'),
            variable=self.patch_size,
            value=16,
            font=("Arial", 13),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        self.patch_16.pack(anchor="w", padx=30, pady=(0, 5))

        self.patch_16_desc = ctk.CTkLabel(
            mode_frame,
            text=self.i18n.t('gui.patch_16x16_detail'),
            font=("Arial", 11),
            text_color="#888888"
        )
        self.patch_16_desc.pack(anchor="w", padx=30, pady=(0, 8))

        # 32×32オプション
        self.patch_32 = ctk.CTkRadioButton(
            mode_frame,
            text=self.i18n.t('gui.patch_32x32'),
            variable=self.patch_size,
            value=32,
            font=("Arial", 13),
            text_color="#ffffff",
            fg_color="#4ecdc4",
            hover_color="#3db8af"
        )
        self.patch_32.pack(anchor="w", padx=30, pady=(0, 5))

        self.patch_32_desc = ctk.CTkLabel(
            mode_frame,
            text=self.i18n.t('gui.patch_32x32_detail'),
            font=("Arial", 11),
            text_color="#888888"
        )
        self.patch_32_desc.pack(anchor="w", padx=30, pady=(0, 15))

        # 元画像（必須）
        self.original_accordion = AccordionSection(input_section, self.i18n.t('sections.original_image_required'), bg_color="#1b3d1b", title_color="#00ff88", font_size=18)
        # デフォルトで開く（閉じない）
        original_frame = self.original_accordion.get_content_frame()

        # PNG推奨警告（翻訳対応）
        self.single_png_warning = ctk.CTkLabel(
            original_frame,
            text=self.i18n.t('warnings.png_recommended'),
            font=("Arial", 12, "bold"),
            text_color="#ff6b6b"
        )
        self.single_png_warning.pack(anchor="w", padx=15, pady=(15, 5))

        original_sublabel = ctk.CTkLabel(
            original_frame,
            text=self.i18n.t('gui.original_note'),
            font=("Arial", 12),
            text_color="#888888",
            justify="left"
        )
        original_sublabel.pack(anchor="w", padx=15, pady=(5, 10))

        original_entry = ctk.CTkEntry(
            original_frame,
            textvariable=self.original_path,
            placeholder_text=self.i18n.t('gui.placeholder_select_original'),
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        original_entry.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.original_browse_btn = ctk.CTkButton(
            original_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_original,
            height=45,
            width=200,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#00ff88",
            text_color="#000000",
            hover_color="#00cc66"
        )
        self.original_browse_btn.pack(padx=15, pady=(0, 15), anchor="w")

        # 画像1（必須）
        self.img1_accordion = AccordionSection(input_section, self.i18n.t('sections.upscaled_image_1'), font_size=18)
        img1_frame = self.img1_accordion.get_content_frame()

        img1_entry = ctk.CTkEntry(
            img1_frame,
            textvariable=self.img1_path,
            placeholder_text=self.i18n.t('gui.placeholder_select_image'),
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        img1_entry.pack(fill=tk.X, padx=15, pady=(15, 10))

        self.img1_browse_btn = ctk.CTkButton(
            img1_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_image1,
            height=45,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.img1_browse_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 画像2（オプション）
        self.img2_accordion = AccordionSection(input_section, self.i18n.t('sections.upscaled_image_2'), bg_color="#1e2740", title_color="#4A90E2", font_size=18)
        img2_frame = self.img2_accordion.get_content_frame()

        img2_entry = ctk.CTkEntry(
            img2_frame,
            textvariable=self.img2_path,
            placeholder_text=self.i18n.t('gui.placeholder_select_image'),
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        img2_entry.pack(fill=tk.X, padx=15, pady=(15, 10))

        self.img2_browse_btn = ctk.CTkButton(
            img2_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_image2,
            height=45,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.img2_browse_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 画像3（オプション）
        self.img3_accordion = AccordionSection(input_section, self.i18n.t('sections.upscaled_image_3'), bg_color="#1e2740", title_color="#4A90E2", font_size=18)
        self.img3_accordion.toggle()  # デフォルトで閉じる
        img3_frame = self.img3_accordion.get_content_frame()

        img3_entry = ctk.CTkEntry(
            img3_frame,
            textvariable=self.img3_path,
            placeholder_text=self.i18n.t('gui.placeholder_select_image'),
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        img3_entry.pack(fill=tk.X, padx=15, pady=(15, 10))

        self.img3_browse_btn = ctk.CTkButton(
            img3_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_image3,
            height=45,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.img3_browse_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 画像4（オプション）
        self.img4_accordion = AccordionSection(input_section, self.i18n.t('sections.upscaled_image_4'), bg_color="#1e2740", title_color="#4A90E2", font_size=18)
        self.img4_accordion.toggle()  # デフォルトで閉じる
        img4_frame = self.img4_accordion.get_content_frame()

        img4_entry = ctk.CTkEntry(
            img4_frame,
            textvariable=self.img4_path,
            placeholder_text=self.i18n.t('gui.placeholder_select_image'),
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        img4_entry.pack(fill=tk.X, padx=15, pady=(15, 10))

        self.img4_browse_btn = ctk.CTkButton(
            img4_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_image4,
            height=45,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.img4_browse_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 画像5（オプション）
        self.img5_accordion = AccordionSection(input_section, self.i18n.t('sections.upscaled_image_5'), bg_color="#1e2740", title_color="#4A90E2", font_size=18)
        self.img5_accordion.toggle()  # デフォルトで閉じる
        img5_frame = self.img5_accordion.get_content_frame()

        img5_entry = ctk.CTkEntry(
            img5_frame,
            textvariable=self.img5_path,
            placeholder_text=self.i18n.t('gui.placeholder_select_image'),
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        img5_entry.pack(fill=tk.X, padx=15, pady=(15, 10))

        self.img5_browse_btn = ctk.CTkButton(
            img5_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_image5,
            height=45,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.img5_browse_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 出力フォルダ（アコーディオン）
        self.output_accordion = AccordionSection(input_section, self.i18n.t('sections.output_folder'), font_size=18)
        output_frame = self.output_accordion.get_content_frame()

        output_entry = ctk.CTkEntry(
            output_frame,
            textvariable=self.output_dir,
            height=45,
            corner_radius=10,
            font=("Arial", 13)
        )
        output_entry.pack(fill=tk.X, padx=15, pady=(15, 10))

        self.output_browse_btn = ctk.CTkButton(
            output_frame,
            text=self.i18n.t('buttons.browse'),
            command=self.browse_output,
            height=45,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.output_browse_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 分析開始ボタン（大きく目立つ）（翻訳対応）
        self.analyze_btn = ctk.CTkButton(
            input_section,
            text=f"[RUN] {self.i18n.t('buttons.analyze')}",
            command=self.start_analysis,
            height=70,
            corner_radius=15,
            font=("Arial", 20, "bold"),
            fg_color="#00ff88",
            text_color="#000000",
            hover_color="#00dd77"
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 20))

        # プログレスバー
        self.progress = ctk.CTkProgressBar(
            input_section,
            height=15,
            corner_radius=10,
            progress_color="#00ffff"
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))
        self.progress.set(0)

        # ステータス
        self.status_label = ctk.CTkLabel(
            input_section,
            text=self.i18n.t('gui.status_start'),
            font=("Arial", 11),
            text_color="#888888",
            wraplength=350
        )
        self.status_label.pack()

        # ボタングループ
        button_group = ctk.CTkFrame(self.single_mode_frame, fg_color="transparent")
        button_group.pack(fill=tk.X, pady=(20, 0))

        btn_report = ctk.CTkButton(
            button_group,
            text=self.i18n.t('gui.report_button'),
            command=self.show_comparison_report,
            height=40,
            corner_radius=10,
            font=("Arial", 11, "bold"),
            fg_color="#4a5568",
            hover_color="#2d3748"
        )
        btn_report.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        btn_folder = ctk.CTkButton(
            button_group,
            text=self.i18n.t('gui.folder_button'),
            command=self.open_output_folder,
            height=40,
            corner_radius=10,
            font=("Arial", 11, "bold"),
            fg_color="#4a5568",
            hover_color="#2d3748"
        )
        btn_folder.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        btn_clear = ctk.CTkButton(
            button_group,
            text=self.i18n.t('gui.clear_button'),
            command=self.clear_results,
            height=40,
            corner_radius=10,
            font=("Arial", 11, "bold"),
            fg_color="#4a5568",
            hover_color="#2d3748"
        )
        btn_clear.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # === 右側パネル：単一モード用フレーム ===
        self.single_right_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.single_right_frame.pack(fill=tk.BOTH, expand=True)

        # 画像比較プレビューエリア
        preview_title = ctk.CTkLabel(
            self.single_right_frame,
            text=self.i18n.t('gui.preview_title'),
            font=("Arial", 18, "bold"),
            text_color="#4A90E2"
        )
        preview_title.pack(pady=(20, 10))

        # 画像表示フレーム（Before/After）
        image_compare_frame = ctk.CTkFrame(self.single_right_frame, fg_color="#0a0e27", corner_radius=10, height=300)
        image_compare_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        image_compare_frame.pack_propagate(False)

        # 元画像（Before）
        img_before_container = ctk.CTkFrame(image_compare_frame, fg_color="transparent")
        img_before_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.img_before_title = ctk.CTkLabel(
            img_before_container,
            text=self.i18n.t('gui.original_before'),
            font=("Arial", 12, "bold"),
            text_color="#FFA500"
        )
        self.img_before_title.pack(pady=(0, 5))

        self.preview_img_before_label = tk.Label(
            img_before_container,
            bg="#0a0e27",
            text=self.i18n.t('gui.select_original_prompt'),
            fg="#888888",
            font=("Arial", 10)
        )
        self.preview_img_before_label.pack(fill=tk.BOTH, expand=True)

        # 画像1（超解像結果1 - After）
        img1_container = ctk.CTkFrame(image_compare_frame, fg_color="transparent")
        img1_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.img1_title = ctk.CTkLabel(
            img1_container,
            text=self.i18n.t('gui.sr_result_1'),
            font=("Arial", 12, "bold"),
            text_color="#00ff88"
        )
        self.img1_title.pack(pady=(0, 5))

        self.preview_img1_label = tk.Label(
            img1_container,
            bg="#0a0e27",
            text=self.i18n.t('gui.select_sr1_prompt'),
            fg="#888888",
            font=("Arial", 10)
        )
        self.preview_img1_label.pack(fill=tk.BOTH, expand=True)

        # 画像2（超解像結果2 - After）
        img2_container = ctk.CTkFrame(image_compare_frame, fg_color="transparent")
        img2_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.img2_title = ctk.CTkLabel(
            img2_container,
            text=self.i18n.t('gui.sr_result_2'),
            font=("Arial", 12, "bold"),
            text_color="#00ff88"
        )
        self.img2_title.pack(pady=(0, 5))

        self.preview_img2_label = tk.Label(
            img2_container,
            bg="#0a0e27",
            text=self.i18n.t('gui.select_sr2_prompt'),
            fg="#888888",
            font=("Arial", 10)
        )
        self.preview_img2_label.pack(fill=tk.BOTH, expand=True)

        # 上半分：解釈結果エリア
        interpretation_frame = ctk.CTkFrame(self.single_right_frame, fg_color="transparent")
        interpretation_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 5))

        interpretation_title = ctk.CTkLabel(
            interpretation_frame,
            text=self.i18n.t('gui.interpretation_title'),
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        interpretation_title.pack(pady=(0, 5), padx=0, anchor="w")

        self.interpretation_text = ctk.CTkTextbox(
            interpretation_frame,
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#4A90E2",
            corner_radius=10,
            height=200
        )
        self.interpretation_text.pack(fill=tk.BOTH, expand=True)

        # 下半分：詳細ログエリア
        log_frame = ctk.CTkFrame(self.single_right_frame, fg_color="transparent")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 15))

        log_title = ctk.CTkLabel(
            log_frame,
            text=self.i18n.t('gui.detailed_log_title'),
            font=("Arial", 16, "bold"),
            text_color="#00ff88"
        )
        log_title.pack(pady=(0, 5), padx=0, anchor="w")

        self.result_text = ctk.CTkTextbox(
            log_frame,
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10,
            height=300
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # === バッチモード用の右パネル ===
        self.batch_right_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")

        # 上部：進捗状況エリア（コンパクト）
        batch_progress_section = ctk.CTkFrame(self.batch_right_frame, fg_color="transparent")
        batch_progress_section.pack(fill=tk.X, padx=15, pady=(15, 5))

        self.batch_progress_title = ctk.CTkLabel(
            batch_progress_section,
            text=self.i18n.t('gui.batch_progress_title'),
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        self.batch_progress_title.pack(pady=(0, 5), anchor="w")

        # 進捗表示フレーム
        self.batch_progress_frame = ctk.CTkFrame(batch_progress_section, fg_color="#0a0e27", corner_radius=10)
        self.batch_progress_frame.pack(fill=tk.X, pady=(0, 0))

        self.batch_status_label = ctk.CTkLabel(
            self.batch_progress_frame,
            text=self.i18n.t('gui.batch_start_prompt'),
            font=("Arial", 12),
            text_color="#888888"
        )
        self.batch_status_label.pack(pady=(10, 5))

        # プログレスバー
        self.batch_progress = ctk.CTkProgressBar(
            self.batch_progress_frame,
            width=400,
            height=15,
            corner_radius=10,
            fg_color="#2d3748",
            progress_color="#00ffff"
        )
        self.batch_progress.pack(pady=(0, 10), padx=20)
        self.batch_progress.set(0)

        # 下部：詳細ログエリア
        batch_log_section = ctk.CTkFrame(self.batch_right_frame, fg_color="transparent")
        batch_log_section.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 15))

        self.batch_result_label = ctk.CTkLabel(
            batch_log_section,
            text=self.i18n.t('gui.batch_log_title'),
            font=("Arial", 16, "bold"),
            text_color="#00ff88"
        )
        self.batch_result_label.pack(pady=(0, 5), anchor="w")

        self.batch_result_text = ctk.CTkTextbox(
            batch_log_section,
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10,
            height=400
        )
        self.batch_result_text.pack(fill=tk.BOTH, expand=True)

        # === 論文用ベンチマーク評価モード用の右パネル ===
        self.academic_right_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")

        # 上部：進捗状況エリア（コンパクト）
        academic_progress_section = ctk.CTkFrame(self.academic_right_frame, fg_color="transparent")
        academic_progress_section.pack(fill=tk.X, padx=15, pady=(15, 5))

        self.academic_progress_title = ctk.CTkLabel(
            academic_progress_section,
            text=self.i18n.t('gui.academic_progress_title'),
            font=("Arial", 16, "bold"),
            text_color="#9b59b6"
        )
        self.academic_progress_title.pack(pady=(0, 5), anchor="w")

        # 進捗表示フレーム
        self.academic_progress_frame = ctk.CTkFrame(academic_progress_section, fg_color="#0a0e27", corner_radius=10)
        self.academic_progress_frame.pack(fill=tk.X, pady=(0, 0))

        self.academic_status_label = ctk.CTkLabel(
            self.academic_progress_frame,
            text=self.i18n.t('gui.academic_start_prompt'),
            font=("Arial", 12),
            text_color="#888888"
        )
        self.academic_status_label.pack(pady=(10, 5))

        # プログレスバー
        self.academic_progress = ctk.CTkProgressBar(
            self.academic_progress_frame,
            width=400,
            height=15,
            corner_radius=10,
            fg_color="#2d3748",
            progress_color="#9b59b6"
        )
        self.academic_progress.pack(pady=(0, 10), padx=20)
        self.academic_progress.set(0)

        # 下部：詳細ログエリア
        academic_log_section = ctk.CTkFrame(self.academic_right_frame, fg_color="transparent")
        academic_log_section.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 15))

        self.academic_result_label = ctk.CTkLabel(
            academic_log_section,
            text=self.i18n.t('gui.academic_log_title'),
            font=("Arial", 16, "bold"),
            text_color="#00ff88"
        )
        self.academic_result_label.pack(pady=(0, 5), anchor="w")

        self.academic_result_text = ctk.CTkTextbox(
            academic_log_section,
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10,
            height=400
        )
        self.academic_result_text.pack(fill=tk.BOTH, expand=True)

        # バッチモード用のUIを作成（左パネル）
        self.create_batch_mode_ui()

        # 論文用ベンチマーク評価モード用のUIを作成（左パネル）
        self.create_academic_mode_ui()

    def _improve_scroll_speed(self, scrollable_frame):
        """CTkScrollableFrameのスクロール速度を改善（適度な速度）"""
        try:
            # CTkScrollableFrameの内部Canvasにアクセス
            canvas = scrollable_frame._parent_canvas

            # スクロール速度を上げるための設定
            # Canvasのyscrollincrement（1回のスクロール量）を調整
            canvas.configure(yscrollincrement=25)  # デフォルトは20程度、1.75倍に設定（操作しやすい速度）

        except Exception as e:
            # デバッグ用
            print(f"スクロール速度改善エラー: {e}")
            pass

    def browse_image1(self):
        filename = filedialog.askopenfilename(
            title="画像1を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.img1_path.set(filename)
            self.load_preview_image1(filename)

    def browse_image2(self):
        filename = filedialog.askopenfilename(
            title="画像2を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.img2_path.set(filename)
            self.load_preview_image2(filename)

    def browse_image3(self):
        filename = filedialog.askopenfilename(
            title="画像3を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.img3_path.set(filename)

    def browse_image4(self):
        filename = filedialog.askopenfilename(
            title="画像4を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.img4_path.set(filename)

    def browse_image5(self):
        filename = filedialog.askopenfilename(
            title="画像5を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.img5_path.set(filename)

    def browse_original(self):
        filename = filedialog.askopenfilename(
            title="元画像を選択（処理前/Before）",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.original_path.set(filename)
            self.load_preview_image_before(filename)

    def browse_output(self):
        dirname = filedialog.askdirectory(title="出力フォルダを選択")
        if dirname:
            self.output_dir.set(dirname)

    def load_preview_image1(self, filepath):
        """画像1のプレビューを読み込んで表示"""
        try:
            img = Image.open(filepath)
            # プレビューサイズに合わせてリサイズ（アスペクト比維持）
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_img1_label.configure(image=photo, text="")
            self.preview_img1_label.image = photo  # 参照を保持
        except Exception as e:
            self.preview_img1_label.configure(
                text=self.i18n.t('gui.image_load_error').format(error=str(e)),
                image=""
            )

    def load_preview_image2(self, filepath):
        """画像2のプレビューを読み込んで表示"""
        try:
            img = Image.open(filepath)
            # プレビューサイズに合わせてリサイズ（アスペクト比維持）
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_img2_label.configure(image=photo, text="")
            self.preview_img2_label.image = photo  # 参照を保持
        except Exception as e:
            self.preview_img2_label.configure(
                text=self.i18n.t('gui.image_load_error').format(error=str(e)),
                image=""
            )

    def load_preview_image_before(self, filepath):
        """元画像（Before）のプレビューを読み込んで表示"""
        try:
            img = Image.open(filepath)
            # プレビューサイズに合わせてリサイズ（アスペクト比維持）
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_img_before_label.configure(image=photo, text="")
            self.preview_img_before_label.image = photo  # 参照を保持
        except Exception as e:
            self.preview_img_before_label.configure(
                text=self.i18n.t('gui.image_load_error').format(error=str(e)),
                image=""
            )

    def start_analysis(self):
        # 元画像（GT画像）は必須
        if not self.original_path.get():
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('gui.error_select_original')
            )
            return

        if not os.path.exists(self.original_path.get()):
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('gui.error_original_not_found').format(path=self.original_path.get())
            )
            return

        # 画像1は必須
        if not self.img1_path.get():
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('gui.error_select_image1')
            )
            return

        if not os.path.exists(self.img1_path.get()):
            messagebox.showerror(
                self.i18n.t('messages.error'),
                self.i18n.t('gui.error_image1_not_found').format(path=self.img1_path.get())
            )
            return

        # 画像2-5はオプション（存在チェックのみ）
        for i, path_var in enumerate([self.img2_path, self.img3_path, self.img4_path, self.img5_path], 2):
            if path_var.get() and not os.path.exists(path_var.get()):
                messagebox.showerror(
                    self.i18n.t('messages.error'),
                    self.i18n.t('gui.error_image_not_found').format(num=i, path=path_var.get())
                )
                return

        # UIを無効化
        self.analyze_btn.configure(state='disabled')
        self.progress.set(0)
        self.progress.start()
        self.current_step = "初期化中..."
        self.status_label.configure(text=self.i18n.t('gui.status_analyzing').format(step=self.current_step), text_color="#00ffff")
        self.result_text.delete("1.0", tk.END)
        self.interpretation_text.delete("1.0", tk.END)

        # 進捗更新スレッドを開始
        self.update_progress_display()

        # 別スレッドで分析実行
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()

    def update_progress_display(self):
        """進捗状況を定期的に更新（最適化版）"""
        if self.current_step and self.analyze_btn.cget('state') == 'disabled':
            self.status_label.configure(text=self.i18n.t('gui.status_analyzing').format(step=self.current_step))
            # 100ms→300msに変更してパフォーマンス向上
            self.root.after(300, self.update_progress_display)

    def progress_callback(self, step_name):
        """分析ステップ更新用コールバック"""
        self.current_step = step_name

    def run_analysis(self):
        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout

            # カスタム出力クラスで進捗を捕捉
            class ProgressCapture(StringIO):
                def __init__(self, gui, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.gui = gui

                def write(self, s):
                    super().write(s)
                    # 進捗情報を抽出
                    if '【' in s and '】' in s:
                        step = s.split('【')[1].split('】')[0]
                        self.gui.current_step = step
                    return len(s)

            sys.stdout = captured_output = ProgressCapture(self)

            # 複数画像対応
            all_results = []
            image_paths = []

            # 画像1-5のパスを収集
            for i, path_var in enumerate([self.img1_path, self.img2_path, self.img3_path, self.img4_path, self.img5_path], 1):
                if path_var.get():
                    image_paths.append((i, path_var.get()))

            # 元画像（GT）と各AI処理結果を比較
            gt_path = self.original_path.get()

            for img_num, img_path in image_paths:
                self.current_step = f"画像{img_num}の精度を評価中..."

                # 出力ディレクトリを画像番号ごとに分ける
                output_base = self.output_dir.get() or 'analysis_results'
                output_subdir = os.path.join(output_base, f"image_{img_num}")

                results = analyze_images(
                    gt_path,  # 元画像（GT）
                    img_path,  # AI処理結果
                    output_subdir,
                    None,  # original_pathはNone
                    evaluation_mode=self.evaluation_mode.get(),
                    patch_size=self.patch_size.get()  # P6ヒートマップのパッチサイズ
                )

                # 画像番号を結果に追加
                results['image_number'] = img_num
                results['image_name'] = os.path.basename(img_path)
                all_results.append(results)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            self.analysis_results = all_results
            self.root.after(0, self.display_multi_results, output, all_results)

        except Exception as e:
            sys.stdout = old_stdout
            import traceback
            error_detail = f"{str(e)}\n\n詳細:\n{traceback.format_exc()}"
            print(f"エラー発生:\n{error_detail}")  # コンソールにも出力
            self.root.after(0, self.display_error, error_detail)

    def display_multi_results(self, output, all_results):
        """複数画像の結果を表示"""
        self.progress.stop()
        self.progress.set(1)
        self.analyze_btn.configure(state='normal')

        # 詳細データタブに結果表示
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", self.i18n.t('gui.accuracy_eval_header').format(count=len(all_results)))
        self.result_text.insert(tk.END, output)

        # わかりやすい解釈タブに複数結果を表示
        self.interpretation_text.delete("1.0", tk.END)
        self.interpretation_text.insert("1.0", "=== 精度評価（元画像 vs AI処理結果） ===\n\n")
        self.interpretation_text.insert(tk.END, "各AI処理結果を元画像（GT）と比較し、精度を評価しています。\n\n")

        for idx, results in enumerate(all_results, 1):
            img_num = results.get('image_number', idx)
            img_name = results.get('image_name', f'画像{img_num}')

            self.interpretation_text.insert(tk.END, f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.interpretation_text.insert(tk.END, f"[IMG] 画像 {img_num}: {img_name}\n")
            self.interpretation_text.insert(tk.END, f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

            if results and 'interpretation' in results:
                from result_interpreter import format_interpretation_text
                interpretation_text = format_interpretation_text(results['interpretation'])
                self.interpretation_text.insert(tk.END, interpretation_text)
                self.interpretation_text.insert(tk.END, "\n\n")

        # ステータス更新
        self.status_label.configure(text=self.i18n.t('gui.status_analysis_complete').format(count=len(all_results)), text_color="#00ff88")

        output_folder = self.output_dir.get() or 'analysis_results'
        messagebox.showinfo(
            self.i18n.t('gui.complete_title'),
            self.i18n.t('gui.analysis_complete_multi_message').format(count=len(all_results), folder=output_folder)
        )

    def display_results(self, output, results):
        """単一結果表示（互換性のため残す）"""
        self.progress.stop()
        self.progress.set(1)
        self.analyze_btn.configure(state='normal')

        # 詳細データタブに結果表示
        self.result_text.insert("1.0", output)

        # わかりやすい解釈タブに表示
        if results and 'interpretation' in results:
            from result_interpreter import format_interpretation_text
            interpretation_text = format_interpretation_text(results['interpretation'])
            self.interpretation_text.insert("1.0", interpretation_text)
            # フォント確認デバッグ
            if hasattr(self.interpretation_text, '_textbox'):
                import tkinter.font as tkfont
                actual_font_name = self.interpretation_text._textbox.cget('font')
                print(f"DEBUG: Font name configured: {actual_font_name}")

                # 実際のフォントオブジェクトを取得
                try:
                    font_obj = tkfont.Font(font=actual_font_name)
                    actual_family = font_obj.actual('family')
                    actual_size = font_obj.actual('size')
                    print(f"DEBUG: Actual font family being rendered: {actual_family}")
                    print(f"DEBUG: Actual font size being rendered: {actual_size}")
                except Exception as e:
                    print(f"DEBUG: Could not get font details: {e}")

                # 強制的にメイリオを再設定（名前付きフォントとして）
                new_font = tkfont.Font(family="Meiryo", size=11)
                self.interpretation_text._textbox.configure(font=new_font)

            interp = results['interpretation']
            winner = interp['winner']
            summary_msg = interp['summary']['message']

            if winner == 'img1':
                color = "#00aaff"
            elif winner == 'img2':
                color = "#00ff88"
            else:
                color = "#ffaa00"

            self.status_label.configure(text=f"[OK] {summary_msg}", text_color=color)
        else:
            self.status_label.configure(text=self.i18n.t('gui.status_complete'), text_color="#00ff88")

        output_folder = self.output_dir.get() or 'analysis_results'
        messagebox.showinfo(
            self.i18n.t('gui.complete_title'),
            self.i18n.t('gui.analysis_complete_single_message').format(folder=output_folder)
        )

    def display_error(self, error_msg):
        self.progress.stop()
        self.progress.set(0)
        self.analyze_btn.configure(state='normal')
        self.status_label.configure(text=self.i18n.t('gui.status_error'), text_color="#ff4444")

        self.result_text.insert("1.0", self.i18n.t('gui.error_prefix').format(error=error_msg))
        messagebox.showerror(self.i18n.t('gui.error_title'), self.i18n.t('gui.error_analysis_failed').format(error=error_msg))

    def open_output_folder(self):
        output_path = self.output_dir.get() or 'analysis_results'
        if os.path.exists(output_path):
            os.startfile(output_path)
        else:
            messagebox.showwarning(self.i18n.t('gui.warning_title'), self.i18n.t('gui.warning_output_not_found').format(path=output_path))

    def clear_results(self):
        self.result_text.delete("1.0", tk.END)
        self.interpretation_text.delete("1.0", tk.END)
        self.status_label.configure(text=self.i18n.t('gui.status_cleared'), text_color="#888888")
        self.progress.set(0)
        self.analysis_results = None

    def show_comparison_report(self):
        output_base = self.output_dir.get() or 'analysis_results'
        report_path = os.path.join(output_base, 'comparison_report.png')

        if not os.path.exists(report_path):
            messagebox.showwarning(self.i18n.t('gui.warning_title'), self.i18n.t('gui.warning_no_report'))
            return

        # 新しいウィンドウで画像表示
        report_window = ctk.CTkToplevel(self.root)
        report_window.title(self.i18n.t('gui.comparison_report_title'))
        report_window.geometry("1200x800")

        # 画像読み込み
        img = Image.open(report_path)
        display_width = 1180
        display_height = 750
        img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        label = tk.Label(report_window, image=photo, bg="#0a0e27")
        label.image = photo
        label.pack(padx=10, pady=10)

    def switch_to_single_mode(self):
        """単一画像比較モードに切り替え"""
        # ボタンの色を変更
        self.single_mode_btn.configure(fg_color="#4A90E2", text_color="#FFFFFF")
        self.batch_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")
        self.academic_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")

        # 左パネルの表示切り替え
        self.batch_mode_frame.pack_forget()
        self.academic_mode_frame.pack_forget()
        self.single_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 右パネルの表示切り替え
        self.batch_right_frame.pack_forget()
        self.academic_right_frame.pack_forget()
        self.single_right_frame.pack(fill=tk.BOTH, expand=True)

    def switch_to_batch_mode(self):
        """バッチ処理モードに切り替え"""
        # ボタンの色を変更
        self.batch_mode_btn.configure(fg_color="#4A90E2", text_color="#FFFFFF")
        self.single_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")
        self.academic_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")

        # 左パネルの表示切り替え
        self.single_mode_frame.pack_forget()
        self.academic_mode_frame.pack_forget()
        self.batch_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 右パネルの表示切り替え
        self.single_right_frame.pack_forget()
        self.academic_right_frame.pack_forget()
        self.batch_right_frame.pack(fill=tk.BOTH, expand=True)

    def switch_to_academic_mode(self):
        """論文用ベンチマーク評価モードに切り替え"""
        # ボタンの色を変更
        self.academic_mode_btn.configure(fg_color="#9b59b6", text_color="#FFFFFF")
        self.single_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")
        self.batch_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")

        # 左パネルの表示切り替え
        self.single_mode_frame.pack_forget()
        self.batch_mode_frame.pack_forget()
        self.academic_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 右パネルの表示切り替え
        self.single_right_frame.pack_forget()
        self.batch_right_frame.pack_forget()
        self.academic_right_frame.pack(fill=tk.BOTH, expand=True)

    def toggle_language(self):
        """言語を切り替え"""
        # 言語を切り替え
        if self.current_language == 'ja':
            self.current_language = 'en'
            self.i18n.set_language('en')
            self.lang_button.configure(text="🇬🇧 English")
        else:
            self.current_language = 'ja'
            self.i18n.set_language('ja')
            self.lang_button.configure(text=self.i18n.t('gui.lang_japanese'))

        # UI全体を更新
        self.update_ui_language()

    def update_ui_language(self):
        """UI全体の言語を更新"""
        # ヘッダー
        self.root.title(self.i18n.t('app.title'))
        self.title_label.configure(text=self.i18n.t('app.title'))
        self.subtitle_label.configure(text=self.i18n.t('app.subtitle'))

        # メインタブボタン
        self.single_mode_btn.configure(text=f"[IMG] {self.i18n.t('tabs.single_analysis')}")
        self.batch_mode_btn.configure(text=f"[BATCH] {self.i18n.t('tabs.batch_processing')}")
        self.academic_mode_btn.configure(text=f"[ACAD] {self.i18n.t('tabs.academic_benchmark')}")

        # 主要ボタン
        # 単一画像分析
        self.analyze_btn.configure(text=f"[RUN] {self.i18n.t('buttons.analyze')}")

        # バッチ処理
        self.batch_analyze_btn.configure(text=f"[RUN] {self.i18n.t('buttons.analyze_batch')}")
        self.stats_analyze_btn.configure(text=f"[ANALYZE] {self.i18n.t('buttons.analyze_stats')}")

        # 論文用ベンチマーク評価
        self.academic_analyze_btn.configure(text=f"[RUN] {self.i18n.t('buttons.analyze_academic')}")
        self.academic_stats_analyze_btn.configure(text=f"[ANALYZE] {self.i18n.t('buttons.analyze_stats')}")

        # 評価モード（単一画像分析タブ）
        self.mode_image.configure(text=self.i18n.t('modes.image'))
        self.mode_image_desc.configure(text=f"  {self.i18n.t('modes.image_desc')}")
        self.mode_document.configure(text=self.i18n.t('modes.document'))
        self.mode_document_desc.configure(text=f"  {self.i18n.t('modes.document_desc')}")
        self.mode_developer.configure(text=self.i18n.t('modes.developer'))
        self.mode_developer_desc.configure(text=f"  {self.i18n.t('modes.developer_desc')}")

        # 評価モード（バッチ処理タブ）
        self.batch_mode_image.configure(text=self.i18n.t('modes.image'))
        self.batch_mode_image_desc.configure(text=f"  {self.i18n.t('modes.image_desc')}")
        self.batch_mode_document.configure(text=self.i18n.t('modes.document'))
        self.batch_mode_document_desc.configure(text=f"  {self.i18n.t('modes.document_desc')}")
        self.batch_mode_developer.configure(text=self.i18n.t('modes.developer'))
        self.batch_mode_developer_desc.configure(text=f"  {self.i18n.t('modes.developer_desc')}")

        # 参照ボタン（単一画像分析）
        self.original_browse_btn.configure(text=self.i18n.t('buttons.browse'))
        self.img1_browse_btn.configure(text=self.i18n.t('buttons.browse'))
        self.img2_browse_btn.configure(text=self.i18n.t('buttons.browse'))
        self.img3_browse_btn.configure(text=self.i18n.t('buttons.browse'))
        self.img4_browse_btn.configure(text=self.i18n.t('buttons.browse'))
        self.img5_browse_btn.configure(text=self.i18n.t('buttons.browse'))
        self.output_browse_btn.configure(text=self.i18n.t('buttons.browse'))

        # 参照ボタン（バッチ処理）
        self.batch_original_browse_btn.configure(text=self.i18n.t('buttons.browse'))

        # PNG警告
        self.single_png_warning.configure(text=self.i18n.t('warnings.png_recommended'))
        self.batch_png_warning.configure(text=self.i18n.t('warnings.png_recommended'))
        self.academic_png_warning.configure(text=self.i18n.t('warnings.png_recommended'))

        # アコーディオンタイトル（単一画像分析タブ）
        self.mode_accordion.update_title(self.i18n.t('sections.evaluation_settings'))
        self.original_accordion.update_title(self.i18n.t('sections.original_image_required'))
        self.img1_accordion.update_title(self.i18n.t('sections.upscaled_image_1'))
        self.img2_accordion.update_title(self.i18n.t('sections.upscaled_image_2'))
        self.img3_accordion.update_title(self.i18n.t('sections.upscaled_image_3'))
        self.img4_accordion.update_title(self.i18n.t('sections.upscaled_image_4'))
        self.img5_accordion.update_title(self.i18n.t('sections.upscaled_image_5'))
        self.output_accordion.update_title(self.i18n.t('sections.output_folder'))

        # アコーディオンタイトル（バッチ処理タブ）
        self.eval_accordion.update_title(self.i18n.t('sections.evaluation_settings'))
        self.folder_accordion.update_title(self.i18n.t('sections.folder_settings'))
        self.output_csv_accordion.update_title(self.i18n.t('sections.csv_settings'))
        self.detail_accordion.update_title(self.i18n.t('sections.detail_output'))
        self.stats_accordion.update_title(self.i18n.t('sections.stats_analysis'))

        # アコーディオンタイトル（論文用タブ）
        self.workflow_accordion.update_title(self.i18n.t('help.workflow_title'))
        self.bicubic_accordion.update_title(self.i18n.t('sections.bicubic_info'))
        self.config_accordion.update_title(self.i18n.t('sections.evaluation_settings'))
        self.academic_stats_accordion.update_title(self.i18n.t('sections.stats_analysis'))

        # プレビュー画像のタイトル（画像が選択されていない場合のみ更新）
        if hasattr(self, 'img_before_title'):
            self.img_before_title.configure(text=self.i18n.t('gui.original_before'))
        if hasattr(self, 'img1_title'):
            self.img1_title.configure(text=self.i18n.t('gui.sr_result_1'))
        if hasattr(self, 'img2_title'):
            self.img2_title.configure(text=self.i18n.t('gui.sr_result_2'))

        # プレビュープレースホルダー（画像が読み込まれていない場合）
        if hasattr(self, 'preview_img_before_label') and not hasattr(self.preview_img_before_label, 'image'):
            self.preview_img_before_label.configure(text=self.i18n.t('gui.select_original_prompt'))
        if hasattr(self, 'preview_img1_label') and not hasattr(self.preview_img1_label, 'image'):
            self.preview_img1_label.configure(text=self.i18n.t('gui.select_sr1_prompt'))
        if hasattr(self, 'preview_img2_label') and not hasattr(self.preview_img2_label, 'image'):
            self.preview_img2_label.configure(text=self.i18n.t('gui.select_sr2_prompt'))

        # バッチ処理進捗エリア
        if hasattr(self, 'batch_progress_title'):
            self.batch_progress_title.configure(text=self.i18n.t('gui.batch_progress_title'))
        if hasattr(self, 'batch_status_label'):
            # 処理中でない場合のみ更新
            if self.batch_analyze_btn.cget('state') != 'disabled':
                self.batch_status_label.configure(text=self.i18n.t('gui.batch_start_prompt'))
        if hasattr(self, 'batch_result_label'):
            self.batch_result_label.configure(text=self.i18n.t('gui.batch_log_title'))

        # アカデミック評価進捗エリア
        if hasattr(self, 'academic_progress_title'):
            self.academic_progress_title.configure(text=self.i18n.t('gui.academic_progress_title'))
        if hasattr(self, 'academic_status_label'):
            # 処理中でない場合のみ更新
            if self.academic_analyze_btn.cget('state') != 'disabled':
                self.academic_status_label.configure(text=self.i18n.t('gui.academic_start_prompt'))
        if hasattr(self, 'academic_result_label'):
            self.academic_result_label.configure(text=self.i18n.t('gui.academic_log_title'))

        # バッチモードのタイトルと説明
        if hasattr(self, 'batch_info_title'):
            self.batch_info_title.configure(text=self.i18n.t('batch.title'))
        if hasattr(self, 'batch_info_text'):
            self.batch_info_text.configure(text=self.i18n.t('batch.description'))
        if hasattr(self, 'batch_original_label'):
            self.batch_original_label.configure(text=self.i18n.t('batch.original_folder_label'))
        if hasattr(self, 'batch_upscaled_label'):
            self.batch_upscaled_label.configure(text=self.i18n.t('batch.model_folder_label'))

        # アカデミックモードのタイトルと説明
        if hasattr(self, 'academic_info_title'):
            self.academic_info_title.configure(text=self.i18n.t('academic.title'))
        if hasattr(self, 'academic_info_text'):
            self.academic_info_text.configure(text=self.i18n.t('academic.description'))

        # パッチサイズセクション（単一画像分析タブ）
        if hasattr(self, 'patch_label'):
            self.patch_label.configure(text=self.i18n.t('gui.patch_size_title'))
        if hasattr(self, 'patch_8'):
            self.patch_8.configure(text=self.i18n.t('gui.patch_8x8'))
        if hasattr(self, 'patch_8_desc'):
            self.patch_8_desc.configure(text=self.i18n.t('gui.patch_8x8_detail'))
        if hasattr(self, 'patch_16'):
            self.patch_16.configure(text=self.i18n.t('gui.patch_16x16'))
        if hasattr(self, 'patch_16_desc'):
            self.patch_16_desc.configure(text=self.i18n.t('gui.patch_16x16_detail'))
        if hasattr(self, 'patch_32'):
            self.patch_32.configure(text=self.i18n.t('gui.patch_32x32'))
        if hasattr(self, 'patch_32_desc'):
            self.patch_32_desc.configure(text=self.i18n.t('gui.patch_32x32_detail'))

        # パッチサイズセクション（バッチ処理タブ）
        if hasattr(self, 'batch_patch_info'):
            self.batch_patch_info.configure(text=self.i18n.t('batch.patch_size_title'))
        if hasattr(self, 'batch_patch_desc'):
            self.batch_patch_desc.configure(text=self.i18n.t('batch.patch_size_desc'))
        if hasattr(self, 'batch_patch_8'):
            self.batch_patch_8.configure(text=self.i18n.t('batch.patch_8x8'))
        if hasattr(self, 'batch_patch_16'):
            self.batch_patch_16.configure(text=self.i18n.t('batch.patch_16x16'))
        if hasattr(self, 'batch_patch_32'):
            self.batch_patch_32.configure(text=self.i18n.t('batch.patch_32x32'))

        # パッチサイズセクション（アカデミック評価タブ）
        if hasattr(self, 'academic_patch_info'):
            self.academic_patch_info.configure(text=self.i18n.t('academic.patch_size_title'))
        if hasattr(self, 'academic_patch_desc'):
            self.academic_patch_desc.configure(text=self.i18n.t('academic.patch_size_desc'))
        if hasattr(self, 'academic_patch_8'):
            self.academic_patch_8.configure(text=self.i18n.t('academic.patch_8x8'))
        if hasattr(self, 'academic_patch_16'):
            self.academic_patch_16.configure(text=self.i18n.t('academic.patch_16x16'))
        if hasattr(self, 'academic_patch_32'):
            self.academic_patch_32.configure(text=self.i18n.t('academic.patch_32x32'))

def main():
    root = ctk.CTk()
    app = ModernImageAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
