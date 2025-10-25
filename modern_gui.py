import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
from advanced_image_analyzer import analyze_images
import json
from PIL import Image, ImageTk, ImageDraw
import customtkinter as ctk

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

class ModernImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Analyzer Pro")
        self.root.geometry("1200x800")

        # 背景色設定（サイバーパンク風）
        self.bg_color = "#0a0e27"
        self.accent_color = "#00ffff"
        self.secondary_color = "#1e2740"

        # 変数
        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()
        self.original_path = tk.StringVar()  # 元画像（オプション）
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

        # タイトル
        title_label = ctk.CTkLabel(
            header_frame,
            text="AI Image Analyzer Pro",
            font=("Arial", 32, "bold"),
            text_color="#4A90E2"
        )
        title_label.place(x=130, y=25)

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="高解像度画像品質分析システム",
            font=("Arial", 14),
            text_color="#888888"
        )
        subtitle_label.place(x=130, y=70)

        # システムモニター（右上）
        if MONITORING_AVAILABLE:
            monitor_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
            monitor_frame.place(x=850, y=15)

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

        # モード切り替えボタンエリア
        mode_frame = ctk.CTkFrame(main_container, fg_color="#1e2740", height=60, corner_radius=0)
        mode_frame.pack(fill=tk.X, padx=0, pady=0)
        mode_frame.pack_propagate(False)

        # モード切り替えボタン
        button_container = ctk.CTkFrame(mode_frame, fg_color="transparent")
        button_container.place(relx=0.5, rely=0.5, anchor="center")

        self.single_mode_btn = ctk.CTkButton(
            button_container,
            text="📸 単一画像比較",
            command=self.switch_to_single_mode,
            height=40,
            width=200,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        self.single_mode_btn.pack(side=tk.LEFT, padx=10)

        self.batch_mode_btn = ctk.CTkButton(
            button_container,
            text="🔬 バッチ処理",
            command=self.switch_to_batch_mode,
            height=40,
            width=200,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#4a5568",
            text_color="#ffffff",
            hover_color="#2d3748"
        )
        self.batch_mode_btn.pack(side=tk.LEFT, padx=10)

        # コンテンツエリア（2カラムレイアウト）
        content_frame = ctk.CTkFrame(main_container, fg_color="#0a0e27")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 左側パネル（入力エリア）
        self.left_panel = ctk.CTkFrame(content_frame, fg_color="#1e2740", width=480, corner_radius=15)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)

        # スクロール可能なフレーム（単一モード用）
        self.single_mode_frame = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self.single_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # バッチモード用フレーム（後で作成）
        self.batch_mode_frame = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")

        # 右側パネル（画像比較・結果表示エリア）
        self.right_panel = ctk.CTkFrame(content_frame, fg_color="#1e2740", corner_radius=15)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 単一画像比較モードのUI ===
        # 画像選択セクション
        input_section = ctk.CTkFrame(self.single_mode_frame, fg_color="transparent")
        input_section.pack(fill=tk.X)

        # 評価モード選択
        mode_frame = ctk.CTkFrame(input_section, fg_color="#1e2740", corner_radius=10)
        mode_frame.pack(fill=tk.X, pady=(0, 20))

        mode_title = ctk.CTkLabel(
            mode_frame,
            text="📊 評価モード",
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        mode_title.pack(anchor="w", padx=15, pady=(15, 10))

        # 評価モード変数
        self.evaluation_mode = tk.StringVar(value="image")

        # 画像モード
        mode_image = ctk.CTkRadioButton(
            mode_frame,
            text="画像（レントゲン、内視鏡、写真など）",
            variable=self.evaluation_mode,
            value="image",
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        mode_image.pack(anchor="w", padx=30, pady=(0, 8))

        mode_image_desc = ctk.CTkLabel(
            mode_frame,
            text="  └─ CLIP基準: 0.70、全指標使用、診断テキスト自動検出",
            font=("Arial", 10),
            text_color="#888888"
        )
        mode_image_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # 文書モード
        mode_document = ctk.CTkRadioButton(
            mode_frame,
            text="文書（医療カルテ、契約書、レシートなど）",
            variable=self.evaluation_mode,
            value="document",
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        mode_document.pack(anchor="w", padx=30, pady=(0, 8))

        mode_document_desc = ctk.CTkLabel(
            mode_frame,
            text="  └─ CLIP基準: 0.90（厳格）、テキストMAE重視",
            font=("Arial", 10),
            text_color="#888888"
        )
        mode_document_desc.pack(anchor="w", padx=30, pady=(0, 10))

        # 開発者モード
        mode_developer = ctk.CTkRadioButton(
            mode_frame,
            text="開発者モード（バグテスト・デバッグ用）",
            variable=self.evaluation_mode,
            value="developer",
            font=("Arial", 12),
            text_color="#ffffff",
            fg_color="#ffa500",
            hover_color="#cc8400"
        )
        mode_developer.pack(anchor="w", padx=30, pady=(0, 8))

        mode_developer_desc = ctk.CTkLabel(
            mode_frame,
            text="  └─ 評価不能判定なし、すべての警告を表示",
            font=("Arial", 10),
            text_color="#888888"
        )
        mode_developer_desc.pack(anchor="w", padx=30, pady=(0, 15))

        # 画像1
        img1_label = ctk.CTkLabel(
            input_section,
            text="📸 画像 1",
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        img1_label.pack(anchor="w", pady=(0, 10))

        img1_entry = ctk.CTkEntry(
            input_section,
            textvariable=self.img1_path,
            placeholder_text="画像ファイルを選択...",
            height=40,
            corner_radius=10,
            font=("Arial", 11)
        )
        img1_entry.pack(fill=tk.X, pady=(0, 10))

        img1_btn = ctk.CTkButton(
            input_section,
            text="参照",
            command=self.browse_image1,
            height=40,
            corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        img1_btn.pack(fill=tk.X, pady=(0, 20))

        # 画像2
        img2_label = ctk.CTkLabel(
            input_section,
            text="📸 画像 2",
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        img2_label.pack(anchor="w", pady=(0, 10))

        img2_entry = ctk.CTkEntry(
            input_section,
            textvariable=self.img2_path,
            placeholder_text="画像ファイルを選択...",
            height=40,
            corner_radius=10,
            font=("Arial", 11)
        )
        img2_entry.pack(fill=tk.X, pady=(0, 10))

        img2_btn = ctk.CTkButton(
            input_section,
            text="参照",
            command=self.browse_image2,
            height=40,
            corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        img2_btn.pack(fill=tk.X, pady=(0, 20))

        # 元画像（オプション）
        original_label = ctk.CTkLabel(
            input_section,
            text="🎯 元画像（オプション）",
            font=("Arial", 16, "bold"),
            text_color="#ffa500"
        )
        original_label.pack(anchor="w", pady=(0, 10))

        original_sublabel = ctk.CTkLabel(
            input_section,
            text="※ AI超解像の精度評価用（低解像度画像）",
            font=("Arial", 10),
            text_color="#888888"
        )
        original_sublabel.pack(anchor="w", pady=(0, 5))

        original_entry = ctk.CTkEntry(
            input_section,
            textvariable=self.original_path,
            placeholder_text="元画像を選択（省略可）...",
            height=40,
            corner_radius=10,
            font=("Arial", 11)
        )
        original_entry.pack(fill=tk.X, pady=(0, 10))

        original_btn_frame = ctk.CTkFrame(input_section, fg_color="transparent")
        original_btn_frame.pack(fill=tk.X, pady=(0, 20))

        original_btn = ctk.CTkButton(
            original_btn_frame,
            text="参照",
            command=self.browse_original,
            height=40,
            width=200,
            corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#ffa500",
            text_color="#000000",
            hover_color="#cc8400"
        )
        original_btn.pack(side=tk.LEFT, padx=(0, 10))

        clear_original_btn = ctk.CTkButton(
            original_btn_frame,
            text="クリア",
            command=lambda: self.original_path.set(""),
            height=40,
            width=100,
            corner_radius=10,
            font=("Arial", 12),
            fg_color="#555555",
            text_color="#ffffff",
            hover_color="#777777"
        )
        clear_original_btn.pack(side=tk.LEFT)

        # 出力フォルダ
        output_label = ctk.CTkLabel(
            input_section,
            text="💾 出力フォルダ",
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        output_label.pack(anchor="w", pady=(0, 10))

        output_entry = ctk.CTkEntry(
            input_section,
            textvariable=self.output_dir,
            height=40,
            corner_radius=10,
            font=("Arial", 11)
        )
        output_entry.pack(fill=tk.X, pady=(0, 10))

        output_btn = ctk.CTkButton(
            input_section,
            text="参照",
            command=self.browse_output,
            height=40,
            corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        output_btn.pack(fill=tk.X, pady=(0, 30))

        # 分析開始ボタン（大きく目立つ）
        self.analyze_btn = ctk.CTkButton(
            input_section,
            text="🚀 分析開始",
            command=self.start_analysis,
            height=60,
            corner_radius=15,
            font=("Arial", 18, "bold"),
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
            text="画像を選択して分析を開始してください",
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
            text="📊 レポート",
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
            text="📁 フォルダ",
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
            text="🗑️ クリア",
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
            text="📸 画像比較プレビュー",
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

        img_before_title = ctk.CTkLabel(
            img_before_container,
            text="📄 元画像 (Before)",
            font=("Arial", 12, "bold"),
            text_color="#FFA500"
        )
        img_before_title.pack(pady=(0, 5))

        self.preview_img_before_label = tk.Label(
            img_before_container,
            bg="#0a0e27",
            text="元画像を選択してください",
            fg="#888888",
            font=("Arial", 10)
        )
        self.preview_img_before_label.pack(fill=tk.BOTH, expand=True)

        # 画像1（超解像結果1 - After）
        img1_container = ctk.CTkFrame(image_compare_frame, fg_color="transparent")
        img1_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        img1_title = ctk.CTkLabel(
            img1_container,
            text="🎨 超解像結果1 (After)",
            font=("Arial", 12, "bold"),
            text_color="#00ff88"
        )
        img1_title.pack(pady=(0, 5))

        self.preview_img1_label = tk.Label(
            img1_container,
            bg="#0a0e27",
            text="超解像結果1を選択してください",
            fg="#888888",
            font=("Arial", 10)
        )
        self.preview_img1_label.pack(fill=tk.BOTH, expand=True)

        # 画像2（超解像結果2 - After）
        img2_container = ctk.CTkFrame(image_compare_frame, fg_color="transparent")
        img2_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        img2_title = ctk.CTkLabel(
            img2_container,
            text="🎨 超解像結果2 (After)",
            font=("Arial", 12, "bold"),
            text_color="#00ff88"
        )
        img2_title.pack(pady=(0, 5))

        self.preview_img2_label = tk.Label(
            img2_container,
            bg="#0a0e27",
            text="超解像結果2を選択してください",
            fg="#888888",
            font=("Arial", 10)
        )
        self.preview_img2_label.pack(fill=tk.BOTH, expand=True)

        # タブビュー（結果表示）
        self.tabview = ctk.CTkTabview(
            self.single_right_frame,
            corner_radius=15,
            fg_color="#1e2740",
            segmented_button_fg_color="#2d3748",
            segmented_button_selected_color="#00ffff",
            segmented_button_selected_hover_color="#00cccc",
            text_color="#ffffff"
        )
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # タブ作成（単一モード用）
        self.tabview.add("📊 わかりやすい解釈")
        self.tabview.add("📝 詳細データ")

        # わかりやすい解釈タブ
        self.interpretation_text = ctk.CTkTextbox(
            self.tabview.tab("📊 わかりやすい解釈"),
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#4A90E2",
            corner_radius=10
        )
        self.interpretation_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 詳細データタブ
        self.result_text = ctk.CTkTextbox(
            self.tabview.tab("📝 詳細データ"),
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === バッチモード用の右パネル ===
        self.batch_right_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")

        # バッチ処理進捗エリア
        batch_progress_title = ctk.CTkLabel(
            self.batch_right_frame,
            text="📊 バッチ処理進捗",
            font=("Arial", 18, "bold"),
            text_color="#4A90E2"
        )
        batch_progress_title.pack(pady=(20, 10))

        # 進捗表示フレーム
        self.batch_progress_frame = ctk.CTkFrame(self.batch_right_frame, fg_color="#0a0e27", corner_radius=10)
        self.batch_progress_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        self.batch_status_label = ctk.CTkLabel(
            self.batch_progress_frame,
            text="バッチ処理を開始してください",
            font=("Arial", 14),
            text_color="#888888"
        )
        self.batch_status_label.pack(pady=20)

        # プログレスバー（既存のものを使用）
        self.batch_progress = ctk.CTkProgressBar(
            self.batch_progress_frame,
            width=400,
            height=20,
            corner_radius=10,
            fg_color="#2d3748",
            progress_color="#00ffff"
        )
        self.batch_progress.pack(pady=(0, 20), padx=20)
        self.batch_progress.set(0)

        # 結果表示テキストエリア
        batch_result_label = ctk.CTkLabel(
            self.batch_right_frame,
            text="📝 処理結果ログ",
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        batch_result_label.pack(pady=(10, 5), padx=15, anchor="w")

        self.batch_result_text = ctk.CTkTextbox(
            self.batch_right_frame,
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10
        )
        self.batch_result_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # バッチモード用のUIを作成（左パネル）
        self.create_batch_mode_ui()

    def create_batch_mode_ui(self):
        """バッチ処理モードのUI作成（左パネル）"""

        # 説明セクション
        info_frame = ctk.CTkFrame(self.batch_mode_frame, fg_color="#2d3748", corner_radius=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))

        info_title = ctk.CTkLabel(
            info_frame,
            text="📚 バッチ処理について",
            font=("Arial", 16, "bold"),
            text_color="#4A90E2"
        )
        info_title.pack(anchor="w", padx=15, pady=(15, 5))

        info_text = ctk.CTkLabel(
            info_frame,
            text="大量の画像ペア（300枚以上）を自動で分析し、統計的に妥当な閾値を決定します。\n"
                 "医療画像研究・AIモデル比較に最適です。",
            font=("Arial", 11),
            text_color="#cccccc",
            justify="left"
        )
        info_text.pack(anchor="w", padx=15, pady=(0, 15))

        # 設定セクション
        config_frame = ctk.CTkFrame(self.batch_mode_frame, fg_color="#1e2740", corner_radius=10)
        config_frame.pack(fill=tk.X, pady=(0, 15))

        config_title = ctk.CTkLabel(
            config_frame,
            text="⚙️ バッチ処理設定",
            font=("Arial", 14, "bold"),
            text_color="#4A90E2"
        )
        config_title.pack(anchor="w", padx=15, pady=(15, 10))

        # 元画像フォルダ
        self.batch_original_dir = tk.StringVar()
        original_label = ctk.CTkLabel(
            config_frame,
            text="📁 元画像フォルダ（低解像度）",
            font=("Arial", 12, "bold"),
            text_color="#ffffff"
        )
        original_label.pack(anchor="w", padx=15, pady=(10, 5))

        original_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        original_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        original_entry = ctk.CTkEntry(
            original_frame,
            textvariable=self.batch_original_dir,
            placeholder_text="dataset/original/",
            height=35,
            font=("Arial", 11)
        )
        original_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        original_btn = ctk.CTkButton(
            original_frame,
            text="参照",
            command=self.browse_batch_original,
            width=80,
            height=35,
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        original_btn.pack(side=tk.RIGHT)

        # 超解像モデルフォルダ（複数）
        upscaled_label = ctk.CTkLabel(
            config_frame,
            text="🤖 超解像モデルフォルダ（最大5個まで）",
            font=("Arial", 12, "bold"),
            text_color="#ffffff"
        )
        upscaled_label.pack(anchor="w", padx=15, pady=(10, 5))

        # モデルフォルダ入力欄（5個）
        self.batch_model_vars = []
        self.batch_model_name_vars = []

        for i in range(5):
            model_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
            model_frame.pack(fill=tk.X, padx=15, pady=5)

            model_name_var = tk.StringVar(value=f"model{i+1}")
            model_path_var = tk.StringVar()

            self.batch_model_name_vars.append(model_name_var)
            self.batch_model_vars.append(model_path_var)

            # モデル名入力
            name_entry = ctk.CTkEntry(
                model_frame,
                textvariable=model_name_var,
                placeholder_text=f"モデル{i+1}名",
                width=120,
                height=30,
                font=("Arial", 10)
            )
            name_entry.pack(side=tk.LEFT, padx=(0, 5))

            # パス入力
            path_entry = ctk.CTkEntry(
                model_frame,
                textvariable=model_path_var,
                placeholder_text=f"dataset/upscayl_model{i+1}/",
                height=30,
                font=("Arial", 10)
            )
            path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

            # 参照ボタン
            browse_btn = ctk.CTkButton(
                model_frame,
                text="📁",
                command=lambda idx=i: self.browse_batch_model(idx),
                width=40,
                height=30,
                fg_color="#555555",
                hover_color="#777777"
            )
            browse_btn.pack(side=tk.RIGHT)

        # 出力先設定
        output_label = ctk.CTkLabel(
            config_frame,
            text="💾 出力設定",
            font=("Arial", 12, "bold"),
            text_color="#ffffff"
        )
        output_label.pack(anchor="w", padx=15, pady=(15, 5))

        self.batch_output_csv = tk.StringVar(value="results/batch_analysis.csv")
        self.batch_output_detail = tk.StringVar(value="results/detailed/")
        self.batch_limit = tk.IntVar(value=0)  # 0 = 全て
        self.batch_append_mode = tk.BooleanVar(value=True)  # True = 追加（デフォルト）, False = 上書き

        csv_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        csv_frame.pack(fill=tk.X, padx=15, pady=5)

        csv_label = ctk.CTkLabel(csv_frame, text="CSV:", width=80, anchor="w", font=("Arial", 10))
        csv_label.pack(side=tk.LEFT)

        csv_entry = ctk.CTkEntry(
            csv_frame,
            textvariable=self.batch_output_csv,
            height=30,
            font=("Arial", 10)
        )
        csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        csv_browse_btn = ctk.CTkButton(
            csv_frame,
            text="📁",
            command=self.browse_batch_csv_output,
            width=40,
            height=30,
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        csv_browse_btn.pack(side=tk.RIGHT)

        detail_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        detail_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        detail_label = ctk.CTkLabel(detail_frame, text="詳細:", width=80, anchor="w", font=("Arial", 10))
        detail_label.pack(side=tk.LEFT)

        detail_entry = ctk.CTkEntry(
            detail_frame,
            textvariable=self.batch_output_detail,
            height=30,
            font=("Arial", 10)
        )
        detail_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        detail_browse_btn = ctk.CTkButton(
            detail_frame,
            text="📁",
            command=self.browse_batch_detail_output,
            width=40,
            height=30,
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        detail_browse_btn.pack(side=tk.RIGHT)

        # 追加モード選択チェックボックス
        append_mode_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        append_mode_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        append_checkbox = ctk.CTkCheckBox(
            append_mode_frame,
            text="既存CSVにデータを追加（チェックなし = 上書きモード）",
            variable=self.batch_append_mode,
            font=("Arial", 11),
            text_color="#4A90E2",
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        append_checkbox.pack(anchor="w")

        # 分割実行設定
        limit_label = ctk.CTkLabel(
            config_frame,
            text="🔢 分割実行（テスト用）",
            font=("Arial", 12, "bold"),
            text_color="#ffffff"
        )
        limit_label.pack(anchor="w", padx=15, pady=(15, 5))

        limit_info = ctk.CTkLabel(
            config_frame,
            text="※ 0 = 全画像処理、10 = 最初の10枚のみ処理（テスト用）",
            font=("Arial", 9),
            text_color="#888888",
            justify="left"
        )
        limit_info.pack(anchor="w", padx=15, pady=(0, 5))

        # 処理枚数制限フレーム（縦に2段構成）
        limit_container = ctk.CTkFrame(config_frame, fg_color="transparent")
        limit_container.pack(fill=tk.X, padx=15, pady=(0, 15))

        # タイトル
        limit_title = ctk.CTkLabel(
            limit_container,
            text="処理枚数制限:",
            font=("Arial", 11, "bold"),
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
            text="全て",
            font=("Arial", 13, "bold"),
            text_color="#00ff88",
            width=80
        )
        self.limit_value_label.pack(side=tk.LEFT)

        # 第2段：直接入力
        entry_frame = ctk.CTkFrame(limit_container, fg_color="transparent")
        entry_frame.pack(fill=tk.X)

        entry_label = ctk.CTkLabel(
            entry_frame,
            text="直接入力（大量処理用）:",
            font=("Arial", 10),
            text_color="#888888",
            anchor="w"
        )
        entry_label.pack(side=tk.LEFT, padx=(0, 10))

        self.limit_entry = ctk.CTkEntry(
            entry_frame,
            width=120,
            height=35,
            font=("Arial", 13),
            placeholder_text="0 = 全て処理",
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

        # 実行ボタン
        self.batch_analyze_btn = ctk.CTkButton(
            self.batch_mode_frame,
            text="🚀 バッチ処理開始",
            command=self.start_batch_analysis,
            height=50,
            corner_radius=10,
            font=("Arial", 16, "bold"),
            fg_color="#00ff88",
            text_color="#000000",
            hover_color="#00dd77"
        )
        self.batch_analyze_btn.pack(fill=tk.X, pady=(0, 15))

        # プログレスバー
        self.batch_progress = ctk.CTkProgressBar(
            self.batch_mode_frame,
            height=20,
            corner_radius=10,
            progress_color="#00ffff"
        )
        self.batch_progress.pack(fill=tk.X, pady=(0, 10))
        self.batch_progress.set(0)

        # ステータス
        self.batch_status_label = ctk.CTkLabel(
            self.batch_mode_frame,
            text="設定を入力してバッチ処理を開始してください",
            font=("Arial", 11),
            text_color="#888888",
            wraplength=600
        )
        self.batch_status_label.pack(pady=(0, 15))

        # 統計分析セクション
        stats_frame = ctk.CTkFrame(self.batch_mode_frame, fg_color="#1e2740", corner_radius=10)
        stats_frame.pack(fill=tk.X, pady=(0, 15))

        stats_title = ctk.CTkLabel(
            stats_frame,
            text="📊 統計分析・プロット生成",
            font=("Arial", 14, "bold"),
            text_color="#4A90E2"
        )
        stats_title.pack(anchor="w", padx=15, pady=(15, 10))

        stats_info = ctk.CTkLabel(
            stats_frame,
            text="バッチ処理完了後、CSVファイルを統計分析して23種類の研究用プロットを生成します。",
            font=("Arial", 11),
            text_color="#cccccc",
            justify="left"
        )
        stats_info.pack(anchor="w", padx=15, pady=(0, 10))

        # CSV選択
        self.stats_csv_path = tk.StringVar()

        csv_select_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        csv_select_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        csv_select_entry = ctk.CTkEntry(
            csv_select_frame,
            textvariable=self.stats_csv_path,
            placeholder_text="results/batch_analysis.csv を選択...",
            height=35,
            font=("Arial", 11)
        )
        csv_select_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        csv_select_btn = ctk.CTkButton(
            csv_select_frame,
            text="📁 CSV選択",
            command=self.browse_stats_csv,
            width=100,
            height=35,
            fg_color="#4A90E2",
            text_color="#FFFFFF",
            hover_color="#357ABD"
        )
        csv_select_btn.pack(side=tk.RIGHT)

        # ボタン配置用フレーム
        button_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        button_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 統計分析実行ボタン
        self.stats_analyze_btn = ctk.CTkButton(
            button_frame,
            text="📈 統計分析＋プロット生成（23種類）",
            command=self.start_stats_analysis,
            height=50,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#ffa500",
            text_color="#000000",
            hover_color="#cc8400"
        )
        self.stats_analyze_btn.pack(fill=tk.X, pady=(0, 10))

        # ハルシネーション抽出ボタン
        self.hallucination_extract_btn = ctk.CTkButton(
            button_frame,
            text="⚠️ ハルシネーション疑いデータ抽出",
            command=self.extract_hallucination_suspects,
            height=40,
            corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#ff4444",
            text_color="#ffffff",
            hover_color="#cc3333"
        )
        self.hallucination_extract_btn.pack(fill=tk.X, pady=(0, 5))

        # クリーンデータセット抽出ボタン（NEW in v1.5）
        self.clean_dataset_btn = ctk.CTkButton(
            button_frame,
            text="✨ 正常データ抽出（AI学習用）",
            command=self.extract_clean_dataset,
            height=40,
            corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#44ff44",
            text_color="#000000",
            hover_color="#33cc33"
        )
        self.clean_dataset_btn.pack(fill=tk.X)

        # 結果表示エリア
        self.batch_result_text = ctk.CTkTextbox(
            self.batch_mode_frame,
            font=("Meiryo", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10,
            height=200
        )
        self.batch_result_text.pack(fill=tk.BOTH, expand=True)

    def browse_batch_original(self):
        dirname = filedialog.askdirectory(title="元画像フォルダを選択")
        if dirname:
            self.batch_original_dir.set(dirname)

    def browse_batch_model(self, index):
        dirname = filedialog.askdirectory(title=f"モデル{index+1}のフォルダを選択")
        if dirname:
            self.batch_model_vars[index].set(dirname)

    def browse_batch_csv_output(self):
        """CSV出力先選択"""
        filename = filedialog.asksaveasfilename(
            title="CSV出力先を選択",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("すべてのファイル", "*.*")],
            initialfile="batch_analysis.csv"
        )
        if filename:
            self.batch_output_csv.set(filename)

    def browse_batch_detail_output(self):
        """詳細レポート出力先フォルダ選択"""
        dirname = filedialog.askdirectory(title="詳細レポート出力先フォルダを選択")
        if dirname:
            self.batch_output_detail.set(dirname)

    def browse_stats_csv(self):
        filename = filedialog.askopenfilename(
            title="CSVファイルを選択",
            filetypes=[("CSV", "*.csv"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.stats_csv_path.set(filename)

    def on_slider_change(self, value):
        """スライダー変更時のコールバック"""
        int_value = int(value)
        self.batch_limit.set(int_value)

        # 数値入力フィールドも更新
        self.limit_entry.delete(0, tk.END)
        self.limit_entry.insert(0, str(int_value))

        # ラベル更新
        if int_value == 0:
            self.limit_value_label.configure(text="全て", text_color="#00ff88")
        else:
            self.limit_value_label.configure(text=f"{int_value}枚", text_color="#00ffff")

    def on_entry_typing(self, event=None):
        """入力中のリアルタイムフィードバック"""
        try:
            value = self.limit_entry.get().strip()
            if value == "" or value == "0":
                self.limit_value_label.configure(text="全て", text_color="#00ff88")
            else:
                int_value = int(value)
                if int_value > 0:
                    self.limit_value_label.configure(text=f"{int_value}枚", text_color="#00ffff")
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
                self.limit_value_label.configure(text="全て", text_color="#00ff88")
            else:
                self.limit_value_label.configure(text=f"{int_value}枚", text_color="#00ffff")

        except ValueError:
            # 無効な入力の場合は0にリセット
            self.limit_entry.delete(0, tk.END)
            self.limit_entry.insert(0, "0")
            self.batch_limit.set(0)
            self.limit_value_label.configure(text="全て", text_color="#00ff88")

    def update_limit_label(self, *args):
        """処理枚数ラベル更新（trace用）"""
        limit = self.batch_limit.get()
        if limit == 0:
            self.limit_value_label.configure(text="全て", text_color="#00ff88")
        else:
            self.limit_value_label.configure(text=f"{limit}枚", text_color="#00ffff")

    def start_batch_analysis(self):
        """バッチ処理開始"""
        # バリデーション
        if not self.batch_original_dir.get():
            messagebox.showerror("エラー", "元画像フォルダを選択してください")
            return

        # 有効なモデルフォルダをカウント
        valid_models = {}
        for i in range(5):
            model_name = self.batch_model_name_vars[i].get().strip()
            model_path = self.batch_model_vars[i].get().strip()

            if model_path:
                if not model_name:
                    messagebox.showerror("エラー", f"モデル{i+1}の名前を入力してください")
                    return
                valid_models[model_name] = model_path

        if len(valid_models) == 0:
            messagebox.showerror("エラー", "少なくとも1つの超解像モデルフォルダを選択してください")
            return

        # 設定ファイル作成
        config = {
            "original_dir": self.batch_original_dir.get(),
            "upscaled_dirs": valid_models,
            "output_csv": self.batch_output_csv.get(),
            "output_detail_dir": self.batch_output_detail.get(),
            "limit": self.batch_limit.get(),  # 処理枚数制限
            "append_mode": self.batch_append_mode.get()  # 追加モード
        }

        # UIを無効化
        self.batch_analyze_btn.configure(state='disabled')
        self.batch_progress.set(0)
        self.batch_status_label.configure(text="バッチ処理を開始します...", text_color="#00ffff")
        self.batch_result_text.delete("1.0", tk.END)

        # 別スレッドで実行
        thread = threading.Thread(target=self.run_batch_analysis, args=(config,))
        thread.daemon = True
        thread.start()

    def update_batch_progress(self, current, total, message):
        """バッチ処理進捗更新（別スレッドから呼ばれる）"""
        progress = current / total if total > 0 else 0
        self.root.after(0, lambda: self.batch_progress.set(progress))
        self.root.after(0, lambda: self.batch_status_label.configure(
            text=f"処理中: {current}/{total} - {message}",
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
                text=f"✅ バッチ処理完了！CSVファイル: {csv_path}",
                text_color="#00ff88"
            )

            # CSVパスを統計分析欄に自動入力
            if csv_path:
                self.stats_csv_path.set(csv_path)

            messagebox.showinfo(
                "完了",
                f"バッチ処理が完了しました。\n\n"
                f"CSV: {csv_path}\n\n"
                f"統計分析を実行して23種類のプロットを生成できます。"
            )
        else:
            self.batch_status_label.configure(
                text="❌ バッチ処理エラー",
                text_color="#ff4444"
            )
            messagebox.showerror("エラー", f"バッチ処理中にエラーが発生しました:\n{output}")

    def start_stats_analysis(self):
        """統計分析開始"""
        csv_path = self.stats_csv_path.get()

        if not csv_path:
            messagebox.showerror("エラー", "CSVファイルを選択してください")
            return

        if not os.path.exists(csv_path):
            messagebox.showerror("エラー", f"CSVファイルが見つかりません:\n{csv_path}")
            return

        # UIを無効化
        self.stats_analyze_btn.configure(state='disabled')
        self.batch_status_label.configure(text="統計分析を実行中...", text_color="#ffa500")

        # 別スレッドで実行
        thread = threading.Thread(target=self.run_stats_analysis, args=(csv_path,))
        thread.daemon = True
        thread.start()

    def run_stats_analysis(self, csv_path):
        """統計分析実行"""
        try:
            import sys
            from io import StringIO
            from analyze_results import analyze_batch_results

            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # 統計分析実行
            analyze_batch_results(csv_path)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            self.root.after(0, self.display_stats_results, output, True)

        except Exception as e:
            sys.stdout = old_stdout
            self.root.after(0, self.display_stats_results, str(e), False)

    def display_stats_results(self, output, success):
        """統計分析結果表示"""
        self.stats_analyze_btn.configure(state='normal')

        self.batch_result_text.delete("1.0", tk.END)
        self.batch_result_text.insert("1.0", output)

        if success:
            self.batch_status_label.configure(
                text="✅ 統計分析完了！23種類のプロットが analysis_output/ に保存されました",
                text_color="#00ff88"
            )

            messagebox.showinfo(
                "完了",
                "統計分析が完了しました。\n\n"
                "23種類の研究用プロット（300dpi）が\n"
                "analysis_output/ フォルダに保存されました。\n\n"
                "・ハルシネーション検出（4種類）\n"
                "・品質トレードオフ（5種類）\n"
                "・医療画像特化（4種類）\n"
                "・分布・PCA分析（4種類）\n"
                "・その他（6種類）"
            )

            # フォルダを開くか確認
            result = messagebox.askyesno(
                "フォルダを開く",
                "analysis_output フォルダを開きますか？"
            )
            if result:
                output_dir = "analysis_output"
                if os.path.exists(output_dir):
                    os.startfile(output_dir)
        else:
            self.batch_status_label.configure(
                text="❌ 統計分析エラー",
                text_color="#ff4444"
            )
            messagebox.showerror("エラー", f"統計分析中にエラーが発生しました:\n{output}")

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
            result_text += f"✅ 疑いデータCSV: {output_path}\n"

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
            result_text += f"✅ サマリーCSV: {summary_path}\n"

            # 詳細統計レポート保存（テキスト）
            report_path = str(Path(csv_path).parent / f"hallucination_report_{Path(csv_path).stem}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
                f.write(f"\n{'='*60}\n")
                f.write("【モデル別詳細統計】\n")
                f.write(f"{'='*60}\n\n")
                f.write(summary_stats.to_string())
            result_text += f"✅ 詳細レポート: {report_path}\n"

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

            result_text += f"✅ 分析グラフ: {graph_path}\n"
            result_text += f"{'='*60}\n"

            # 結果表示
            self.batch_result_text.delete("1.0", tk.END)
            self.batch_result_text.insert("1.0", result_text)

            self.batch_status_label.configure(
                text=f"✅ ハルシネーション疑いデータ抽出完了（{len(hallucination_all)}件）",
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
            result_text += "✅ クリーンデータセット作成完了\n"
            result_text += "=" * 70 + "\n\n"
            result_text += f"📁 出力先: {output_dir}\n\n"
            result_text += f"📊 統計:\n"
            result_text += f"  総データ数: {len(df)}件\n"
            result_text += f"  正常データ: {len(normal_df)}件 ({len(normal_df)/len(df)*100:.1f}%)\n"
            result_text += f"  正常画像数: {len(normal_df['image_id'].unique())}枚\n\n"
            result_text += f"【モデル別正常データ】\n"
            for model in sorted(df['model'].unique()):
                count = len(normal_df[normal_df['model'] == model])
                total = len(df[df['model'] == model])
                result_text += f"  {model}: {count}/{total}件 ({count/total*100:.1f}%)\n"
            result_text += f"\n📄 ファイル:\n"
            result_text += f"  metadata.csv : メタデータ\n"
            result_text += f"  README.txt   : 説明書\n"
            result_text += f"  コピーファイル数: {len(copied_files)}個\n"

            self.batch_result_text.delete("1.0", tk.END)
            self.batch_result_text.insert("1.0", result_text)

            self.batch_status_label.configure(
                text=f"✅ クリーンデータセット作成完了（{len(normal_df['image_id'].unique())}枚）",
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

    def draw_circular_meter(self, canvas, percentage, color):
        """円形メーターを描画"""
        canvas.delete("all")

        # 背景円
        canvas.create_oval(5, 5, 55, 55, outline="#444444", width=3)

        # 使用率の円弧
        if percentage > 0:
            extent = -percentage * 3.6  # 360度 = 100%
            canvas.create_arc(
                5, 5, 55, 55,
                start=90,
                extent=extent,
                outline=color,
                width=4,
                style=tk.ARC
            )

        # パーセンテージ表示
        canvas.create_text(
            30, 30,
            text=f"{int(percentage)}%",
            fill=color,
            font=("Arial", 12, "bold")
        )

    def update_system_monitor(self):
        """システム使用率を更新"""
        if not MONITORING_AVAILABLE:
            return

        try:
            # CPU使用率
            self.cpu_usage = psutil.cpu_percent(interval=0.1)

            # RAM使用率
            self.ram_usage = psutil.virtual_memory().percent

            # GPU使用率
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usage = gpus[0].load * 100
                else:
                    self.gpu_usage = 0
            except:
                self.gpu_usage = 0

            # メーター更新
            self.draw_circular_meter(self.cpu_canvas, self.cpu_usage, "#00ffff")
            self.draw_circular_meter(self.gpu_canvas, self.gpu_usage, "#00ff88")
            self.draw_circular_meter(self.ram_canvas, self.ram_usage, "#ffaa00")

        except Exception as e:
            pass

        # 1秒後に再実行
        if self.monitoring_active:
            self.root.after(1000, self.update_system_monitor)

    def start_monitoring(self):
        """モニタリング開始"""
        self.monitoring_active = True
        self.update_system_monitor()

    def stop_monitoring(self):
        """モニタリング停止"""
        self.monitoring_active = False

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
                text=f"画像読み込みエラー:\n{str(e)}",
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
                text=f"画像読み込みエラー:\n{str(e)}",
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
                text=f"画像読み込みエラー:\n{str(e)}",
                image=""
            )

    def start_analysis(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("エラー", "2つの画像を選択してください")
            return

        if not os.path.exists(self.img1_path.get()):
            messagebox.showerror("エラー", f"画像1が見つかりません:\n{self.img1_path.get()}")
            return

        if not os.path.exists(self.img2_path.get()):
            messagebox.showerror("エラー", f"画像2が見つかりません:\n{self.img2_path.get()}")
            return

        # UIを無効化
        self.analyze_btn.configure(state='disabled')
        self.progress.set(0)
        self.progress.start()
        self.current_step = "初期化中..."
        self.status_label.configure(text=f"分析中: {self.current_step}", text_color="#00ffff")
        self.result_text.delete("1.0", tk.END)
        self.interpretation_text.delete("1.0", tk.END)

        # 進捗更新スレッドを開始
        self.update_progress_display()

        # 別スレッドで分析実行
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()

    def update_progress_display(self):
        """進捗状況を定期的に更新"""
        if self.current_step and self.analyze_btn.cget('state') == 'disabled':
            self.status_label.configure(text=f"分析中: {self.current_step}")
            self.root.after(100, self.update_progress_display)

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

            results = analyze_images(
                self.img1_path.get(),
                self.img2_path.get(),
                self.output_dir.get(),
                self.original_path.get() if self.original_path.get() else None,
                evaluation_mode=self.evaluation_mode.get()  # 評価モードを渡す
            )

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            self.analysis_results = results
            self.root.after(0, self.display_results, output, results)

        except Exception as e:
            sys.stdout = old_stdout
            self.root.after(0, self.display_error, str(e))

    def display_results(self, output, results):
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

            self.status_label.configure(text=f"✅ {summary_msg}", text_color=color)
        else:
            self.status_label.configure(text="✅ 分析完了", text_color="#00ff88")

        messagebox.showinfo(
            "完了",
            f"分析が完了しました。\n\n"
            f"結果は '{self.output_dir.get()}' フォルダに保存されました。\n\n"
            f"「📊 わかりやすい解釈」タブで優劣を確認できます。"
        )

    def display_error(self, error_msg):
        self.progress.stop()
        self.progress.set(0)
        self.analyze_btn.configure(state='normal')
        self.status_label.configure(text="❌ エラーが発生しました", text_color="#ff4444")

        self.result_text.insert("1.0", f"エラー:\n{error_msg}")
        messagebox.showerror("エラー", f"分析中にエラーが発生しました:\n{error_msg}")

    def open_output_folder(self):
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            os.startfile(output_path)
        else:
            messagebox.showwarning("警告", f"出力フォルダが見つかりません:\n{output_path}")

    def clear_results(self):
        self.result_text.delete("1.0", tk.END)
        self.interpretation_text.delete("1.0", tk.END)
        self.status_label.configure(text="結果をクリアしました", text_color="#888888")
        self.progress.set(0)
        self.analysis_results = None

    def show_comparison_report(self):
        report_path = os.path.join(self.output_dir.get(), 'comparison_report.png')

        if not os.path.exists(report_path):
            messagebox.showwarning("警告", "比較レポートが見つかりません。\n先に分析を実行してください。")
            return

        # 新しいウィンドウで画像表示
        report_window = ctk.CTkToplevel(self.root)
        report_window.title("比較レポート")
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

        # 左パネルの表示切り替え
        self.batch_mode_frame.pack_forget()
        self.single_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 右パネルの表示切り替え
        self.batch_right_frame.pack_forget()
        self.single_right_frame.pack(fill=tk.BOTH, expand=True)

    def switch_to_batch_mode(self):
        """バッチ処理モードに切り替え"""
        # ボタンの色を変更
        self.batch_mode_btn.configure(fg_color="#4A90E2", text_color="#FFFFFF")
        self.single_mode_btn.configure(fg_color="#4a5568", text_color="#ffffff")

        # 左パネルの表示切り替え
        self.single_mode_frame.pack_forget()
        self.batch_mode_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 右パネルの表示切り替え
        self.single_right_frame.pack_forget()
        self.batch_right_frame.pack(fill=tk.BOTH, expand=True)

def main():
    root = ctk.CTk()
    app = ModernImageAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
