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
            text_color="#00ffff"
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
                text_color="#00ffff"
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

        # コンテンツエリア
        content_frame = ctk.CTkFrame(main_container, fg_color="#0a0e27")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 左側パネル（入力エリア）
        left_panel = ctk.CTkFrame(content_frame, fg_color="#1e2740", width=450, corner_radius=15)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # スクロール可能なフレーム
        scrollable_frame = ctk.CTkScrollableFrame(left_panel, fg_color="transparent")
        scrollable_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 画像選択セクション
        input_section = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        input_section.pack(fill=tk.X)

        # 画像1
        img1_label = ctk.CTkLabel(
            input_section,
            text="📸 画像 1",
            font=("Arial", 16, "bold"),
            text_color="#00ffff"
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
            fg_color="#00ffff",
            text_color="#000000",
            hover_color="#00cccc"
        )
        img1_btn.pack(fill=tk.X, pady=(0, 20))

        # 画像2
        img2_label = ctk.CTkLabel(
            input_section,
            text="📸 画像 2",
            font=("Arial", 16, "bold"),
            text_color="#00ffff"
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
            fg_color="#00ffff",
            text_color="#000000",
            hover_color="#00cccc"
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
            text_color="#00ffff"
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
            fg_color="#00ffff",
            text_color="#000000",
            hover_color="#00cccc"
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
        button_group = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
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

        # 右側パネル（結果表示エリア）
        right_panel = ctk.CTkFrame(content_frame, fg_color="#1e2740", corner_radius=15)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # タブビュー
        self.tabview = ctk.CTkTabview(
            right_panel,
            corner_radius=15,
            fg_color="#1e2740",
            segmented_button_fg_color="#2d3748",
            segmented_button_selected_color="#00ffff",
            segmented_button_selected_hover_color="#00cccc",
            text_color="#ffffff"
        )
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # タブ作成
        self.tabview.add("📊 わかりやすい解釈")
        self.tabview.add("📝 詳細データ")
        self.tabview.add("🔬 バッチ処理")

        # わかりやすい解釈タブ
        self.interpretation_text = ctk.CTkTextbox(
            self.tabview.tab("📊 わかりやすい解釈"),
            font=("Yu Gothic UI", 12),
            fg_color="#0a0e27",
            text_color="#00ffff",
            corner_radius=10
        )
        self.interpretation_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 詳細データタブ
        self.result_text = ctk.CTkTextbox(
            self.tabview.tab("📝 詳細データ"),
            font=("Yu Gothic UI", 11),
            fg_color="#0a0e27",
            text_color="#00ff88",
            corner_radius=10
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # バッチ処理タブ
        self.create_batch_tab()

    def create_batch_tab(self):
        """バッチ処理タブのUI作成"""
        batch_tab = self.tabview.tab("🔬 バッチ処理")

        # スクロール可能なフレーム
        batch_scroll = ctk.CTkScrollableFrame(batch_tab, fg_color="transparent")
        batch_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 説明セクション
        info_frame = ctk.CTkFrame(batch_scroll, fg_color="#2d3748", corner_radius=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))

        info_title = ctk.CTkLabel(
            info_frame,
            text="📚 バッチ処理について",
            font=("Arial", 16, "bold"),
            text_color="#00ffff"
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
        config_frame = ctk.CTkFrame(batch_scroll, fg_color="#1e2740", corner_radius=10)
        config_frame.pack(fill=tk.X, pady=(0, 15))

        config_title = ctk.CTkLabel(
            config_frame,
            text="⚙️ バッチ処理設定",
            font=("Arial", 14, "bold"),
            text_color="#00ffff"
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
            fg_color="#00ffff",
            text_color="#000000",
            hover_color="#00cccc"
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
        csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        detail_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 実行ボタン
        self.batch_analyze_btn = ctk.CTkButton(
            batch_scroll,
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
            batch_scroll,
            height=20,
            corner_radius=10,
            progress_color="#00ffff"
        )
        self.batch_progress.pack(fill=tk.X, pady=(0, 10))
        self.batch_progress.set(0)

        # ステータス
        self.batch_status_label = ctk.CTkLabel(
            batch_scroll,
            text="設定を入力してバッチ処理を開始してください",
            font=("Arial", 11),
            text_color="#888888",
            wraplength=600
        )
        self.batch_status_label.pack(pady=(0, 15))

        # 統計分析セクション
        stats_frame = ctk.CTkFrame(batch_scroll, fg_color="#1e2740", corner_radius=10)
        stats_frame.pack(fill=tk.X, pady=(0, 15))

        stats_title = ctk.CTkLabel(
            stats_frame,
            text="📊 統計分析・プロット生成",
            font=("Arial", 14, "bold"),
            text_color="#00ffff"
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
            fg_color="#00ffff",
            text_color="#000000",
            hover_color="#00cccc"
        )
        csv_select_btn.pack(side=tk.RIGHT)

        # 統計分析実行ボタン
        self.stats_analyze_btn = ctk.CTkButton(
            stats_frame,
            text="📈 統計分析＋プロット生成（23種類）",
            command=self.start_stats_analysis,
            height=50,
            corner_radius=10,
            font=("Arial", 14, "bold"),
            fg_color="#ffa500",
            text_color="#000000",
            hover_color="#cc8400"
        )
        self.stats_analyze_btn.pack(fill=tk.X, padx=15, pady=(0, 15))

        # 結果表示エリア
        self.batch_result_text = ctk.CTkTextbox(
            batch_scroll,
            font=("Yu Gothic UI", 10),
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

    def browse_stats_csv(self):
        filename = filedialog.askopenfilename(
            title="CSVファイルを選択",
            filetypes=[("CSV", "*.csv"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.stats_csv_path.set(filename)

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
            "output_detail_dir": self.batch_output_detail.get()
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

            # バッチ処理実行
            batch_analyze(temp_config_path)

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

    def browse_original(self):
        filename = filedialog.askopenfilename(
            title="元画像を選択（低解像度）",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("すべてのファイル", "*.*")
            ]
        )
        if filename:
            self.original_path.set(filename)

    def browse_output(self):
        dirname = filedialog.askdirectory(title="出力フォルダを選択")
        if dirname:
            self.output_dir.set(dirname)

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
                self.original_path.get() if self.original_path.get() else None
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

def main():
    root = ctk.CTk()
    app = ModernImageAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
