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
