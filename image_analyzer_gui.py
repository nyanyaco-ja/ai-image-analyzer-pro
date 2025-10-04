import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import threading
import os
from advanced_image_analyzer import analyze_images
import json
from PIL import Image, ImageTk

class ImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("画像比較分析ツール - AI高解像度変換評価")
        self.root.geometry("900x700")

        # 変数
        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="analysis_results")

        self.create_widgets()

    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # タイトル
        title_label = ttk.Label(main_frame, text="AI高解像度変換 画像比較分析ツール",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # 画像1選択
        ttk.Label(main_frame, text="画像1 (例: chaiNNer):", font=('Arial', 10)).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.img1_path, width=50).grid(
            row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="参照...", command=self.browse_image1).grid(
            row=1, column=2, pady=5)

        # 画像2選択
        ttk.Label(main_frame, text="画像2 (例: Upscayl):", font=('Arial', 10)).grid(
            row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.img2_path, width=50).grid(
            row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="参照...", command=self.browse_image2).grid(
            row=2, column=2, pady=5)

        # 出力ディレクトリ
        ttk.Label(main_frame, text="出力フォルダ:", font=('Arial', 10)).grid(
            row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(
            row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="参照...", command=self.browse_output).grid(
            row=3, column=2, pady=5)

        # 分析開始ボタン
        self.analyze_button = ttk.Button(main_frame, text="分析開始",
                                         command=self.start_analysis,
                                         style='Accent.TButton')
        self.analyze_button.grid(row=4, column=0, columnspan=3, pady=20)

        # プログレスバー
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress.grid(row=5, column=0, columnspan=3, pady=10)

        # ステータスラベル
        self.status_label = ttk.Label(main_frame, text="画像を選択して分析を開始してください",
                                      foreground="blue")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)

        # 結果表示エリア（タブ化）
        result_frame = ttk.LabelFrame(main_frame, text="分析結果", padding="10")
        result_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        # タブ作成
        tab_control = ttk.Notebook(result_frame)

        # タブ1: 詳細データ
        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='詳細データ')

        self.result_text = scrolledtext.ScrolledText(tab1, width=100, height=20,
                                                     font=('Consolas', 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # タブ2: わかりやすい解釈
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text='📊 わかりやすい解釈')

        self.interpretation_text = scrolledtext.ScrolledText(tab2, width=100, height=20,
                                                             font=('Consolas', 10),
                                                             fg='#2c3e50', bg='#ecf0f1')
        self.interpretation_text.pack(fill=tk.BOTH, expand=True)

        tab_control.pack(fill=tk.BOTH, expand=True)

        # ボタンフレーム
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="比較レポートを表示",
                  command=self.show_comparison_report).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="結果フォルダを開く",
                  command=self.open_output_folder).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="結果をクリア",
                  command=self.clear_results).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="JSONで保存",
                  command=self.save_json).grid(row=0, column=3, padx=5)

        # ウィンドウのリサイズ設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        # 分析結果データ
        self.analysis_results = None

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

    def browse_output(self):
        dirname = filedialog.askdirectory(title="出力フォルダを選択")
        if dirname:
            self.output_dir.set(dirname)

    def start_analysis(self):
        # 入力チェック
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
        self.analyze_button.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="分析中... しばらくお待ちください", foreground="orange")
        self.result_text.delete(1.0, tk.END)

        # 別スレッドで分析実行
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()

    def run_analysis(self):
        try:
            # 標準出力をキャプチャするためのクラス
            import sys
            from io import StringIO

            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # 分析実行
            results = analyze_images(
                self.img1_path.get(),
                self.img2_path.get(),
                self.output_dir.get()
            )

            # 標準出力を復元
            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # 結果を保存
            self.analysis_results = results

            # UIスレッドで結果表示
            self.root.after(0, self.display_results, output, results)

        except Exception as e:
            # エラー時
            sys.stdout = old_stdout
            self.root.after(0, self.display_error, str(e))

    def display_results(self, output, results):
        # プログレスバー停止
        self.progress.stop()
        self.analyze_button.config(state='normal')

        # 詳細データタブに結果表示
        self.result_text.insert(1.0, output)

        # わかりやすい解釈タブに表示
        if results and 'interpretation' in results:
            from result_interpreter import format_interpretation_text
            interpretation_text = format_interpretation_text(results['interpretation'])
            self.interpretation_text.delete(1.0, tk.END)
            self.interpretation_text.insert(1.0, interpretation_text)

            # 解釈結果に基づいてステータス更新
            interp = results['interpretation']
            winner = interp['winner']
            summary_msg = interp['summary']['message']

            if winner == 'img1':
                color = "blue"
            elif winner == 'img2':
                color = "green"
            else:
                color = "orange"

            self.status_label.config(text=f"分析完了 - {summary_msg}", foreground=color)
        else:
            # スコアをハイライト（フォールバック）
            if results and 'total_score' in results:
                score = results['total_score']['img2']
                if score >= 95:
                    color = "green"
                    message = f"分析完了 - 画像2のスコア: {score}/100 (優秀)"
                elif score >= 85:
                    color = "blue"
                    message = f"分析完了 - 画像2のスコア: {score}/100 (良好)"
                elif score >= 70:
                    color = "orange"
                    message = f"分析完了 - 画像2のスコア: {score}/100 (普通)"
                else:
                    color = "red"
                    message = f"分析完了 - 画像2のスコア: {score}/100 (要改善)"

                self.status_label.config(text=message, foreground=color)
            else:
                self.status_label.config(text="分析完了", foreground="green")

        # 完了メッセージ
        messagebox.showinfo("完了",
                           f"分析が完了しました。\n結果は '{self.output_dir.get()}' フォルダに保存されました。\n\n"
                           f"「📊 わかりやすい解釈」タブで優劣を確認できます。")

    def display_error(self, error_msg):
        # プログレスバー停止
        self.progress.stop()
        self.analyze_button.config(state='normal')
        self.status_label.config(text="エラーが発生しました", foreground="red")

        # エラー表示
        self.result_text.insert(1.0, f"エラー:\n{error_msg}")
        messagebox.showerror("エラー", f"分析中にエラーが発生しました:\n{error_msg}")

    def open_output_folder(self):
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            os.startfile(output_path)  # Windowsでフォルダを開く
        else:
            messagebox.showwarning("警告", f"出力フォルダが見つかりません:\n{output_path}")

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)
        self.interpretation_text.delete(1.0, tk.END)
        self.status_label.config(text="結果をクリアしました", foreground="blue")
        self.analysis_results = None

    def save_json(self):
        if not self.analysis_results:
            messagebox.showwarning("警告", "保存する分析結果がありません")
            return

        filename = filedialog.asksaveasfilename(
            title="JSON保存",
            defaultextension=".json",
            filetypes=[("JSONファイル", "*.json"), ("すべてのファイル", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("成功", f"結果を保存しました:\n{filename}")
            except Exception as e:
                messagebox.showerror("エラー", f"保存中にエラーが発生しました:\n{e}")

    def show_comparison_report(self):
        """比較レポート画像を表示"""
        report_path = os.path.join(self.output_dir.get(), 'comparison_report.png')

        if not os.path.exists(report_path):
            messagebox.showwarning("警告", "比較レポートが見つかりません。\n先に分析を実行してください。")
            return

        # 新しいウィンドウで画像表示
        report_window = tk.Toplevel(self.root)
        report_window.title("比較レポート")
        report_window.geometry("1200x800")

        # 画像読み込み
        img = Image.open(report_path)

        # ウィンドウサイズに合わせてリサイズ
        display_width = 1180
        display_height = 750
        img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        # スクロール可能なキャンバス
        canvas = tk.Canvas(report_window, width=display_width, height=display_height)
        scrollbar_y = ttk.Scrollbar(report_window, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(report_window, orient="horizontal", command=canvas.xview)

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # 画像表示
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # 参照を保持

        # スクロール領域設定
        canvas.configure(scrollregion=canvas.bbox("all"))

        # リサイズ設定
        report_window.columnconfigure(0, weight=1)
        report_window.rowconfigure(0, weight=1)

def main():
    root = tk.Tk()
    app = ImageAnalyzerGUI(root)

    # アイコン設定（オプション）
    try:
        root.iconbitmap('icon.ico')  # アイコンファイルがあれば
    except:
        pass

    root.mainloop()

if __name__ == "__main__":
    main()
