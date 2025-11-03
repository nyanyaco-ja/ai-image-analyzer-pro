import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys

class StatsAnalysisMixin:
    """統計分析機能を提供するMixinクラス"""

    def browse_stats_csv(self):
        filename = filedialog.askopenfilename(
            title="CSVファイルを選択",
            filetypes=[("CSV", "*.csv"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.stats_csv_path.set(filename)


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

            # 統計分析実行（出力ディレクトリのパスを取得）
            output_dir = analyze_batch_results(csv_path)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            self.root.after(0, self.display_stats_results, output, True, output_dir)

        except Exception as e:
            sys.stdout = old_stdout
            self.root.after(0, self.display_stats_results, str(e), False, None)

    def display_stats_results(self, output, success, output_dir=None):
        """統計分析結果表示"""
        self.stats_analyze_btn.configure(state='normal')

        self.batch_result_text.delete("1.0", tk.END)
        self.batch_result_text.insert("1.0", output)

        if success:
            self.batch_status_label.configure(
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
            self.batch_status_label.configure(
                text="[ERROR] 統計分析エラー",
                text_color="#ff4444"
            )
            messagebox.showerror("エラー", f"統計分析中にエラーが発生しました:\n{output}")

