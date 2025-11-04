import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
from datetime import datetime

class TextRedirector:
    """標準出力をGUIテキストウィジェットにリダイレクトするクラス"""
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self.buffer = []

    def write(self, string):
        if string:  # 空でなければ
            self.buffer.append(string)
            # GUIスレッドで更新
            self.root.after(0, self._update_gui, string)

    def _update_gui(self, string):
        """GUIテキストウィジェットを更新"""
        try:
            self.text_widget.insert(tk.END, string)
            self.text_widget.see(tk.END)  # 自動スクロール
        except:
            pass

    def flush(self):
        pass

    def getvalue(self):
        return ''.join(self.buffer)

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
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            from analyze_results import analyze_batch_results

            # ログエリアをクリア
            self.root.after(0, lambda: self.batch_result_text.delete("1.0", tk.END))

            # 標準出力と標準エラーをGUIにリアルタイム表示
            text_redirector = TextRedirector(self.batch_result_text, self.root)
            sys.stdout = text_redirector
            sys.stderr = text_redirector  # エラーもGUIに表示

            # GUIの現在の言語設定を取得（デフォルトは日本語）
            lang = getattr(self, 'current_language', 'ja')

            # 統計分析実行（出力ディレクトリのパスを取得）
            # 言語パラメータを渡す
            output_dir = analyze_batch_results(csv_path, lang=lang)

            # 出力を取得
            output = text_redirector.getvalue()

            # デバッグ：出力が空の場合
            if not output.strip():
                output = f"[INFO] 統計分析が完了しました。\n出力ディレクトリ: {output_dir}\n\n(詳細ログが表示されない場合は、analyze_results.pyのprint文を確認してください)"
                self.root.after(0, lambda: self.batch_result_text.insert(tk.END, output))

            # ログを自動保存
            self.root.after(0, lambda: self.save_stats_log(output, output_dir))

            self.root.after(0, self.display_stats_results, output, True, output_dir)

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            error_msg = f"[ERROR] 統計分析中にエラーが発生しました:\n\n{str(e)}\n\n詳細:\n{error_detail}"
            self.root.after(0, self.display_stats_results, error_msg, False, None)
        finally:
            # 必ず標準出力/エラーを復元
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def display_stats_results(self, output, success, output_dir=None):
        """統計分析結果表示"""
        self.stats_analyze_btn.configure(state='normal')

        # ログは既にリアルタイム表示されているので、ここでは再表示しない
        # エラーの場合のみ表示
        if not success:
            self.batch_result_text.delete("1.0", tk.END)
            self.batch_result_text.insert("1.0", output)

        if success:
            self.batch_status_label.configure(
                text=f"[OK] 統計分析完了！26種類のプロットが {output_dir}/ に保存されました",
                text_color="#00ff88"
            )

            messagebox.showinfo(
                "完了",
                f"統計分析が完了しました。\n\n"
                f"26種類の研究用プロット（300dpi）が\n"
                f"{output_dir}/ フォルダに保存されました。\n\n"
                f"・ハルシネーション検出（4種類）\n"
                f"・品質トレードオフ（5種類）\n"
                f"・医療画像特化（4種類）\n"
                f"・分布・PCA分析（4種類）\n"
                f"・LFV法則証明（3種類：相関・分布・座標）\n"
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

    def save_stats_log(self, log_content, output_dir=None):
        """統計分析ログをファイルに保存"""
        try:
            # 保存先ディレクトリを決定
            if output_dir and os.path.exists(output_dir):
                log_dir = output_dir
            else:
                log_dir = "analysis_output"
                os.makedirs(log_dir, exist_ok=True)

            # タイムスタンプ付きログファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"analysis_log_{timestamp}.txt")

            # ログを保存
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"統計分析ログ\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
                f.write(log_content)

            print(f"\n[LOG] ログファイル保存: {log_file}")

        except Exception as e:
            print(f"\n[WARNING] ログ保存に失敗しました: {str(e)}")

    def export_current_log(self):
        """現在表示中のログを手動でエクスポート"""
        try:
            # 現在のログ内容を取得
            log_content = self.batch_result_text.get("1.0", tk.END)

            if not log_content.strip():
                messagebox.showinfo("情報", "ログが空です。")
                return

            # 保存先をダイアログで選択
            file_path = filedialog.asksaveasfilename(
                title="ログを保存",
                defaultextension=".txt",
                filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")],
                initialfile=f"stats_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"統計分析ログ（手動エクスポート）\n")
                    f.write(f"エクスポート日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(log_content)

                messagebox.showinfo("保存完了", f"ログを保存しました:\n{file_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"ログ保存に失敗しました:\n{str(e)}")

