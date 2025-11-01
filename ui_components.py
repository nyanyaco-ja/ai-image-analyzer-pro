import tkinter as tk
import customtkinter as ctk
from datetime import datetime

def get_timestamp_filename(base_name, extension=".csv"):
    """タイムスタンプ付きファイル名を生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_without_ext = base_name.replace(extension, "")
    return f"{name_without_ext}_{timestamp}{extension}"

class AccordionSection:
    """アコーディオンセクション（クリックで開閉）"""
    def __init__(self, parent, title, bg_color="#1e2740", title_color="#4A90E2", font_size=16):
        self.is_open = True

        # メインフレーム
        self.main_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.main_frame.pack(fill=tk.X, pady=(0, 10))

        # ヘッダー（クリック可能）
        self.header_btn = ctk.CTkButton(
            self.main_frame,
            text=f"▼ {title}",
            command=self.toggle,
            height=45,
            corner_radius=10,
            font=("Arial", font_size, "bold"),
            fg_color=bg_color,
            text_color=title_color,
            hover_color="#2d3748",
            anchor="w"
        )
        self.header_btn.pack(fill=tk.X, padx=0, pady=0)

        # コンテンツフレーム
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color=bg_color, corner_radius=10)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(5, 0))

        self.title = title

    def toggle(self):
        """開閉切り替え"""
        if self.is_open:
            self.content_frame.pack_forget()
            self.header_btn.configure(text=f"▶ {self.title}")
            self.is_open = False
        else:
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(5, 0))
            self.header_btn.configure(text=f"▼ {self.title}")
            self.is_open = True

    def get_content_frame(self):
        """コンテンツフレームを取得"""
        return self.content_frame

    def update_title(self, new_title):
        """タイトルを更新（多言語対応用）"""
        self.title = new_title
        if self.is_open:
            self.header_btn.configure(text=f"▼ {new_title}")
        else:
            self.header_btn.configure(text=f"▶ {new_title}")
