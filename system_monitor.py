import tkinter as tk

# CPU/GPUモニタリング
try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class SystemMonitorMixin:
    """システムモニター機能を提供するMixinクラス"""

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
        """システム使用率を更新（最適化版）"""
        if not MONITORING_AVAILABLE:
            return

        try:
            # CPU使用率（interval=0でブロッキングなし、前回からの平均）
            self.cpu_usage = psutil.cpu_percent(interval=0)

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

        # 3秒後に再実行（1秒→3秒に変更してパフォーマンス向上）
        if self.monitoring_active:
            self.root.after(3000, self.update_system_monitor)

    def start_monitoring(self):
        """モニタリング開始"""
        self.monitoring_active = True
        self.update_system_monitor()

    def stop_monitoring(self):
        """モニタリング停止"""
        self.monitoring_active = False
