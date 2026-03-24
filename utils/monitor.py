import psutil
import time
import os
import csv
from datetime import datetime

class EdgeMonitor:
    def __init__(self, log_dir="logs"):
        """
        Khởi tạo bộ giám sát tài nguyên Edge Device.
        """
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File log cho mỗi lần chạy
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"resource_log_{current_time}.csv")
        
        # Mở file và ghi header
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "CPU_Percent", "RAM_Percent", "RAM_Used_MB", "Temperature_C", "FPS_Actual"])
            
        # Khởi tạo psutil CPU 
        psutil.cpu_percent(interval=None)

    def get_cpu_load(self):
        """Lấy phần trăm CPU hiện tại"""
        return psutil.cpu_percent(interval=None)

    def get_ram_usage(self):
        """Lấy phần trăm RAM và RAM đã dùng (MB)"""
        mem = psutil.virtual_memory()
        return mem.percent, mem.used / (1024 * 1024)

    def get_pi_temperature(self):
        """
        Đọc nhiệt độ CPU (Rất quan trọng với Raspberry Pi).
        Nếu chạy trên Windows/Linux PC không hỗ trợ, sẽ trả về 0.
        """
        try:
            # File system chuẩn của Raspberry Pi (Raspbian/Ubuntu)
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
            return temp
        except FileNotFoundError:
            # Nếu chạy giả lập trên Windows thì bỏ qua
            return 0.0

    def log_status(self, current_fps):
        """
        Lấy tất cả các thông số, ghi vào file CSV và trả về đễ hiển thị lên màn hình.
        """
        cpu = self.get_cpu_load()
        ram_pct, ram_mb = self.get_ram_usage()
        temp = self.get_pi_temperature()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Ghi vào DB/CSV
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, cpu, ram_pct, ram_mb, temp, current_fps])

        return {
            "cpu": cpu,
            "ram_pct": ram_pct,
            "ram_mb": ram_mb,
            "temp": temp,
            "fps": current_fps
        }
