import cv2
import torch
import time
import psutil
import threading
import argparse
import os
import numpy as np
from core.victim_model import VictimModel
from utils.monitor import EdgeMonitor

# 1. MÔ PHỎNG RASPBERRY PI: Ép PyTorch xài ĐÚNG 1 LUỒNG CPU
torch.set_num_threads(1)

class WebcamStream:
    def __init__(self, src=0):
        # KHÔNG DÙNG DSHOW NỮA ĐỂ TRÁNH LỖI VĂNG CODE
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, frame = self.cap.read()
            if self.ret:
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description="Hệ thống thực nghiệm Physical DoS")
    parser.add_argument('--cam', type=int, default=0, help='ID camera')
    parser.add_argument('--patch', type=str, default=None, help='Đường dẫn tới file sponge_patch_final.png để dán đè ảo')
    args = parser.parse_args()

    print("=== HỆ THỐNG THỰC NGHIỆM VISUAL DoS (MÔ PHỎNG EDGE-AI) ===")
    victim = VictimModel()

    cam_id = args.cam
    print(f"[*] Đang kết nối tới Camera (ID: {cam_id}) ...")
    
    stream = WebcamStream(src=cam_id)
    time.sleep(2)

    if not stream.ret:
        print("[-] LỖI KHÔNG TÌM THẤY CAMERA!")
        return

    # 2. CỐ ĐỊNH TẢI TRỌNG (TARGET FPS = 10)
    # Camera giám sát thông thường chỉ chạy ở 10 FPS
    TARGET_FPS = 10
    FRAME_TIME = 1.0 / TARGET_FPS

    monitor = EdgeMonitor(log_dir="logs")

    # Load patch ảo nếu có
    virtual_patch = None
    if args.patch and os.path.exists(args.patch):
        print(f"[*] Đã tải Patch Ảo từ: {args.patch}")
        virtual_patch = cv2.imread(args.patch)
        virtual_patch = cv2.resize(virtual_patch, (100, 100)) # Resize nhỏ lại dán góc

    try:
        while True:
            loop_start = time.time() # Bắt đầu đếm giờ cho 1 khung hình
            
            ret, frame = stream.read()
            if not ret: continue

            # === DIGITAL OVERLAY PATCH (Áp Patch ngay trong Code) ===
            if virtual_patch is not None:
                h, w = virtual_patch.shape[:2]
                # Dán patch vào góc phải dưới của khung hình
                frame[-h:, -w:] = virtual_patch

            # --- TIỀN XỬ LÝ & ĐẨY QUA AI ---
            frame_ai = cv2.resize(frame, (320, 320))
            input_tensor = torch.from_numpy(frame_ai).float() / 255.0
            input_tensor = input_tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(victim.device)
            
            # Lấy kết quả
            outputs = victim.get_raw_predictions(input_tensor)

            # Đếm Box (Màng lọc 5%)
            scores = outputs[0]['scores']
            num_boxes = len(scores[scores > 0.05])

            # --- KỸ THUẬT ÉP ÉP XUNG BẰNG CODE ---
            process_time = time.time() - loop_start # Thời gian AI và NMS đã ngốn
            
            # Nếu AI tính nhanh hơn 0.1 giây (đang nhàn rỗi), bắt hệ thống đi ngủ để hạ CPU
            sleep_time = FRAME_TIME - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # Tính FPS thực tế TỔNG CỘNG (Tránh chia cho 0)
            elapsed_time = max(time.time() - loop_start, 0.001)
            actual_fps = 1.0 / elapsed_time

            # Đo tài nguyên bằng Monitor
            stats = monitor.log_status(actual_fps)

            # --- HIỂN THỊ GIAO DIỆN ---
            frame_display = cv2.resize(frame, (640, 480))
            color = (0, 0, 255) if num_boxes > 100 else (0, 255, 0)
            
            cv2.putText(frame_display, "PHYSICAL DoS TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_display, f"FPS: {actual_fps:.1f} / {TARGET_FPS}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame_display, f"CPU: {stats['cpu']:.1f}% | RAM: {stats['ram_pct']:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            temp_text = f"Temp: {stats['temp']:.1f}C" if stats['temp'] > 0 else "Temp: N/A (PC)"
            cv2.putText(frame_display, temp_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame_display, f"Raw Boxes: {num_boxes}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Edge-AI Physical Attack", frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[!] Dừng khẩn cấp bằng bàn phím.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[*] ĐÃ ĐÓNG HỆ THỐNG AN TOÀN!")

if __name__ == "__main__":
    main()