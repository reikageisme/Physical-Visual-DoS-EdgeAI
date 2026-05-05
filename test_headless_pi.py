import cv2
import torch
import time
import psutil
import csv
import subprocess
from core.victim_model import VictimModel

def get_pi_temperature():
    """Hàm chuyên dụng để đọc nhiệt độ lõi của Raspberry Pi"""
    try:
        temp_str = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
        # Kết quả có dạng "temp=45.6'C", cần cắt lấy số
        temp = float(temp_str.replace('temp=', '').replace('\'C\n', ''))
        return temp
    except:
        return 0.0

def main():
    print("=== EDGE-AI VISUAL DoS ATTACK (HEADLESS MODE) ===")
    victim = VictimModel()

    # Load miếng dán Sponge Patch
    patch_img = cv2.imread("sponge_patch_final.png")
    patch_img = cv2.resize(patch_img, (64, 64)) 
    patch_tensor = torch.from_numpy(patch_img).float() / 255.0
    patch_tensor = patch_tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(victim.device)

    # Cấu hình IP Camera
    ip_cam_url = "http://192.168.1.100:8080/video" # Sửa lại đúng IP điện thoại nhé
    cap = cv2.VideoCapture(ip_cam_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 

    if not cap.isOpened():
        print("[-] LỖI: Không kết nối được IP Camera.")
        return

    # Mở file CSV để lưu log vẽ biểu đồ
    csv_file = open("attack_metrics.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time(s)", "Mode", "FPS", "RawBoxes", "CPU_Load(%)", "RAM_Usage(%)", "Temperature(C)"])

    print("[+] KẾT NỐI THÀNH CÔNG! Đang thu thập số liệu...")
    print("Nhấn CTRL+C trên Terminal để dừng chương trình.\n")

    attack_mode = True # Bật tấn công mặc định để test tải
    start_time_global = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_resized = cv2.resize(frame, (320, 320))
            input_tensor = torch.from_numpy(frame_resized).float() / 255.0
            input_tensor = input_tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(victim.device)

            if attack_mode:
                start_x, start_y = 128, 128
                input_tensor[0, :, start_y:start_y+64, start_x:start_x+64] = patch_tensor[0]

            # Bắt đầu suy luận
            start_time = time.time()
            outputs = victim.get_raw_predictions(input_tensor)
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0

            # Đo lường thông số
            scores = outputs[0]['scores']
            num_boxes = len(scores[scores > 0.05])
            
            # Cứ mỗi 10 frame (tránh lag quá) thì in thông số và lưu CSV 1 lần
            frame_count += 1
            if frame_count % 10 == 0:
                cpu_load = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                temp = get_pi_temperature()
                elapsed_time = round(time.time() - start_time_global, 1)
                
                mode_str = "ATTACK" if attack_mode else "CLEAN"
                
                # In ra màn hình console của VS Code
                print(f"[{elapsed_time}s] {mode_str} | FPS: {fps:.1f} | Boxes: {num_boxes} | CPU: {cpu_load}% | Temp: {temp}°C")
                
                # Ghi vào file CSV
                csv_writer.writerow([elapsed_time, mode_str, round(fps, 1), num_boxes, cpu_load, ram_usage, temp])

    except KeyboardInterrupt:
        print("\n[!] Đã nhận lệnh dừng (CTRL+C).")
    
    finally:
        cap.release()
        csv_file.close()
        print("[+] Đã lưu toàn bộ số liệu đo lường vào file 'attack_metrics.csv'.")

if __name__ == "__main__":
    main()