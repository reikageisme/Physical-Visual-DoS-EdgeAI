import cv2
import torch
import time
from core.victim_model import VictimModel

def main():
    print("=== HỆ THỐNG TEST VISUAL DoS QUA IP CAMERA ===")
    victim = VictimModel()

    # 1. Load miếng dán sát thủ
    patch_img = cv2.imread("sponge_patch_final.png")
    if patch_img is None:
        print("Lỗi: Không tìm thấy file sponge_patch_final.png. Hãy đảm bảo file đang nằm cùng thư mục.")
        return
        
    patch_img = cv2.resize(patch_img, (64, 64)) 
    
    # Tiền xử lý Patch sang Tensor (0-1), đổi BGR sang RGB
    patch_tensor = torch.from_numpy(patch_img).float() / 255.0
    patch_tensor = patch_tensor[:, :, [2, 1, 0]] 
    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(victim.device)

    # 2. Kết nối tới IP Camera
    # Lưu ý: Thường các app IP Webcam cần thêm "/video" ở cuối để lấy luồng MJPEG. 
    # Nếu không chạy, ông có thể thử xóa chữ "/video" đi nhé.
    ip_cam_url = "http://192.168.1.100:8080/video" 
    
    print(f"Đang kết nối tới luồng stream: {ip_cam_url} ...")
    cap = cv2.VideoCapture(ip_cam_url)
    
    # Khắc phục lỗi buffer của OpenCV khi đọc stream mạng (Giúp giảm độ trễ/lag)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 

    if not cap.isOpened():
        print("[-] LỖI: Không thể kết nối tới IP Camera. Hãy kiểm tra lại kết nối mạng hoặc đường dẫn URL!")
        return
        
    print("[+] KẾT NỐI THÀNH CÔNG! Phím T: Bật/Tắt tấn công | Phím Q: Thoát")

    attack_mode = False 

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("[-] Luồng video bị ngắt kết nối...")
            break

        # Resize khung hình về 320x320 cho MobileNet
        frame_resized = cv2.resize(frame, (320, 320))
        
        # Tiền xử lý khung hình để đưa vào AI
        input_tensor = torch.from_numpy(frame_resized).float() / 255.0
        input_tensor = input_tensor[:, :, [2, 1, 0]]
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(victim.device)

        # Chèn miếng dán Sponge Patch vào luồng video nếu BẬT tấn công
        if attack_mode:
            start_x, start_y = 128, 128 # Tọa độ chèn (Giữa màn hình)
            input_tensor[0, :, start_y:start_y+64, start_x:start_x+64] = patch_tensor[0]
            # Vẽ viền đỏ để đánh dấu vùng bị chèn nhiễu
            cv2.rectangle(frame_resized, (start_x, start_y), (start_x+64, start_y+64), (0, 0, 255), 2)

        # Bắt đầu tính giờ suy luận của AI
        start_time = time.time()
        
        # Đẩy dữ liệu qua MobileNet-SSD (Không đi qua NMS)
        outputs = victim.get_raw_predictions(input_tensor)

        # Tính toán FPS
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0

        # Trích xuất số lượng Bounding Box sinh ra (Ngưỡng tin cậy > 5%)
        scores = outputs[0]['scores']
        num_boxes = len(scores[scores > 0.05])

        # Giao diện hiển thị (HUD)
        color = (0, 0, 255) if attack_mode else (0, 255, 0)
        mode_text = "ATTACK: ON (Sponge)" if attack_mode else "ATTACK: OFF (Clean)"
        
        cv2.putText(frame_resized, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame_resized, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_resized, f"Raw Boxes: {num_boxes}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Edge-AI Visual DoS Attack - IP Camera", frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            attack_mode = not attack_mode

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()