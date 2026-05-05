import torch
import cv2
import numpy as np
import argparse
import os
from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness
from attack.genetic_algo import SpongeGA

def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Thuật toán GA tạo Sponge Patch")
    # THEO YÊU CẦU: Ép xung siêu tốc - giảm số lượng khảo sát thừa thãi
    parser.add_argument('--pop', type=int, default=64, help="Kích thước lô nhỏ, siêu tốc")
    parser.add_argument('--gen', type=int, default=30, help="Số thế hệ tiến hóa siêu nhanh (mặc định 30)")
    parser.add_argument('--size', type=int, default=500, help="Kích thước pixel của miếng dán (mặc định 500)")
    args = parser.parse_args()

    print(f"=== HỆ THỐNG TẠO MIẾNG DÁN SPONGE PATCH (VISUAL DoS) ===")
    print(f"[*] Cấu hình: Quần thể={args.pop}, Thế hệ={args.gen}, Kích thước Patch={args.size}x{args.size}")
    
    # Bật tăng tốc CNN Cudnn 
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Kiểm tra sức mạnh VRAM thực tế
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] Phát hiện GPU: {torch.cuda.get_device_name(0)} | Tổng VRAM: {vram_gb:.1f} GB")
        if vram_gb > 24 and args.pop < 500:
            print("[!] Cảnh báo: GPU của bạn siêu mạnh (>24GB VRAM). Bạn đang để Pop hơi thấp, hãy thử `--pop 1000` để tối đa hóa hiệu năng!")

    # 1. Khởi tạo Victim Model (YOLOv8 Edge)
    print("[1] Đang tải mô hình mục tiêu (YOLOv8 Nano)...")
    victim = VictimModel()
    
    # 2. Chuẩn bị ảnh nền (Background Image) - SỬ DỤNG MÔI TRƯỜNG 720p (720x1280)
    print("[2] Đang chuẩn bị môi trường giả lập (Camera 720p)...")
    base_image = torch.rand((1, 3, 720, 1280), dtype=torch.float32, device=victim.device)
    
    # 3. Wrapper cho hàm đánh giá để truyền vào GA
    def evaluate_fitness(outputs, conf_thresh=0.01):
        return calculate_sponge_fitness(outputs, conf_thresh)
        
    # 4. Khởi tạo Thuật toán Di truyền (SpongeGA)
    print("[3] Khởi tạo Giải thuật Di truyền (GA)...")
    ga = SpongeGA(patch_size=args.size, pop_size=args.pop, generations=args.gen, mutation_rate=0.1)
    
    # 5. Khai hỏa! Tiến hóa miếng dán
    best_patch_tensor = ga.evolve(victim, evaluate_fitness, base_image)
    
    # 6. Xử lý hậu kỳ và lưu kết quả ra file ảnh PNG cho A4 (chuẩn 300DPI)
    print("\n[4] Đang xuất file miếng dán sát thủ khổ A4 (2480x2480)...")
    
    # Chuyển từ Tensor PyTorch (0.0 - 1.0, shape: C x H x W) 
    # Sang Numpy Array (0 - 255, shape: H x W x C) để OpenCV có thể đọc được
    patch_numpy = best_patch_tensor.cpu().numpy()
    patch_numpy = np.transpose(patch_numpy, (1, 2, 0)) 
    patch_numpy = (patch_numpy * 255).astype(np.uint8)
    
    # PyTorch dùng hệ màu RGB, OpenCV dùng hệ màu BGR -> Cần đổi lại hệ màu
    patch_bgr = cv2.cvtColor(patch_numpy, cv2.COLOR_RGB2BGR)
    
    # Phóng to miếng dán lên 2480x2480 (kích thước siêu nét in trên A4 tại 300DPI) bằng thuật toán nội suy CUBIC
    patch_large = cv2.resize(patch_bgr, (2480, 2480), interpolation=cv2.INTER_CUBIC)
    
    # Tạo thư mục outputs nếu chưa có
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"sponge_patch_A4_g{args.gen}_p{args.pop}.png")
    
    cv2.imwrite(out_path, patch_large)
    print(f"=> THÀNH CÔNG! Đã lưu miếng dán chuẩn A4 tại: {out_path}")

if __name__ == "__main__":
    main()