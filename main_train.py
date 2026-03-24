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
    parser.add_argument('--pop', type=int, default=30, help="Kích thước quần thể (Population size)")
    parser.add_argument('--gen', type=int, default=50, help="Số thế hệ tiến hóa (Generations)")
    parser.add_argument('--size', type=int, default=64, help="Kích thước pixel của Sponge Patch")
    args = parser.parse_args()

    print(f"=== HỆ THỐNG TẠO MIẾNG DÁN SPONGE PATCH (VISUAL DoS) ===")
    print(f"[*] Cấu hình: Quần thể={args.pop}, Thế hệ={args.gen}, Kích thước Patch={args.size}x{args.size}")
    
    # 1. Khởi tạo Victim Model (MobileNet-SSD)
    print("[1] Đang tải mô hình mục tiêu (MobileNet-SSD)...")
    victim = VictimModel()
    
    # 2. Chuẩn bị ảnh nền (Background Image)
    # Tạm thời dùng một ảnh nhiễu ngẫu nhiên kích thước 320x320 làm nền (Ảnh Tensor 0-1)
    # Sau này test thực tế, fen có thể dùng cv2.imread để load khung hình từ Camera
    print("[2] Đang chuẩn bị môi trường giả lập...")
    base_image = torch.rand((1, 3, 320, 320)).to(victim.device)
    
    # 3. Wrapper cho hàm đánh giá để truyền vào GA
    def evaluate_fitness(outputs, conf_thresh=0.01):
        return calculate_sponge_fitness(outputs, conf_thresh)
        
    # 4. Khởi tạo Thuật toán Di truyền (SpongeGA)
    print("[3] Khởi tạo Giải thuật Di truyền (GA)...")
    ga = SpongeGA(patch_size=args.size, pop_size=args.pop, generations=args.gen, mutation_rate=0.1)
    
    # 5. Khai hỏa! Tiến hóa miếng dán
    best_patch_tensor = ga.evolve(victim, evaluate_fitness, base_image)
    
    # 6. Xử lý hậu kỳ và lưu kết quả ra file ảnh PNG
    print("\n[4] Đang xuất file miếng dán sát thủ...")
    
    # Chuyển từ Tensor PyTorch (0.0 - 1.0, shape: C x H x W) 
    # Sang Numpy Array (0 - 255, shape: H x W x C) để OpenCV có thể đọc được
    patch_numpy = best_patch_tensor.cpu().numpy()
    patch_numpy = np.transpose(patch_numpy, (1, 2, 0)) 
    patch_numpy = (patch_numpy * 255).astype(np.uint8)
    
    # PyTorch dùng hệ màu RGB, OpenCV dùng hệ màu BGR -> Cần đổi lại hệ màu
    patch_bgr = cv2.cvtColor(patch_numpy, cv2.COLOR_RGB2BGR)
    
    # Phóng to miếng dán lên 256x256 để nhìn cho rõ (dùng thuật toán nội suy gần nhất để giữ nguyên pixel)
    patch_large = cv2.resize(patch_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Tạo thư mục outputs nếu chưa có
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"sponge_patch_g{args.gen}_p{args.pop}.png")
    
    cv2.imwrite(out_path, patch_large)
    print(f"=> THÀNH CÔNG! Đã lưu miếng dán tại: {out_path}")

if __name__ == "__main__":
    main()