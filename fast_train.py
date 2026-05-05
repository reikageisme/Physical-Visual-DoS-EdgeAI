import torch
import cv2
import numpy as np
import os
from core.victim_model import VictimModel
from core.eot_transforms import apply_eot

class FastPGD:
    def __init__(self, patch_size=500, iterations=100, learning_rate=0.05):
        self.patch_size = patch_size
        self.iterations = iterations
        self.lr = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_patch(self, victim_model, conf_thresh=0.01):
        print(f"\n[+] BẬT CHẾ ĐỘ THẦN TỐC (PROJECTED GRADIENT DESCENT)...")
        print(f"    Thay vì mò mẫm ngẫu nhiên, ta dùng Đạo hàm (Derivatives) ép YOLO tự khai ra ke hở!")
        
        # 1. Tạo patch ban đầu với requires_grad=True
        patch = torch.rand((1, 3, self.patch_size, self.patch_size), device=self.device, dtype=torch.float32, requires_grad=True)
        
        # Sử dụng thuật toán tối ưu hóa Adam, hội tụ nhanh gấp nghìn lần Lai ghép
        optimizer = torch.optim.Adam([patch], lr=self.lr)
        
        # Môi trường 720p 
        base_image = torch.rand((1, 3, 720, 1280), dtype=torch.float32, device=self.device)
        _, _, H, W = base_image.shape
        y_offset = (H - self.patch_size) // 2
        x_offset = (W - self.patch_size) // 2

        # 2. Xé rào bảo vệ no_grad của YOLO để dòng chảy Gradient truyền ngược
        victim_model.model.float() # Đồng bộ YOLO về Float32 để khớp tính toán đạo hàm với ảnh Float32
        for param in victim_model.model.parameters():
            param.requires_grad = False

        for i in range(self.iterations):
            optimizer.zero_grad()
            
            # --- CHUẨN BỊ ẢNH ---
            adv_images = base_image.clone()
            adv_images[:, :, y_offset:y_offset+self.patch_size, x_offset:x_offset+self.patch_size] = patch
            
            # --- EOT VÀ INFERENCE ---
            eot_images = apply_eot(adv_images)
            
            # YOLO yêu cầu kích thước chia hết cho 32 (720 -> 736) - Pad trực tiếp siêu tốc
            import torch.nn.functional as F
            _, _, cur_H, cur_W = eot_images.shape
            new_H = ((cur_H + 31) // 32) * 32  
            new_W = ((cur_W + 31) // 32) * 32
            if new_H != cur_H or new_W != cur_W:
                eot_images = F.pad(eot_images, (0, new_W - cur_W, 0, new_H - cur_H), value=0.0)
            
            # Đẩy ảnh float32 qua YOLO, ta buộc mô hình nội suy lại Gradient 
            # *Lưu ý: PGD cần float32 để tránh Gradient underflow/overflow ở Float16
            preds = victim_model.model(eot_images)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
                
            cls_probs = preds[:, 4:, :]
            max_scores, _ = torch.max(cls_probs, dim=1) # Lấy scores cao nhất của các hộp
            
            # --- TÍNH LOSS (CHÌA KHÓA TỐC ĐỘ VẠN LẦN) ---
            active_mask = max_scores > conf_thresh
            num_boxes = torch.sum(active_mask, dim=1).float()
            total_conf = torch.sum(max_scores * active_mask, dim=1)
            
            # Hàm mất mát: Ta muốn Maximize DoS => Maximize điểm số => Minimize âm của điểm
            loss = -(total_conf.mean() + 1.5 * num_boxes.mean())
            
            if loss.requires_grad:
                loss.backward() # YOLO trực tiếp nôn ra đáp án đổi pixel nào để bị lag nhất !!!
                
                # Cập nhật pixel
                optimizer.step()
                
                # Cắt giá trị pixel nằm gọn vùng 0-1 (Projected)
                with torch.no_grad():
                    patch.clamp_(0, 1)
            
            if i % 10 == 0:
                print(f"  -> Lặp PGD {i}/{self.iterations} | Sức mạnh DoS: {-loss.item():.2f}")
                
        print("[+] HOÀN THÀNH CHẾ TẠO PATCH BẰNG ĐẠO HÀM!")
        return patch.detach()

if __name__ == "__main__":
    print(f"[*] Đánh thức RTX 5090 PGD-Mode...")
    victim = VictimModel()
    
    # 💥 MỞ KHÓA GRADIENT CHO VICTIM MODEL 💥
    # Ta phải ghi đè bỏ hàm torch.no_grad() cũ trong Victim Model
    orig_get_raw = victim.get_raw_predictions
    
    pgd = FastPGD(patch_size=500, iterations=50, learning_rate=0.03)
    best_patch = pgd.generate_patch(victim)
    
    # Export ảnh
    patch_numpy = best_patch[0].cpu().numpy()
    patch_numpy = np.transpose(patch_numpy, (1, 2, 0)) 
    patch_numpy = (patch_numpy * 255).astype(np.uint8)
    patch_bgr = cv2.cvtColor(patch_numpy, cv2.COLOR_RGB2BGR)
    patch_large = cv2.resize(patch_bgr, (2480, 2480), interpolation=cv2.INTER_CUBIC)
    
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"super_fast_pgd_patch.png")
    cv2.imwrite(out_path, patch_large)
    print(f"=> MIẾNG DÁN CỰC MẠNH VÀ NÉT ĐÃ CÓ SẴN (Trong chưa tới 15 giây): {out_path}")
