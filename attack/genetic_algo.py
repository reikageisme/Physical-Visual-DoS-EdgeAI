import torch
import numpy as np
import cv2
import gc
from core.eot_transforms import apply_eot

class SpongeGA:
    def __init__(self, patch_size=64, pop_size=30, generations=50, mutation_rate=0.1):
        self.patch_size = patch_size
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Khởi tạo quần thể: Đưa thẳng lên GPU (Sử dụng Float32 để tính toán hình học Kornia không bị lỗi lượng giác)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.population = torch.rand((self.pop_size, 3, self.patch_size, self.patch_size), device=self.device, dtype=torch.float32)

    def apply_patch_batch(self, base_image_batch, patch_batch):
        """
        Dán patch lên ảnh theo từng Chunk (Mini-batch)
        """
        adv_images = base_image_batch.clone()
        _, _, H, W = adv_images.shape
        # Tính toán offset để dán patch vào giữa khung hình
        y_offset = (H - self.patch_size) // 2
        x_offset = (W - self.patch_size) // 2
        
        adv_images[:, :, y_offset:y_offset+self.patch_size, x_offset:x_offset+self.patch_size] = patch_batch
        return adv_images

    def evolve(self, victim_model, fitness_function, base_image):
        print(f"\n[+] BẮT ĐẦU VỚI GPU CHUNKING (Chống tràn RAM, ép {self.pop_size} cá thể vào 5090)...")
        
        # Đẩy Batch Size lên mức trung bình tối ưu
        batch_size = 64 
        
        # KHÔNG PHÓNG TO HÀNG TRĂM ẢNH BASE IMAGE. CHỈ GIỮ 1 BẢN MẪU TRÊN VRAM.
        # Broadcast (expand) chỉ tạo View logic, tránh Clone vùng nhớ khổng lồ.
        base_image_expanded = base_image.expand(batch_size, -1, -1, -1)
        
        for gen in range(self.generations):
            fitness_scores = []
            
            with torch.no_grad():
                # CHIA QUẦN THỂ ĐỂ TRỊ (Mini-batch) 
                for i in range(0, self.pop_size, batch_size):
                    end_idx = min(i + batch_size, self.pop_size)
                    curr_bs = end_idx - i
                    print(f"\r  -> Đang quét lô ảnh YOLO: {end_idx}/{self.pop_size}...", end="", flush=True)
                    
                    # Lấy từng miếng chunk
                    patch_chunk = self.population[i:end_idx]
                    
                    # Nếu chunk cuối bị lẻ thì xén base image lại
                    base_chunk = base_image_expanded[:curr_bs]
                    
                    # 1. Dán lô quần thể patch lên ảnh
                    adv_images = self.apply_patch_batch(base_chunk, patch_chunk)
                    
                    # 2. Áp dụng EOT KORNIA (Luôn chạy tính toán tọa độ ma trận trên mức Float32 mặc định để tránh lỗi phân bổ bộ nhớ AMP)
                    eot_images = apply_eot(adv_images)
                    
                    # 3. Yolo inference siêu việt bằng Tensor Half nội bộ trong Object
                    batched_scores = victim_model.get_raw_predictions(eot_images)
                    
                    # 4. CHẤM ĐIỂM BATCH NGAY TRÊN GPU VÀ BỎ FOR LẶP PYTHON CỦA CPU
                    chunk_fitness, _ = fitness_function(batched_scores, conf_thresh=0.01)
                    
                    # Gắn kết quả tensor trực tiếp vào danh sách mảng cuối cùng (với tốc độ ánh sáng)
                    fitness_scores.extend(chunk_fitness.tolist())
                        
                    # ===== ⚡ ÉP DỌN DẸP RÁC NGAY LẬP TỨC =====
                    del adv_images, eot_images, batched_scores, base_chunk, patch_chunk
                    # BỎ torch.cuda.empty_cache() ở đây! PyTorch tự cấp phát lại vùng nhớ cực nhanh. Gọi hàm này liên tục sẽ quét sạch cache và đập nát tốc độ (treo chờ đồng bộ).
            
            print() # Xuống dòng khi chạy hết population
            # 5. Tìm ra miếng dán "độc" nhất thế hệ hiện tại
            max_score = max(fitness_scores)
            best_idx = fitness_scores.index(max_score)
            best_patch = self.population[best_idx].clone()
            
            print(f"Thế hệ {gen+1}/{self.generations} | Fitness max: {max_score:.2f} | Tương đương ~{int(max_score/1.5)} objects giả")
            
            # 6. Chọn lọc và Lai ghép (Tối ưu hóa GPU hoàn toàn bằng Vectorization)
            scores_tensor = torch.tensor(fitness_scores, device=self.device)
            _, top_indices = torch.topk(scores_tensor, 5) # Lấy 5 con xịn nhất
            
            # Elitism: Trực tiếp giữ lại 5 cá thể ưu tú nhất trên VRAM
            elites = self.population[top_indices]
                
            # Sinh ra toàn bộ con mới cùng một lúc (Loại bỏ vòng lặp for của CPU)
            num_children = self.pop_size - 5
            
            # Lựa chọn cặp cha mẹ ngẫu nhiên từ top 5 cho toàn bộ children
            p1_indices = top_indices[torch.randint(0, 5, (num_children,), device=self.device)]
            p2_indices = top_indices[torch.randint(0, 5, (num_children,), device=self.device)]
            
            parents1 = self.population[p1_indices]
            parents2 = self.population[p2_indices]
            
            # Lai ghép cắt điểm giữa hàng loạt (Batched Crossover)
            split_point = self.patch_size // 2
            children = torch.empty_like(parents1)
            children[:, :, :split_point, :] = parents1[:, :, :split_point, :]
            children[:, :, split_point:, :] = parents2[:, :, split_point:, :]
            
            # Đột biến hàng loạt (Batched Mutation)
            # Tạo mask những cá thể sẽ bị đột biến
            mutation_mask = (torch.rand(num_children, device=self.device) < self.mutation_rate).view(-1, 1, 1, 1)
            noise = torch.rand_like(children) * 0.2 # Nhiễu 20%
            
            # Cập nhật đột biến song song lên các cá thể được chọn
            mutated_children = torch.clamp(children + noise, 0, 1)
            children = torch.where(mutation_mask, mutated_children, children)
                
            # Gộp lại thành quần thể mới trực tiếp trên GPU
            self.population = torch.cat([elites, children], dim=0)
            
            # Giải phóng tensor rác (Không gọi garbage collector và empty_cache để vòng lặp băng băng chạy)
            del children, mutated_children, noise, parents1, parents2
            
        print("[+] HOÀN THÀNH TIẾN HÓA!")
        return best_patch