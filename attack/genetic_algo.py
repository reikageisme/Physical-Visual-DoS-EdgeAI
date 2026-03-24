import torch
import numpy as np
import cv2

class SpongeGA:
    def __init__(self, patch_size=64, pop_size=30, generations=50, mutation_rate=0.1):
        self.patch_size = patch_size
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Khởi tạo quần thể: 30 miếng dán nhiễu ngẫu nhiên (Giá trị từ 0.0 đến 1.0)
        # Shape: [pop_size, 3, patch_size, patch_size]
        self.population = torch.rand((self.pop_size, 3, self.patch_size, self.patch_size))

    def apply_patch(self, base_image, patch, x_offset=100, y_offset=100):
        """
        Dán miếng dán lên ảnh nền (Background image)
        """
        adv_image = base_image.clone()
        adv_image[0, :, y_offset:y_offset+self.patch_size, x_offset:x_offset+self.patch_size] = patch
        return adv_image

    def evolve(self, victim_model, fitness_function, base_image):
        print("\n[+] BẮT ĐẦU TIẾN HÓA MIẾNG DÁN SPONGE PATCH...")
        
        for gen in range(self.generations):
            fitness_scores = []
            
            # 1. Đánh giá toàn bộ quần thể
            for i in range(self.pop_size):
                patch = self.population[i]
                # Dán patch lên ảnh
                adv_image = self.apply_patch(base_image, patch)
                
                # Trích xuất RAW tensor từ AI
                outputs = victim_model.get_raw_predictions(adv_image)
                
                # Chấm điểm DoS
                score, num_boxes = fitness_function(outputs, conf_thresh=0.01)
                fitness_scores.append(score)
            
            # 2. Tìm ra miếng dán "độc" nhất thế hệ hiện tại
            max_score = max(fitness_scores)
            best_idx = fitness_scores.index(max_score)
            best_patch = self.population[best_idx].clone()
            
            print(f"Thế hệ {gen+1}/{self.generations} | Fitness max: {max_score:.2f} | Box sinh ra dự kiến: ~300")
            
            # 3. Chọn lọc và Lai ghép (Giữ lại top 5, lai tạo phần còn lại)
            # Chuyển list sang tensor để sort
            scores_tensor = torch.tensor(fitness_scores)
            _, top_indices = torch.topk(scores_tensor, 5) # Lấy 5 con xịn nhất
            
            new_population = []
            for idx in top_indices:
                new_population.append(self.population[idx]) # Giữ nguyên gen tốt (Elitism)
                
            # Sinh ra 25 con mới bằng cách lai ghép ngẫu nhiên từ top 5
            for _ in range(self.pop_size - 5):
                p1_idx = top_indices[torch.randint(0, 5, (1,))]
                p2_idx = top_indices[torch.randint(0, 5, (1,))]
                
                parent1 = self.population[p1_idx].squeeze(0)
                parent2 = self.population[p2_idx].squeeze(0)
                
                # Lai ghép cắt điểm giữa (Crossover)
                split_point = self.patch_size // 2
                child = torch.zeros_like(parent1)
                child[:, :split_point, :] = parent1[:, :split_point, :]
                child[:, split_point:, :] = parent2[:, split_point:, :]
                
                # Đột biến (Mutation): Thay đổi ngẫu nhiên một số pixel
                if torch.rand(1).item() < self.mutation_rate:
                    noise = torch.rand_like(child) * 0.2 # Nhiễu 20%
                    child = torch.clamp(child + noise, 0, 1)
                    
                new_population.append(child)
                
            # Cập nhật quần thể mới
            self.population = torch.stack(new_population)
            
        print("[+] HOÀN THÀNH TIẾN HÓA!")
        return best_patch