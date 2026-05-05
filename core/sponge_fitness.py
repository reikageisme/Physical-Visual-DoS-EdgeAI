import torch

def calculate_sponge_fitness(batch_scores, conf_thresh=0.01, lambda_weight=1.5):
    """
    Tối thượng vector hóa: Tính Fitness cho toàn bộ hàng trăm cá thể một lúc.
    batch_scores: Tensor GPU shape [Batch_Size, 8400]
    """
    # 1. Tạo mask cho những box vượt ngưỡng tự tin
    active_mask = batch_scores > conf_thresh
    
    # 2. Đếm số box hiện diện trên TỪNG ảnh trong lô batch (Không dùng vòng lặp for đếm từng ảnh)
    num_boxes_per_image = torch.sum(active_mask, dim=1) # Shape: [Batch_Size]
    
    # 3. Tính tổng điềm tin cậy (sum of confidence) TỪNG ảnh bằng phép nhân element-wise với mask
    filtered_scores = batch_scores * active_mask # Các box dưới mask thành số 0
    total_conf_per_image = torch.sum(filtered_scores, dim=1) # Shape: [Batch_Size]
    
    # 4. Trọng số lambda đánh giá DoS
    fitness_scores_gpu = total_conf_per_image + (lambda_weight * num_boxes_per_image)
    
    return fitness_scores_gpu, num_boxes_per_image