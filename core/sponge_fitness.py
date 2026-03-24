import torch

def calculate_sponge_fitness(outputs, conf_thresh=0.05, lambda_weight=1.5):
    """
    Tính điểm "vắt kiệt tài nguyên" cho MobileNet-SSD
    outputs: List chứa dict [{'boxes': ..., 'labels': ..., 'scores': ...}]
    """
    # Trích xuất mảng điểm tin cậy (scores) từ bức ảnh đầu tiên trong batch
    scores = outputs[0]['scores']
    
    # 1. Lọc các hộp có độ tin cậy vượt ngưỡng kích hoạt (conf_thresh)
    active_boxes = scores[scores > conf_thresh]
    num_boxes = len(active_boxes)
    
    # Nếu miếng dán không tạo ra hộp nào -> Phế thải, điểm = 0
    if num_boxes == 0:
        return 0.0
        
    # 2. Tính tổng điểm tin cậy của các hộp
    total_conf = torch.sum(active_boxes).item()
    
    # 3. Chấm điểm Fitness (Trọng số lambda ưu tiên số lượng hộp)
    fitness_score = total_conf + (lambda_weight * num_boxes)
    
    return fitness_score, num_boxes