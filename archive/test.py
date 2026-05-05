import torch
from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness

print("Đang tải mô hình MobileNet-SSD...")
victim = VictimModel()

# Tạo một tensor ảnh ảo (Nhiễu ngẫu nhiên) kích thước 320x320
dummy_image = torch.rand(1, 3, 320, 320).to(victim.device)

print("Đang ép mô hình phân tích ảnh...")
# Lấy dự đoán thô
outputs = victim.get_raw_predictions(dummy_image)

# Tính điểm Sponge Fitness
score, num_boxes = calculate_sponge_fitness(outputs, conf_thresh=0.02)

print(f"=============================")
print(f"SỐ LƯỢNG HỘP SINH RA: {num_boxes}")
print(f"ĐIỂM FITNESS: {score:.4f}")
print(f"=============================")