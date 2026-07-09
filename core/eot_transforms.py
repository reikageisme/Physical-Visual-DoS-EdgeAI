import torch
import kornia.augmentation as K

# TUYỆT CHIÊU TỐI ƯU CỰC ĐẠI THEO THỜI GIAN THỰC: 
# Đưa biến list khởi tạo Kornia ra ngoài toàn cục (Global) để KHÔNG PHẢI tạo lại Class 500 lần mỗi thế hệ.
# Thiết lập same_on_batch=False để EOT tác động ngẫu nhiên trên từng ảnh trong Batch.
AUG_LIST = K.AugmentationSequential(
    # Perspective (Phối cảnh) tính toán trên ảnh 720p HD mất cực kì nhiều thời gian do Grid Sample.
    # Tắt Perspective đi hoặc dùng Affine xoay 3D nhẹ hơn. Ở đây ta dùng Rotation và Jitter là chuẩn xác nhất cho DoS.
    K.RandomRotation(degrees=15.0, p=0.8),
    K.ColorJitter(brightness=0.2, contrast=0.2, p=0.8),
    K.RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.5),
    data_keys=["input"],
    same_on_batch=False
)

def apply_eot(batched_adv_images):
    """
    Áp dụng Expectation over Transformation (EoT) 100% bằng thư viện lõi Kornia trên GPU.
    """
    # Chuyển ảnh qua Kornia (siêu nhanh vì Class AUG_LIST đã nằm sẵn trên RAM)
    transformed = AUG_LIST(batched_adv_images)
    
    # Nhiễu cảm biến Kornia bằng hàm torch
    noise = torch.randn_like(transformed) * 0.02
    transformed = torch.clamp(transformed + noise, 0.0, 1.0)
    
    return transformed
