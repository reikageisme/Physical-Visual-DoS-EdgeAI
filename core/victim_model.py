import torch
import torch.nn.functional as F
from ultralytics import YOLO

class VictimModel:
    def __init__(self):
        # Thiết lập thiết bị chạy (Sử dụng RTX 5090)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tải mô hình YOLOv8n (Bản Nano cực phổ biến cho Camera Edge / Ubuntu Server)
        yolo = YOLO('yolov8n.pt')
        
        # HACK TỐI ĐA CPU: Móc thẳng cái "ruột" thuần PyTorch của YOLO ra ngoài.
        # Chuyển về chế độ eval và ép sang Float16 (Half) để tối đa hoá tốc độ tính toán.
        self.model = yolo.model.to(self.device).eval().half()

    def get_raw_predictions(self, image_tensor):
        """
        Nhận tensor ảnh (shape: [B, 3, H, W], giá trị 0-1)
        Trả về dự đoán đã lách qua NMS để lấy tối đa số object
        """
        with torch.no_grad():
            # YOLOv8 yêu cầu kích thước Tensor (H, W) phải chia hết cho 32 (stride lớn nhất)
            _, _, H, W = image_tensor.shape
            new_H = ((H + 31) // 32) * 32  
            new_W = ((W + 31) // 32) * 32
            
            # CHỐNG LAG: KHÔNG DÙNG F.interpolate THỦ CÔNG ĐỂ RESIZE ẢNH 720p 500 LẦN!
            # Pad viền đen (zero-padding) nhẹ nhàng siêu tốc thẳng trên GPU.
            if new_H != H or new_W != W:
                pad_h = new_H - H
                pad_w = new_W - W
                image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)

            # Ép kiểu dữ liệu về Float16 thẳng thừng trước khi tống vào YOLO 
            # Giữ an toàn cấu trúc PyTorch, không bị xung đột hàm lượng giác hình học như bên AMP Autocast
            image_tensor = image_tensor.half()
            
            # GỌI TRỰC TIẾP LÕI PYTORCH
            # Dòng này BỎ QUA 100% hệ thống Python rườm rà (NMS, Tracking, Dict objects) của thư viện Ultralytics
            preds = self.model(image_tensor)
            
            # Mô hình gốc trả về Tuple, lấy tensor đầu tiên mảng thiết lập chứa dự đoán thô 
            # Dạng mảng: [Batch_size, 84 features, 8400 hộp neo] (84 = 4 bounding box + 80 class xac suat)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
                
            # Cắt bỏ tọa độ (4 cột đầu), chỉ lấy độ phân loại tự tin (Confidence) của 80 cột còn lại
            cls_probs = preds[:, 4:, :]
            
            # Tóm thâu chỉ số tự tin lớn nhất của từng object tại từng box neo
            max_scores, _ = torch.max(cls_probs, dim=1)
            
        # Không cần dùng CPU Loop nữa, GPU Model 
        # Cứ trả về toàn bộ mảng [Batch_Size, 8400 hộp] dưới dạng Tensor nằm sẵn trên CUDA 
        return max_scores