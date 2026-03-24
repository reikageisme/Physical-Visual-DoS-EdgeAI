import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

class VictimModel:
    def __init__(self):
        # Thiết lập thiết bị chạy (Test trên PC trước)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tải trọng số chuẩn của SSDLite-MobileNetV3
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        
        # TUYỆT CHIÊU HACK NMS: 
        # Set score_thresh cực thấp (0.01) để lấy mọi hộp
        # Set nms_thresh = 1.0 (Cho phép các hộp đè lên nhau 100% mà không bị xóa)
        self.model = ssdlite320_mobilenet_v3_large(
            weights=weights, 
            score_thresh=0.01, 
            nms_thresh=1.0
        ).to(self.device)
        
        self.model.eval()

    def get_raw_predictions(self, image_tensor):
        """
        Nhận tensor ảnh (shape: [1, 3, 320, 320], giá trị 0-1)
        Trả về tất cả dự đoán thô (đã lách qua NMS)
        """
        with torch.no_grad():
            # Đầu ra của SSD trong torchvision là 1 list chứa dict
            # [{'boxes': tensor, 'labels': tensor, 'scores': tensor}]
            outputs = self.model(image_tensor)
            
        return outputs