import cv2
import torch
import numpy as np
import time
from flask import Flask, Response, render_template_string
from core.victim_model import VictimModel

app = Flask(__name__)

# Tải mô hình Victim Model
print("[*] Đang tải YOLOv8n...")
victim_model = VictimModel()

# Địa chỉ patch — thử nhiều đường dẫn, fallback sang noise pattern
PATCH_CANDIDATES = [
    "outputs/sponge_patch.png",
    "outputs/sponge_patch_A4_g100_p100.png",
]

patch_img = None
for path in PATCH_CANDIDATES:
    patch_img = cv2.imread(path)
    if patch_img is not None:
        print(f"[+] Đã tải Patch từ: {path}")
        break

if patch_img is not None:
    # Resize patch cho vừa với camera mô phỏng 720p
    patch_img = cv2.resize(patch_img, (200, 200))
else:
    # Fallback: tạo random noise pattern 200x200 nếu không tìm thấy patch nào
    print("[!] Không tìm thấy Patch. Dùng random noise pattern làm fallback.")
    patch_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

patch_height, patch_width, _ = patch_img.shape

def generate_frames():
    # Khởi động Camera (0 là camera mặc định)
    # Vì dùng Ubuntu Server i5 headless, cắm USB Webcam 720p vào port 0
    camera = cv2.VideoCapture(0)
    
    # Ép camera chạy ở độ phân giải HD 720p
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Gắn Patch vào giữa màn hình mô phỏng
        if patch_img is not None:
            h, w, _ = frame.shape
            y_offset = (h - patch_height) // 2
            x_offset = (w - patch_width) // 2
            
            # Gắn đè patch lên frame
            frame[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width] = patch_img

        # TIẾN HÀNH INFERENCE THEO THỜI GIAN THỰC GÂY QUÁ TẢI (DoS)
        # Chuyển đổi sang format pytorch (Giữ nguyên kích thước 720p hoặc YOLO sẽ tự nội bộ normalize)
        input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(victim_model.device)

        # Chạy suy luận bằng API get_predictions_with_nms (đo latency chi tiết)
        result = victim_model.get_predictions_with_nms(
            input_tensor, conf_thresh=0.01, profile_latency=True
        )
        num_raw_boxes = result['num_raw_boxes']
        num_final_boxes = result['num_final_boxes']
        nms_ms = result['latency_ms']['nms_ms']
        total_ms = result['latency_ms']['total_ms']
        fps = 1000.0 / (total_ms + 1e-5)

        # Trích xuất hiệu năng ghi lên màn hình (HUD overlay)
        color = (0, 0, 255) if num_raw_boxes > 200 else (0, 255, 0)
        cv2.putText(frame, f"FPS: {fps:.1f} | RAW BOXES: {num_raw_boxes} | FINAL: {num_final_boxes}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"NMS: {nms_ms:.1f}ms | TOTAL: {total_ms:.1f}ms",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "TARGET: INTEL CORE i5-2400 (UBUNTU SERVER)", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Encode sang JPG để Stream lân web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Visual DoS Monitor - Headless Server</title>
    <style>
        body { background-color: #1e1e1e; color: #fff; font-family: sans-serif; text-align: center; }
        h1 { color: #ff3333; }
        img { border: 5px solid #444; border-radius: 10px; max-width: 100%; height: auto; }
        .info { margin-top: 20px; font-size: 1.2em; color: #bbb; }
    </style>
</head>
<body>
    <h1>🔴 Live Visual DoS Attack Stream</h1>
    <img src="/video_feed" width="1280" height="720" />
    <div class="info">
        Server: Intel i5 2400 | OS: Ubuntu Server Headless | Stream: 720p Webcam<br>
        <i>(The stream will severely drop FPS and lag when the Sponge Patch is recognized)</i>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[*] Đang khởi động Web Server giám sát tại http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
