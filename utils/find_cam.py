import cv2

print("=== ĐANG QUÉT TÌM CAMERA (VUI LÒNG CHỜ) ===")
# Quét từ cổng 0 đến 5
for i in range(5):
    # Dùng cv2.CAP_DSHOW để ép Windows nhận diện đúng driver Camera Ảo
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[+] THÀNH CÔNG: Tìm thấy Camera có hình ảnh tại ID = {i}")
            # Hiển thị thử cái hình lên xem có đúng luồng DroidCam không
            cv2.imshow(f"Test Cam ID {i}", frame)
            cv2.waitKey(3000) # Bật lên cho ông xem 3 giây rồi tự tắt
            cv2.destroyAllWindows()
        else:
            print(f"[-] ID {i}: Nhận diện được cổng nhưng đen/xanh không có luồng Video.")
        cap.release()
    else:
        print(f"[-] ID {i}: Cổng trống.")

print("=== QUÉT HOÀN TẤT ===")