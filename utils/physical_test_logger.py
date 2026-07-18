"""
utils/physical_test_logger.py
─────────────────────────────────────────────────────────────────
Tool for systematic Physical Evaluation Matrix collection.
Helps the user manually collect FPS and CPU usage while holding the 
printed patch at different angles and distances.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import psutil
import pandas as pd
from core.victim_model import VictimModel

def main():
    print("="*60)
    print(" 📸 PHYSICAL TEST LOGGER for Q1/Q2 Evaluation Matrix")
    print("="*60)
    print("Instructions:")
    print("1. Hold the printed Sponge Patch in front of the webcam.")
    print("2. Enter the current angle (e.g., 0, 30, 45) and distance (1m, 2m).")
    print("3. The script will record FPS and NMS latency for 100 frames.")
    print("4. Results will be saved to outputs/physical_matrix.csv")
    print("="*60)
    
    distance = input("Enter distance (e.g. 1m, 2m, 3m): ").strip()
    angle = input("Enter angle in degrees (e.g. 0, 30, 45): ").strip()
    lighting = input("Enter lighting condition (e.g. normal, dark, bright): ").strip()
    
    if not distance or not angle:
        print("[!] Invalid input. Exiting.")
        return
        
    print("\n[*] Initializing camera and model...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Cannot open webcam 0.")
        return
        
    victim = VictimModel()
    
    print("\n[!] GET READY! Capturing in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
        
    print("\n[*] Recording 100 frames. Please hold still...")
    
    metrics = {
        'total_ms': [],
        'nms_ms': [],
        'raw_boxes': [],
        'fps': [],
        'cpu_percent': []
    }
    
    for frame_idx in range(100):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB, resize, and convert to tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (320, 320))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        cpu_usage = psutil.cpu_percent(interval=None)
        
        result = victim.get_predictions_with_nms(tensor, conf_thresh=0.01, profile_latency=True)
        
        total_ms = result['latency_ms']['total_ms']
        nms_ms = result['latency_ms']['nms_ms']
        raw_boxes = result['num_raw_boxes']
        fps = 1000.0 / (total_ms + 1e-5)
        
        metrics['total_ms'].append(total_ms)
        metrics['nms_ms'].append(nms_ms)
        metrics['raw_boxes'].append(raw_boxes)
        metrics['fps'].append(fps)
        metrics['cpu_percent'].append(cpu_usage)
        
        # Draw on frame to show user it's working
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"NMS: {nms_ms:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Boxes: {raw_boxes}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Physical Test Logger (Press 'q' to stop early)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate means
    import statistics
    mean_fps = statistics.mean(metrics['fps'])
    mean_nms = statistics.mean(metrics['nms_ms'])
    mean_boxes = statistics.mean(metrics['raw_boxes'])
    mean_cpu = statistics.mean(metrics['cpu_percent'])
    
    print("\n" + "="*40)
    print(" RECORDING COMPLETE")
    print("="*40)
    print(f"Condition : {distance}, {angle}°, {lighting}")
    print(f"Avg FPS   : {mean_fps:.1f}")
    print(f"Avg NMS   : {mean_nms:.2f} ms")
    print(f"Avg Boxes : {mean_boxes:.0f}")
    print(f"Avg CPU   : {mean_cpu:.1f}%")
    
    # Append to CSV
    csv_path = 'outputs/physical_matrix.csv'
    df_new = pd.DataFrame([{
        'Distance': distance,
        'Angle': angle,
        'Lighting': lighting,
        'Mean_FPS': mean_fps,
        'Mean_NMS_ms': mean_nms,
        'Mean_Raw_Boxes': mean_boxes,
        'Mean_CPU_percent': mean_cpu
    }])
    
    os.makedirs('outputs', exist_ok=True)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
        
    df_final.to_csv(csv_path, index=False)
    print(f"\n[+] Appended results to {csv_path}")

if __name__ == "__main__":
    import torch
    main()
