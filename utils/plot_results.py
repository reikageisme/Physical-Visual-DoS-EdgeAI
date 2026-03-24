import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

def plot_performance(csv_file, output_path=None):
    """
    Đọc file CSV và vẽ biểu đồ thể hiện sự sụt giảm FPS và tăng CPU tải.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[-] Không tìm thấy file báo cáo: {csv_file}")
        return

    # Sinh trục thời gian dựa theo index nếu không parse được thời gian
    df['Time_Index'] = range(len(df))

    # Cấu hình biểu đồ
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 1. Biểu đồ FPS Sụt giảm
    ax1.plot(df['Time_Index'], df['FPS_Actual'], color='red', linewidth=2.5, label='Actual FPS')
    ax1.axhline(y=10.0, color='green', linestyle='--', label='Target FPS (10)')
    ax1.set_title('Tác động của Sponge Patch lên Tốc độ khung hình (FPS)', fontsize=14, weight='bold')
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 15)

    # 2. Biểu đồ Tải CPU
    ax2.plot(df['Time_Index'], df['CPU_Percent'], color='blue', linewidth=2, label='CPU Load (%)', alpha=0.8)
    ax2.set_title('Tải trọng CPU khi xử lý mô hình', fontsize=14, weight='bold')
    ax2.set_xlabel('Thời gian (Số khung hình)', fontsize=12)
    ax2.set_ylabel('CPU (%)', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    
    if output_path is None:
        # Nếu không có output flag, lưu cùng tên cùng thư mục
        base_name = os.path.basename(csv_file).replace('.csv', '.png')
        output_path = os.path.join(os.path.dirname(csv_file), base_name)
        
    plt.savefig(output_path, dpi=300)
    print(f"[+] Vẽ biểu đồ thành công: {output_path}")

def get_latest_log():
    list_of_files = glob.glob('logs/*.csv')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công Cụ Trực Quan Hóa Kết Quả Sponge Attack")
    parser.add_argument('--file', type=str, help='Đường dẫn đến file CSV cần vẽ.')
    parser.add_argument('--out', type=str, help='Nơi lưu biểu đồ PNG.')
    
    args = parser.parse_args()
    
    target_file = args.file if args.file else get_latest_log()
    
    if target_file:
        print(f"[*] Đang phân tích dữ liệu: {target_file}")
        plot_performance(target_file, args.out)
    else:
        print("[-] Không tìm thấy dữ liệu logs nào. Hãy chạy test_physical_dos.py trước!")
