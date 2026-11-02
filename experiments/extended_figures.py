"""Vietnamese figures rendered exclusively from completed extension records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = dict(sg_ga='GA nổi bật',ga_center='GA trung tâm',ga_random_location='GA vị trí ngẫu nhiên',random_search='Tìm kiếm ngẫu nhiên')
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bars(axis,rows,label,statistic,reference=None):
    values = [r[statistic]['mean'] for r in rows]
    intervals = np.asarray([r[statistic]['conditional_paired_seed_bootstrap_95ci'] for r in rows])
    axis.bar(np.arange(len(rows)),values,color=[COLORS[i%4] for i in range(len(rows))],alpha=.85)
    axis.errorbar(np.arange(len(rows)),values,yerr=np.stack([np.asarray(values)-intervals[:,0],intervals[:,1]-values]),
                  fmt='none',ecolor='#333333',capsize=3)
    axis.set_xticks(np.arange(len(rows)),label,rotation=18,ha='right')
    if reference is not None:
        axis.axhline(reference,color='#555555',linestyle='--',linewidth=1)
    axis.grid(axis='y',alpha=.2)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    run = ROOT/args.run_dir
    analysis_path = run/'analysis.json'
    data = json.loads(analysis_path.read_text(encoding='utf-8'))
    output = run/'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metadata = {}

    def save(name,figure,description):
        figure.tight_layout()
        path = output/(name+'.png')
        figure.savefig(path,dpi=200,bbox_inches='tight')
        plt.close(figure)
        metadata[path.name] = dict(sha256=digest(path),description_vi=description,
                                   uncertainty='95% bootstrap ghép cặp theo 10 seed, điều kiện trên bộ ảnh cố định; không gộp ảnh/lượt đo thành seed')

    names = list(LABELS)
    rows = [next(r for r in data['methods'] if r['name']==n and r['budget']==2500) for n in names]
    fig,ax = plt.subplots(figsize=(8,4))
    bars(ax,rows,[LABELS[n] for n in names],'candidate_ratio',1)
    ax.set_ylabel('Tỷ số ứng viên so với ảnh sạch')
    ax.set_title('2.500 truy vấn ảnh; ngưỡng 0,01; không giới hạn trước NMS')
    save('ty_so_ung_vien',fig,'Bốn phương pháp, 10 seed, 128 ảnh kiểm tra; đường đứt tại 1 là ảnh sạch. Không phải tỷ số độ trễ hay xác nhận DoS.')

    fig,ax = plt.subplots(figsize=(8,4))
    for name,color in zip(names,COLORS):
        rows = sorted([r for r in data['same_image_budget_curves'] if r['name']==name],key=lambda r:r['budget'])
        x = [r['budget'] for r in rows]
        y = [r['statistics']['mean'] for r in rows]
        ci = np.asarray([r['statistics']['conditional_paired_seed_bootstrap_95ci'] for r in rows])
        ax.plot(x,y,'o-',color=color,label=LABELS[name])
        ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.12)
    ax.axhline(1,color='#555555',linestyle='--',linewidth=1)
    ax.set_xlabel('Truy vấn ảnh thực tế mỗi seed')
    ax.set_ylabel('Tỷ số ứng viên trên cùng 32 ảnh')
    ax.legend(frameon=False,fontsize=8)
    ax.grid(alpha=.2)
    save('ngan_sach_truy_van',fig,'Mọi điểm dùng đúng 32 ID ảnh chung; vùng mờ là khoảng tin cậy theo seed, không phải độ lệch chuẩn.')

    area_names = ['area_1','area_2','sg_ga','area_8','area_16']
    sizes = [32,45,64,90,128]
    rows = [next(r for r in data['methods'] if r['name']==name and r['budget']==400) for name in area_names]
    fig,ax = plt.subplots(figsize=(8,4))
    bars(ax,rows,[f'{size*size/320**2*100:.2f}%'.replace('.',',') for size in sizes],'candidate_ratio',1)
    ax.set_xlabel('Diện tích miếng dán thực tế trên ảnh 320×320')
    ax.set_ylabel('Tỷ số ứng viên so với ảnh sạch')
    ax.set_title('Loại bỏ thành phần: cùng 400 truy vấn ảnh, 10 seed/mức')
    save('dien_tich_mieng_dan',fig,'Diện tích 1/1,98/4/7,91/16%, cùng ảnh tối ưu/kiểm tra và ngân sách; mức 4% dùng checkpoint SG-GA tại 400.')

    eot_names = ['sg_ga','eot_none','eot_no_rotation','eot_no_color','eot_no_blur','eot_no_noise','eot_four_draws']
    texts = ['Đầy đủ','Không EOT','Bỏ xoay','Bỏ màu','Bỏ mờ','Bỏ nhiễu','4 mẫu/ảnh']
    rows = [next(r for r in data['methods'] if r['name']==name and r['budget']==400) for name in eot_names]
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    bars(axes[0],rows,texts,'candidate_ratio',1)
    axes[0].set_ylabel('Tỷ số ứng viên không EOT khi kiểm tra')
    independent = [next(r for r in data['independent_eot'] if r['name']==name and r['budget']==400) for name in eot_names]
    bars(axes[1],independent,texts,'independent_fitness_difference',0)
    axes[1].set_ylabel('Chênh lệch fitness EOT độc lập với ảnh sạch')
    save('loai_bo_eot',fig,'Bảy cấu hình cùng 400 lượt ảnh qua mô hình; bốn mẫu EOT có ít cá thể hơn. Kiểm tra độc lập dùng tám ảnh, 16 mẫu ghép cặp mới/ảnh, không chọn lại miếng dán.')

    if data['transfer_exploratory']:
        fig,ax = plt.subplots(figsize=(9,4))
        rows = data['transfer_exploratory']
        bars(ax,rows,[r['model'].replace('yolov8','YOLOv8').replace('detr','DETR') for r in rows],
             'within_model_paired_latency_ratio',1)
        ax.set_ylabel('Tỷ số độ trễ có miếng dán/sạch trong từng mô hình')
        save('chuyen_giao',fig,'Cùng 16 ảnh và 10 miếng dán nguồn; bao gồm chuẩn hóa/hậu xử lý riêng của mô hình. Không so sánh độ trễ tuyệt đối giữa các kiến trúc; DETR không có NMS.')

    fig,axes = plt.subplots(1,2,figsize=(11,4))
    configs = [('clean',0,'Sạch'),('sg_ga',400,'GA nổi bật 400'),('sg_ga',2500,'GA nổi bật 2.500'),('adaptive_cap100',400,'Thích nghi 400')]
    for name,budget,label in configs:
        for panel,cap in enumerate([None,100]):
            rows = sorted([r for r in data['defense_tradeoffs'] if r['name']==name and r['budget']==budget and r['pre_nms_cap']==cap],key=lambda r:r['threshold'])
            axes[panel].plot([r['postprocess_mean_over_seed_ms'] for r in rows],
                             [100*r['delta_AP_vs_same_patch_undefended']['mean'] for r in rows],'o-',label=label)
            for row in rows:
                axes[panel].annotate(str(row['threshold']).replace('.',','),
                    (row['postprocess_mean_over_seed_ms'],100*row['delta_AP_vs_same_patch_undefended']['mean']),fontsize=7)
    for ax,cap in zip(axes,[None,100]):
        ax.set_title('Không giới hạn' if cap is None else 'Giới hạn 100 trước NMS')
        ax.set_xlabel('Thời gian hậu xử lý trung bình (ms)')
        ax.set_ylabel('Thay đổi AP so với cùng miếng dán không phòng thủ (điểm %)')
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    save('danh_doi_phong_thu',fig,'AP được tính gộp ảnh bằng COCOeval, sau đó trung bình theo miếng dán/seed. Chỉ so sánh thích nghi với SG-GA cùng 400 truy vấn; đồ thị mô tả, không mang thanh sai số.')
    metadata['danh_doi_phong_thu.png']['uncertainty'] = 'Đồ thị mô tả trung bình theo seed; không có thanh sai số. Khoảng tin cậy chênh lệch AP/AR có trong analysis.json.'

    if data['processing_budget_exploratory']:
        fig,ax = plt.subplots(figsize=(8,4))
        for name in ['clean']+names:
            rows = sorted([r for r in data['processing_budget_exploratory'] if r['condition']==name],key=lambda r:r['cpu_threads'])
            y = [100*r['late_fraction']['mean'] for r in rows]
            ci = 100*np.asarray([r['late_fraction']['conditional_paired_seed_bootstrap_95ci'] for r in rows])
            x = [r['cpu_threads'] for r in rows]
            ax.plot(x,y,'o-',label='Sạch' if name=='clean' else LABELS[name])
            ax.fill_between(x,ci[:,0],ci[:,1],alpha=.10)
        ax.set_xlabel('Số luồng tính toán trên cùng một máy thật')
        ax.set_ylabel('Lượt xử lý vượt ngân sách 300 ms (%)')
        ax.set_xticks([1,2,4])
        ax.set_ylim(-1,101)
        ax.legend(frameon=False,fontsize=8)
        ax.grid(alpha=.2)
        save('ngan_sach_xu_ly',fig,'Một máy vật lý, 10 khối ghép cặp, 16 ảnh, ba lượt đo. Suy luận + giải mã dự đoán + lọc + NMS được đo chung; không phải thời hạn hay tuổi khung camera.')

    if data['public_reference_variants_exploratory']:
        fig,ax = plt.subplots(figsize=(8,4))
        rows = data['public_reference_variants_exploratory']
        bars(ax,rows,[r['method'].replace('phantom_sponges','Phantom Sponges (tham chiếu)').replace('overload','Overload (tham chiếu)') for r in rows],
             'within_reference_candidate_ratio',1)
        ax.set_ylabel('Tỷ số ứng viên so với ảnh sạch — YOLOv5s, 640')
        save('bien_the_tham_chieu',fig,'Chạy hàm nguyên trạng từ kho Underload, chuyển môi trường sang CPU/runtime hiện tại; UAP cộng toàn ảnh, mô hình/kích thước/tập tối ưu/ngân sách khác nhóm miếng dán cục bộ. Không tái lập toàn bộ thí nghiệm bài gốc; không xếp hạng hai đe dọa như cùng điều kiện.')
    manifest = dict(script_sha256=digest(__file__),analysis_sha256=digest(analysis_path),figures=metadata,
                     generated_only_from_complete_records=True,no_synthetic_or_generated_evidence_images=True)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FIGURES COMPLETE: '+str(len(metadata)),flush=True)


if __name__=='__main__':
    main()
