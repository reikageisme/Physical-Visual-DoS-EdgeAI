"""Bảng chi phí phòng thủ: mAP (hộp + mask) dưới các biện pháp giới hạn chi phí mask, trên coco128-seg.
workers=0 + guard __main__ để tránh treo DataLoader đa tiến trình trên Windows. coco128 là chỉ báo, không phải COCO val đầy đủ."""
import json
from ultralytics import YOLO

CONF = dict(data='coco128-seg.yaml', device=0, verbose=False, plots=False, save_json=False, workers=0)
CONFIGS = [
    ('baseline (max_det=300, proto-res)', dict()),
    ('defense max_det=50', dict(max_det=50)),
    ('defense max_det=10', dict(max_det=10)),
    ('retina_masks=True (full-res, dat hon)', dict(retina_masks=True)),
    ('defense imgsz=320', dict(imgsz=320)),
]


def main():
    out = {}
    for mname in ['yolov8n-seg.pt', 'yolo11n-seg.pt']:
        m = YOLO(mname)
        out[mname] = {}
        for name, kw in CONFIGS:
            r = m.val(**CONF, **kw)
            out[mname][name] = dict(box_map=round(float(r.box.map), 4), box_map50=round(float(r.box.map50), 4),
                                    seg_map=round(float(r.seg.map), 4), seg_map50=round(float(r.seg.map50), 4))
            print(mname, '|', name, '-> seg_map=%.4f box_map=%.4f' % (out[mname][name]['seg_map'], out[mname][name]['box_map']), flush=True)
    json.dump(out, open(r'C:\sponge\gpu_defense.json', 'w'), indent=1)
    print('WRITTEN C:\\sponge\\gpu_defense.json', flush=True)


if __name__ == '__main__':
    main()
