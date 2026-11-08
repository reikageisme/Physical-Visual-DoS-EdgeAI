"""Read installed validation API and dataset configuration; no timings."""
import inspect
from pathlib import Path
import ultralytics
from ultralytics.models.yolo.segment import SegmentationValidator

if __name__=='__main__':
    root=Path('C:/sponge')
    (root/'validator_source.txt').write_text(inspect.getsource(SegmentationValidator)+'\n'+inspect.getsource(SegmentationValidator.__mro__[1]),encoding='utf-8')
    print((Path(ultralytics.__file__).parent/'cfg/datasets/coco.yaml').read_text(encoding='utf-8'))
    from ultralytics import YOLO
    model=YOLO(str(root/'yolo26n-seg.pt')).model
    head=model.model[-1]
    print('YOLO26_HEAD',type(head).__name__,'end2end',head.end2end,'one2one',getattr(head,'one2one_cv2',None) is not None,'nm',head.nm)
    print('HEAD_DEFAULTS',getattr(head,'_end2end',None),'model_end2end',model.end2end)
