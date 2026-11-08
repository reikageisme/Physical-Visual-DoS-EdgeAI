import time
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ASSETS
DEV='cuda'; NC,MASK_DIM,CANVAS=80,32,640; CONF,IOU=0.25,0.45; PRED_W=4+NC+MASK_DIM
def flat(o):
    if torch.is_tensor(o): yield o
    elif isinstance(o,(tuple,list)):
        for e in o: yield from flat(e)
    elif isinstance(o,dict):
        for e in o.values(): yield from flat(e)
def find_preds(raw):
    for t in flat(raw):
        if t.dim()==3 and t.shape[1]==PRED_W: return t
def load_img(p):
    from PIL import Image
    im=Image.open(p).convert('RGB').resize((CANVAS,CANVAS))
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2,0,1)[None]/255.0).to(DEV)
def soft(preds): return torch.sigmoid((preds[:,4:4+NC,:].amax(1)-CONF)/0.02).sum()
net=YOLO('yolov8n-seg.pt').model.eval().float().to(DEV)
for p in net.parameters(): p.requires_grad_(False)
img=load_img(sorted(ASSETS.glob('*.jpg'))[1])  # zidane
d=torch.zeros_like(img,requires_grad=True)
for _ in range(100):
    loss=soft(find_preds(net((img+d).clamp(0,1)))); loss.backward()
    with torch.no_grad():
        d.add_((2/255)*d.grad.sign()).clamp_(-16/255,16/255); d.data=(img+d).clamp(0,1)-img; d.grad.zero_()
adv=(img+d).clamp(0,1).detach()
with torch.no_grad(): preds=find_preds(net(adv)).detach()
ntau=int((preds[:,4:4+NC,:].amax(1)>CONF).sum())
print('N_tau',ntau,'shape',tuple(preds.shape),'contig',preds.is_contiguous(),flush=True)
# đo từng lần (giống gpu_decomp/gpu_mechanism)
for i in range(15):
    pc=preds.clone()
    torch.cuda.synchronize(); t0=time.perf_counter_ns()
    out=non_max_suppression(pc,CONF,IOU,nc=NC,max_det=300)
    torch.cuda.synchronize(); dt=(time.perf_counter_ns()-t0)/1e6
    print('rep',i,'ms=%.2f'%dt,'kept',int(len(out[0])),flush=True)
