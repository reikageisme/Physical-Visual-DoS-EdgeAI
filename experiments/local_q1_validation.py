"""Bounded, reproducible CPU validation; results are local digital experiments.

python -u experiments/local_q1_validation.py --out-dir outputs/local_q1_20261002
Uses the installed Ultralytics bus/zidane example photos, not a benchmark dataset.
All optimizer methods use the same image, area, query count and EOT implementation.
Checkpointed seeds are resumed only when their protocol matches the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torchvision
import ultralytics
from PIL import Image
from attack.genetic_algo import SpongeGA
from core.eot_transforms import apply_eot
from core.sponge_fitness import calculate_sponge_fitness
from core.victim_model import VictimModel


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summary(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "median": float(np.median(a)),
            "q1": float(np.quantile(a, .25)), "q3": float(np.quantile(a, .75))}


def load_image(path):
    # Fixed direct resize, RGB [0,1]; intentionally not Ultralytics letterboxing.
    image = Image.open(path).convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
    return (torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255).contiguous()


def cpu_seconds(process):
    t = process.cpu_times()
    return t.user + t.system


def predict(victim, image, threshold, cap=None):
    start = time.perf_counter_ns()
    with torch.no_grad():
        predictions = victim.model(image)
        predictions = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
        pred = predictions[0]
        forward_end = time.perf_counter_ns()
        scores, classes = pred[4:].max(dim=0)
        mask = scores > threshold
        boxes = victim._xywh_to_xyxy(pred[:4, mask].T.float())
        scores = scores[mask].float()
        active = len(scores)
        if cap is not None and active > cap:
            chosen = torch.topk(scores, cap).indices
            boxes, scores = boxes[chosen], scores[chosen]
        nms_start = time.perf_counter_ns()
        indices = victim._nms(boxes, scores, .45)[:300]
        end = time.perf_counter_ns()
    return {"raw_locations": pred.shape[1], "active_candidates": active,
            "nms_input_candidates": len(scores), "final_detections": len(indices),
            "forward_ms": (forward_end-start)/1e6,
            "filter_ms": (nms_start-forward_end)/1e6,
            "nms_ms": (end-nms_start)/1e6, "total_ms": (end-start)/1e6}


def benchmark_nms(out, seeds):
    path = out / "nms_samples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(10000+seed)
        order = [50, 100, 500, 1000, 2100, 3000, 8400]
        random.Random(seed).shuffle(order)
        for n in order:
            xy = torch.rand((n, 2), generator=generator)*320
            size = torch.rand((n, 2), generator=generator)*40+10
            boxes = torch.cat([(xy-size/2).clamp(0,320), (xy+size/2).clamp(0,320)],1)
            scores = torch.rand(n, generator=generator)
            for _ in range(5):
                torchvision.ops.nms(boxes, scores, .45)
            samples=[]
            for _ in range(20):
                start=time.perf_counter_ns()
                kept=torchvision.ops.nms(boxes,scores,.45)
                samples.append((time.perf_counter_ns()-start)/1e6)
            rows.append({"seed":seed,"N":n,"pairs":n*(n-1)//2,
                         "kept":len(kept),"latency_ms_samples":samples,
                         "latency_ms_median":statistics.median(samples)})
        print(f"NMS seed {seed} complete",flush=True)
    save_json(path,rows)
    return rows


def optimize(victim, image, method, seed, protocol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ga = SpongeGA(patch_size=64, pop_size=protocol["population"],
                  generations=protocol["generations"], seed=seed,
                  use_saliency=method!="ga_center", elite_k=5,
                  convergence_patience=protocol["generations"]+1, convergence_delta=0.0)
    started=time.perf_counter()
    if method != "random_search":
        patch=ga.evolve(victim,calculate_sponge_fitness,image,batch_size=protocol["batch_size"])
        result=ga.get_run_summary()
        history=result["fitness_history"]
        location=result["patch_location"]
        score=max(history)
    else:
        location=ga._compute_saliency_location(image)
        score=-float("inf")
        history=[]
        # Common first population and common one-EOT-per-query protocol.
        for generation in range(protocol["generations"]):
            population=ga.population if generation==0 else torch.rand_like(ga.population)
            fitness=[]
            for i in range(0,len(population),protocol["batch_size"]):
                chunk=population[i:i+protocol["batch_size"]]
                adv=ga.apply_patch_batch(image.expand(len(chunk),-1,-1,-1),chunk,*location)
                with torch.no_grad():
                    fit,_=calculate_sponge_fitness(victim.get_raw_predictions(apply_eot(adv)))
                fitness.extend(fit.tolist())
            best=int(np.argmax(fitness))
            history.append(float(fitness[best]))
            if fitness[best]>score:
                score=float(fitness[best]); patch=population[best].clone().cpu()
            print(f"RS seed {seed} generation {generation+1}: {score:.3f}",flush=True)
    return patch, {"seed":seed,"method":method,"best_eot_fitness":score,
                   "generation_best":history,"running_best":np.maximum.accumulate(history).tolist(),
                   "queries":protocol["population"]*len(history),
                   "wall_seconds":time.perf_counter()-started,"location":list(location)}


def evaluate(victim, images, patches, seed, repeats):
    rows=[]
    rng=np.random.default_rng(seed+20000)
    process=psutil.Process()
    for image_name,base in images.items():
        controls={"clean":base}
        loc=SpongeGA(patch_size=64,pop_size=5,generations=1,seed=seed)._compute_saliency_location(base)
        noise=torch.from_numpy(rng.random((3,64,64),dtype=np.float32))
        checker=torch.from_numpy(((np.indices((64,64)).sum(axis=0)//4)%2).astype(np.float32)).repeat(3,1,1)
        for name,patch in [("noise",noise),("checkerboard",checker),("solid",torch.ones(3,64,64)*.5)]:
            x=base.clone(); y,z=loc; x[:,:,y:y+64,z:z+64]=patch; controls[name]=x
        for method,(patch,location) in patches.items():
            x=base.clone(); y,z=location; x[:,:,y:y+64,z:z+64]=patch; controls[method]=x
        conditions=[(name,tau,cap) for name in controls for tau,cap in [(.01,None),(.25,None),(.01,100)]]
        random.Random(seed+30000).shuffle(conditions)
        for name,tau,cap in conditions:
            x=controls[name]
            for _ in range(3):
                predict(victim,x,tau,cap)
            begin_cpu=cpu_seconds(process); begin=time.perf_counter()
            samples=[predict(victim,x,tau,cap) for _ in range(repeats)]
            wall=time.perf_counter()-begin
            cpu=cpu_seconds(process)-begin_cpu
            row={"seed":seed,"image":image_name,"condition":name,"threshold":tau,"pre_nms_cap":cap,
                 "repeats":repeats,"cpu_percent_one_core":100*cpu/wall,
                 "cpu_percent_machine_normalized":100*cpu/wall/psutil.cpu_count(),
                 "rss_mib":process.memory_info().rss/2**20,"samples":samples}
            for key in samples[0]:
                row[key]=statistics.median([s[key] for s in samples])
            row["throughput_fps"]=repeats/wall
            rows.append(row)
    return rows


def make_figures(out, nms, seeds, eval_rows):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,ax=plt.subplots(figsize=(6.8,4.0))
    ns=sorted({r["N"] for r in nms})
    values=[[r["latency_ms_median"] for r in nms if r["N"]==n] for n in ns]
    ax.errorbar(ns,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) for v in values],fmt="o-",capsize=3)
    ax.set(xlabel="Số hộp tổng hợp đưa vào NMS",ylabel="Độ trễ NMS (ms)")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"nms_local.png",dpi=220); plt.close(fig)
    labels={"sg_ga":"GA vị trí nổi bật","ga_center":"GA vị trí trung tâm","random_search":"Tìm kiếm ngẫu nhiên"}
    fig,ax=plt.subplots(figsize=(6.8,4.0))
    for method in labels:
        a=np.array([r["running_best"] for r in seeds if r["method"]==method])
        q=np.arange(1,a.shape[1]+1)*20
        mean=a.mean(0); std=a.std(0,ddof=1)
        ax.plot(q,mean,label=labels[method]); ax.fill_between(q,mean-std,mean+std,alpha=.15)
    ax.set(xlabel="Số truy vấn ảnh (cùng ngân sách)",ylabel="Fitness EOT tốt nhất đã quan sát")
    ax.legend(fontsize=9); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"convergence_local.png",dpi=220); plt.close(fig)
    conditions=["clean","noise","checkerboard","solid","sg_ga","ga_center","random_search"]
    names=["Sạch","Nhiễu","Ô bàn cờ","Màu đặc","GA nổi bật","GA trung tâm","Ngẫu nhiên"]
    for image_name in ["bus","zidane"]:
        fig,axes=plt.subplots(1,2,figsize=(10,4.0))
        for ax,metric,title in zip(axes,["active_candidates","nms_ms"],["Ứng viên sau ngưỡng","Độ trễ NMS (ms)"]):
            for tau,offset,label in [(.01,-.18,"Ngưỡng 0,01"),(.25,.18,"Ngưỡng 0,25")]:
                v=[[r[metric] for r in eval_rows if r["image"]==image_name and r["condition"]==c and r["threshold"]==tau and r["pre_nms_cap"] is None] for c in conditions]
                ax.bar(np.arange(len(v))+offset,[np.mean(x) for x in v],.36,yerr=[np.std(x,ddof=1) for x in v],label=label,capsize=2)
            ax.set_xticks(range(len(names)),names,rotation=35,ha="right"); ax.set_ylabel(title); ax.grid(axis="y",alpha=.2)
        axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/f"candidates_latency_{image_name}.png",dpi=220); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",default="outputs/local_q1_20261002")
    parser.add_argument("--seeds",type=int,default=10)
    parser.add_argument("--generations",type=int,default=20)
    args=parser.parse_args()
    out=ROOT/args.out_dir; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assets=Path(ultralytics.__file__).parent/"assets"
    paths={name:assets/f"{name}.jpg" for name in ["bus","zidane"]}
    images={name:load_image(path) for name,path in paths.items()}
    protocol={"seeds":args.seeds,"generations":args.generations,"population":20,"batch_size":4,
              "queries_per_method_seed":20*args.generations,"threads":2,"device":"cpu",
              "input_hw":[320,320],"patch_hw":[64,64],"training_image":"bus",
              "held_out_image":"zidane","fitness_threshold":.01,"fitness_lambda":1.5,
              "eot_draws_per_image_query":1,"evaluation_repeats":5,"warmup_repeats":3,
              "iou_threshold":.45,"max_det_after_nms":300,"pipeline_nms":"class_agnostic_custom_PyTorch",
              "resize":"PIL bilinear direct 320x320 RGB; no letterbox",
              "nms_benchmark":"torchvision.ops.nms; 10 independent box seeds, 20 repeats each"}
    manifest_path=out/"manifest.json"
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["protocol"]!=protocol or existing["script_sha256"]!=sha256(__file__):
            raise RuntimeError("Existing run has a different protocol/script; use a new output directory.")
    else:
        cpu_info=subprocess.check_output(["powershell","-NoProfile","-Command","Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json"],text=True)
        manifest={"started_utc":datetime.now(timezone.utc).isoformat(),"protocol":protocol,
                  "python":sys.version,"executable":sys.executable,"platform":platform.platform(),
                  "processor":json.loads(cpu_info),"ram_bytes":psutil.virtual_memory().total,
                  "versions":{"torch":torch.__version__,"torchvision":torchvision.__version__,
                              "ultralytics":ultralytics.__version__,"opencv":cv2.__version__,
                              "numpy":np.__version__,"psutil":psutil.__version__,
                              "kornia":importlib.metadata.version("kornia"),"matplotlib":matplotlib.__version__},
                  "distribution_metadata":{m:importlib.metadata.version(m) for m in ["torch","torchvision","ultralytics","opencv-python"]},
                  "weights_sha256":sha256(ROOT/"yolov8n.pt"),"script_sha256":sha256(__file__),
                  "sources_sha256":{p:sha256(ROOT/p) for p in ["core/victim_model.py","core/sponge_fitness.py","core/eot_transforms.py","attack/genetic_algo.py"]},
                  "images":{name:{"source":str(path),"sha256":sha256(path),"role":"optimization and in-sample" if name=="bus" else "held-out single example"} for name,path in paths.items()},
                  "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  "limitations":["two example photographs, not a benchmark dataset","no detection ground truth or mAP","no physical printed patch","CPU controlled to two software threads, not a Raspberry Pi emulator","EOT best score is a stochastic selection statistic, not independently validated expected fitness"]}
        save_json(manifest_path,manifest)
    victim=VictimModel(device="cpu")
    raw_shapes={name:list(victim.get_raw_predictions(image).shape) for name,image in images.items()}
    save_json(out/"observed_shapes.json",raw_shapes)
    for _ in range(5):
        victim.get_raw_predictions(images["bus"])
    nms=benchmark_nms(out,range(args.seeds))
    runs=[]; eval_rows=[]
    for seed in range(args.seeds):
        patches={}
        methods=["sg_ga","ga_center","random_search"]
        random.Random(seed).shuffle(methods)
        for method in methods:
            run_path=out/f"seed_{seed}_{method}.json"
            patch_path=out/f"patch_{seed}_{method}.png"
            tensor_path=out/f"patch_{seed}_{method}.npy"
            if run_path.exists() and tensor_path.exists():
                result=json.loads(run_path.read_text(encoding="utf-8"))
                patch=torch.from_numpy(np.load(tensor_path))
            else:
                patch,result=optimize(victim,images["bus"],method,seed,protocol)
                np.save(tensor_path,patch.numpy())
                Image.fromarray((patch.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)).save(patch_path)
                result["protocol"]=protocol
                save_json(run_path,result)
            runs.append(result); patches[method]=(patch,result["location"])
        eval_path=out/f"evaluation_seed_{seed}.json"
        if eval_path.exists():
            rows=json.loads(eval_path.read_text(encoding="utf-8"))
        else:
            rows=evaluate(victim,images,patches,seed,protocol["evaluation_repeats"])
            save_json(eval_path,rows)
        eval_rows.extend(rows)
        if seed==0:
            fig,axes=plt.subplots(1,3,figsize=(9,3.4))
            for ax,(name,base) in zip(axes,[('Sạch',images['bus']),('GA nổi bật',images['bus']),('Ảnh kiểm tra',images['zidane'])]):
                x=base.clone()
                if name=='GA nổi bật':
                    patch,location=patches['sg_ga']; y,z=location; x[:,:,y:y+64,z:z+64]=patch
                ax.imshow(x[0].permute(1,2,0).numpy()); ax.set_title(name); ax.axis('off')
            fig.tight_layout(); fig.savefig(out/'image_examples.png',dpi=220); plt.close(fig)
        print(f"SEED {seed} FINISHED ({seed+1}/{args.seeds})",flush=True)
    aggregate={"optimizer":{m:{k:summary([r[k] for r in runs if r['method']==m]) for k in ['best_eot_fitness','queries','wall_seconds']} for m in ['sg_ga','ga_center','random_search']},
               "nms":{str(n):summary([r['latency_ms_median'] for r in nms if r['N']==n]) for n in sorted({r['N'] for r in nms})},"evaluation":[]}
    groups=sorted({(r['image'],r['condition'],r['threshold'],-1 if r['pre_nms_cap'] is None else r['pre_nms_cap']) for r in eval_rows})
    for image,condition,tau,cap in groups:
        group=[r for r in eval_rows if (r['image'],r['condition'],r['threshold'],-1 if r['pre_nms_cap'] is None else r['pre_nms_cap'])==(image,condition,tau,cap)]
        aggregate['evaluation'].append({"image":image,"condition":condition,"threshold":tau,"cap":None if cap==-1 else cap,
            **{k:summary([r[k] for r in group]) for k in ['raw_locations','active_candidates','nms_input_candidates','final_detections','forward_ms','nms_ms','total_ms','throughput_fps','cpu_percent_machine_normalized','rss_mib']}})
    save_json(out/'aggregate.json',aggregate)
    flat=[{k:v for k,v in row.items() if k!='samples'} for row in eval_rows]
    with (out/'evaluation.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    make_figures(out,nms,runs,eval_rows)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    manifest['completion']={"optimizer_runs":len(runs),"evaluation_blocks":len(eval_rows),"nms_seed_size_blocks":len(nms)}
    save_json(manifest_path,manifest)
    print(f"COMPLETE: {out}",flush=True)


if __name__ == '__main__':
    main()
