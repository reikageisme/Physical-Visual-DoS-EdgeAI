"""Bounded, validation-only tuning with project-local evidence and native sampling."""
from __future__ import annotations
import argparse, csv, json, random, shutil, sys, time
from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from pipeline_utils import PROJECT_ROOT, configure_project_environment, require_local_venv, project_path, sha256_file, utc_timestamp, write_json
configure_project_environment()
from fullscale_train_one import train_arguments, read_history, metric_summary, Tee

CONFIGS = {
    "original_sgd": {"epochs":300,"lr0":0.01,"lrf":0.01,"cos_lr":False,"warmup_epochs":3.0},
    "low_sgd_cosine": {"epochs":6,"lr0":0.001,"lrf":0.1,"cos_lr":True,"warmup_epochs":1.0,"warmup_bias_lr":0.001,"close_mosaic":2},
    "adamw_cosine": {"epochs":6,"optimizer":"AdamW","lr0":0.0003,"lrf":0.1,"cos_lr":True,"warmup_epochs":1.0,"warmup_bias_lr":0.0003,"close_mosaic":2},
    "adamw_domain_loss_aug": {"epochs":6,"optimizer":"AdamW","lr0":0.0003,"lrf":0.1,"cos_lr":True,"warmup_epochs":1.0,"warmup_bias_lr":0.0003,"close_mosaic":2,"box":9.0,"cls":0.7,"dfl":1.8,"mosaic":0.5,"mixup":0.05,"hsv_s":0.5,"hsv_v":0.3,"scale":0.3,"degrees":5.0,"translate":0.1},
    "adamw_transfer_restored": {"epochs":6,"optimizer":"AdamW","lr0":0.0003,"lrf":0.1,"cos_lr":True,"warmup_epochs":1.0,"warmup_bias_lr":0.0003,"close_mosaic":2},
    "transfer_tiny_lr": {"epochs":6,"optimizer":"AdamW","lr0":0.00001,"lrf":0.1,"cos_lr":True,"warmup_epochs":1.0,"warmup_bias_lr":0.00001,"close_mosaic":2,"mosaic":0.25,"hsv_s":0.4,"hsv_v":0.2,"translate":0.05,"scale":0.2,"freeze":9},
    "transfer_tiny_lr_bn_frozen": {"epochs":6,"optimizer":"AdamW","lr0":0.00001,"lrf":0.1,"cos_lr":True,"warmup_epochs":1.0,"warmup_bias_lr":0.00001,"close_mosaic":2,"mosaic":0.25,"hsv_s":0.4,"hsv_v":0.2,"translate":0.05,"scale":0.2,"freeze":9},
}

def load_api():
    sys.path.insert(0,str(project_path("yolov13_src",must_exist=True)))
    from ultralytics import YOLO
    return YOLO

def evaluate(checkpoint, split="val", original80=False, run_name="validation",data_yaml="dataset/data.yaml",batch=8,half=False):
    from pipeline_utils import materialize_ultralytics_dataset_yaml
    YOLO=load_api()
    model=YOLO(str(project_path(checkpoint,must_exist=True)))
    from ultralytics.utils.torch_utils import get_flops
    # Count before validator may fuse Conv/BN in place, matching prior study.
    parameters_m=sum(p.numel() for p in model.model.parameters())/1e6
    flops_g=get_flops(model.model,imgsz=416)
    args=dict(data=str(materialize_ultralytics_dataset_yaml(data_yaml)),split=split,imgsz=416,batch=batch,half=half,device="0",workers=0,plots=False,verbose=False,project=str(project_path("results/optimization/eval")),name=run_name,exist_ok=True)
    if original80:
        # This repository's native validator ignores args.classes in NMS. Explicit
        # person filtering prevents remapping all80classes into person false positives.
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from ultralytics.utils import ops
        class PersonOnlyValidator(DetectionValidator):
            def postprocess(self,preds):
                return ops.non_max_suppression(preds,self.args.conf,self.args.iou,classes=[0],labels=self.lb,multi_label=True,agnostic=False,max_det=self.args.max_det)
        args.update(classes=[0],single_cls=False,validator=PersonOnlyValidator)
    metrics=model.val(**args)
    result=metric_summary(metrics)
    result.update(parameters_m=parameters_m,flops_g=flops_g)
    result["evaluation_protocol"]={"imgsz":416,"batch":batch,"half":half,"workers":0,"split":split,"data_yaml":str(project_path(data_yaml).relative_to(PROJECT_ROOT)).replace("\\","/")}
    if batch!=8:
        result["validator_speed_ms"]=result.pop("validator_speed_ms_batch8")
        result["validator_latency_ms_per_frame"]=result.pop("validator_latency_ms_per_frame_batch8")
    return result

def run(config,seed,stage,limit,run_tag=None):
    require_local_venv()
    from pipeline_utils import materialize_ultralytics_dataset_yaml
    import torch
    YOLO=load_api()
    name=f"{stage}-yolov13s-{config}-seed{seed}-{limit}epochs"
    if run_tag:name+=f"-{run_tag}"
    output=project_path(f"results/optimization/{stage}_records/{config}-seed{seed}.json")
    run_dir=project_path(f"runs/optimization/{name}")
    if output.exists() and json.loads(output.read_text(encoding="utf-8")).get("status")=="complete":return
    if run_dir.exists():raise RuntimeError(f"Preserving incomplete run; directory already exists: {run_dir}")
    log=project_path(f"results/optimization/logs/{name}.log"); log.parent.mkdir(parents=True,exist_ok=True)
    data=materialize_ultralytics_dataset_yaml("dataset/data.yaml")
    args=train_arguments(data,name,seed,project_path("runs/optimization"));args.update(CONFIGS[config]);args.update(patience=301)
    record={"status":"running","model_key":"yolov13s","configuration":config,"seed":seed,"stage":stage,"maximum_completed_epochs":limit,"started_at_utc":utc_timestamp(),"training_config":args,"dataset_manifest_sha256":sha256_file("dataset/fullscale_person/manifest.json"),"initial_weights_sha256":sha256_file("weights/yolov13s.pt"),"selection":"highest validation mAP50:95 among completed epochs; no test access"}
    if run_tag:record["recovery_attempt"]={"run_tag":run_tag,"evidence":"results/optimization/interruption_recovery_event.json","method":"Fresh rerun of identical frozen six-epoch configuration; interrupted partial trajectory excluded from candidate ranking."}
    write_json(output,record)
    with log.open("a",encoding="utf-8",buffering=1) as logfile,redirect_stdout(Tee(sys.stdout,logfile)),redirect_stderr(Tee(sys.stderr,logfile)):
        model=YOLO(str(project_path("weights/yolov13s.pt",must_exist=True)))
        def stop_after(trainer):
            if trainer.epoch+1>=limit:trainer.stop=True
        model.add_callback("on_fit_epoch_end",stop_after)
        trainer_class=None
        if config in("adamw_transfer_restored","transfer_tiny_lr","transfer_tiny_lr_bn_frozen"):
            from copy import deepcopy
            from ultralytics.models.yolo.detect.train import DetectionTrainer
            class TransferRestoredTrainer(DetectionTrainer):
                def get_model(self,cfg=None,weights=None,verbose=True):
                    if weights is None:raise RuntimeError("Transfer restoration requires initial pretrained model")
                    cfg=deepcopy(cfg)
                    actual_edges=weights.model[9].branch1.m.hgnn.edge_generator.num_hyperedges
                    if actual_edges!=8:raise RuntimeError(f"Unexpected pretrained hyperedge count: {actual_edges}")
                    for row in cfg["head"]:
                        if row[2]=="HyperACE":row[3][1]=actual_edges
                    restored=super().get_model(cfg=cfg,weights=weights,verbose=verbose)
                    source=weights.float().state_dict();destination=restored.state_dict()
                    mismatches=[k for k in source if k in destination and source[k].shape!=destination[k].shape]
                    if len(mismatches)!=6 or not all("cv3." in k for k in mismatches):raise RuntimeError(f"Unexpected tensor mismatches: {mismatches}")
                    with torch.no_grad():
                        for key in mismatches:
                            if source[key].shape[0]!=80 or destination[key].shape[0]!=1 or source[key].shape[1:]!=destination[key].shape[1:]:raise RuntimeError(f"Cannot copy person classifier row: {key}")
                            destination[key].copy_(source[key][:1])
                    record["initialization"]={"strategy":"restore YAML HyperACE edge count to actual8 ofpretrained model; native compatible tensor transfer; copy class0(person) row of80class final classifier weight+bias into oneclass head","actual_hyperedges_per_branch":actual_edges,"classifier_row_keys_copied":mismatches,"compatible_state_tensors":len(destination)-len(mismatches),"parameters_m":sum(p.numel() for p in restored.parameters())/1e6,"architecture_changed_vs_untuned":"HyperACE8 instead of16; nc1 retained; no modification to repository source"}
                    import yaml
                    cfg_output=project_path(f"results/optimization/configs/{config}-seed{seed}-{stage}-architecture.yaml");cfg_output.parent.mkdir(parents=True,exist_ok=True);cfg_output.write_text(yaml.safe_dump(restored.yaml,sort_keys=False),encoding="utf-8")
                    record["architecture_yaml"]=str(cfg_output.relative_to(PROJECT_ROOT)).replace("\\","/");record["architecture_yaml_sha256"]=sha256_file(cfg_output)
                    # Preserve a measured zero-finetuning nc1 control; validate a
                    # deep copy so Conv/BN fusion cannot modify training model.
                    initial_model=YOLO(str(project_path("weights/yolov13s.pt",must_exist=True)))
                    initial_model.model=deepcopy(restored)
                    initial_model.ckpt={"ema":None,"optimizer":None,"train_args":{"task":"detect","imgsz":416,"data":str(data)}}
                    initial_checkpoint=run_dir/"weights"/"initial-restored-person.pt"
                    initial_checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    initial_model.save(str(initial_checkpoint))
                    record["initialization"]["initial_checkpoint"]=str(initial_checkpoint.relative_to(PROJECT_ROOT)).replace("\\","/")
                    record["initialization"]["initial_checkpoint_sha256"]=sha256_file(initial_checkpoint)
                    # Same measured diagnostic only on seed42 screening; all
                    # confirmation runs use the same transfer without repeated
                    # extra validation before training.
                    if stage=="screening" and config=="adamw_transfer_restored":
                        import numpy as np
                        states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
                        try:record["initialization"]["zero_finetuning_validation"]=evaluate(initial_checkpoint,run_name="restored-person-zero-finetuning-validation")
                        finally:
                            random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2]);torch.cuda.set_rng_state_all(states[3])
                    del initial_model
                    write_json(output,record)
                    return restored
            trainer_class=TransferRestoredTrainer
            if config=="transfer_tiny_lr_bn_frozen":
                record["bn_statistics_policy"]={"callback":"on_train_batch_start; after trainer.model.train()","all_bn_running_statistics_frozen":True,"affine_parameters":"trainable outside freeze9 backbone","callback_invocations":0}
                def freeze_bn_statistics(trainer):
                    modules=[m for m in trainer.model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
                    for module in modules:module.eval()
                    record["bn_statistics_policy"]["callback_invocations"]+=1
                    record["bn_statistics_policy"]["batchnorm_modules"]=len(modules)
                model.add_callback("on_train_batch_start",freeze_bn_statistics)
        if config=="adamw_domain_loss_aug":
            from ultralytics.models.yolo.detect.train import DetectionTrainer
            from ultralytics.data.build import InfiniteDataLoader,seed_worker
            from ultralytics.data.utils import PIN_MEMORY
            from torch.utils.data import WeightedRandomSampler
            manifest=json.loads(project_path("dataset/fullscale_person/manifest.json").read_text(encoding="utf-8"))
            source_by_name={Path(x["image"]).name:x["source"] for x in manifest["split_manifest"]["train"]}
            class BalancedTrainer(DetectionTrainer):
                def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
                    if mode!="train":return super().get_dataloader(dataset_path,batch_size,rank,mode)
                    if rank not in(-1,0):raise ValueError("Only one GPU is supported")
                    dataset=self.build_dataset(dataset_path,mode,batch_size)
                    source=[source_by_name[Path(p).name] for p in dataset.im_files]
                    counts=Counter(source)
                    adv=[s for s in counts if "Adv" in s]
                    if len(adv)!=1:raise ValueError(f"Unexpected source map: {counts}")
                    adv_name=adv[0]
                    weights=[(.25/counts[s] if s==adv_name else .75/counts[s]) for s in source]
                    generator=torch.Generator().manual_seed(seed)
                    sampler=WeightedRandomSampler(weights,len(dataset),replacement=True,generator=generator)
                    record["sampling"]={"algorithm":"torch WeightedRandomSampler replacement=True","draws_per_epoch":len(dataset),"primary_image_expected_advtshirt_share":.25,"original_source_counts":dict(counts),"secondary_mosaic_images":"native uniformly drawn indices; not reweighted"}
                    write_json(output,record)
                    return InfiniteDataLoader(dataset,batch_size=min(batch_size,len(dataset)),shuffle=False,sampler=sampler,num_workers=0,pin_memory=PIN_MEMORY,collate_fn=dataset.collate_fn,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed+6148914691236517205))
            trainer_class=BalancedTrainer
        started=time.perf_counter()
        try:
            if trainer_class:model.train(trainer=trainer_class,**args)
            else:model.train(**args)
            wall=time.perf_counter()-started
            history=read_history(run_dir/"results.csv")
            eligible=[x for x in history if int(float(x["epoch"]))<=limit]
            chosen=max(eligible,key=lambda x:float(x["metrics/mAP50-95(B)"]))
            epoch=int(float(chosen["epoch"]))
            checkpoint=run_dir/"weights"/f"epoch{epoch-1}.pt"
            if not checkpoint.exists():raise FileNotFoundError(checkpoint)
            if config=="transfer_tiny_lr_bn_frozen":
                from diagnose_optimization_bn import state_from
                initial_state=state_from(record["initialization"]["initial_checkpoint"])
                selected_state=state_from(str(checkpoint))
                buffers=[key for key in initial_state if key.endswith(("running_mean","running_var","num_batches_tracked"))]
                record["bn_statistics_policy"]["selected_checkpoint_buffer_audit"]={"tensor_count":len(buffers),"exactly_equal_tensor_count":sum(torch.equal(initial_state[key],selected_state[key]) for key in buffers),"maximum_absolute_delta":max(float((initial_state[key].float()-selected_state[key].float()).abs().max()) for key in buffers),"comparison":"saved initial vs native-selected EMA checkpoint; both checkpoint tensor serialization may round FP16"}
            archive=project_path(f"results/optimization/epoch_history/{config}-seed{seed}-{stage}.csv");archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run_dir/"results.csv",archive)
            record.update(status="complete",finished_at_utc=utc_timestamp(),epochs_completed=len(history),actual_training_seconds=float(history[-1]["time"]),training_and_final_native_validation_wall_seconds=wall,selected_epoch=epoch,selected_elapsed_seconds=float(chosen["time"]),checkpoint=str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\","/"),checkpoint_sha256=sha256_file(checkpoint),results_csv=str(archive.relative_to(PROJECT_ROOT)).replace("\\","/"),results_csv_sha256=sha256_file(archive),log=str(log.relative_to(PROJECT_ROOT)).replace("\\","/"),validation={"precision":float(chosen["metrics/precision(B)"]),"recall":float(chosen["metrics/recall(B)"]),"map50":float(chosen["metrics/mAP50(B)"]),"map50_95":float(chosen["metrics/mAP50-95(B)"])},endpoint_validation_map50_95=float(history[-1]["metrics/mAP50-95(B)"]),endpoint_loss={k:float(history[-1][k]) for k in history[-1] if "loss" in k})
            write_json(output,record)
        except Exception as exc:
            record.update(status="failed",error=repr(exc),failed_at_utc=utc_timestamp());write_json(output,record);raise

def main():
    require_local_venv();p=argparse.ArgumentParser();p.add_argument("--config",choices=CONFIGS);p.add_argument("--seed",type=int,default=42);p.add_argument("--stage",default="screening");p.add_argument("--epochs",type=int,default=6);p.add_argument("--run-tag");p.add_argument("--pretrained-val",action="store_true");a=p.parse_args()
    if a.pretrained_val:
        started=time.perf_counter();value=evaluate("weights/yolov13s.pt",original80=True,run_name="pretrained-person-control");write_json("results/optimization/pretrained_validation.json",{"status":"complete","model_key":"yolov13s","weights":"weights/yolov13s.pt","weights_sha256":sha256_file("weights/yolov13s.pt"),"dataset_manifest_sha256":sha256_file("dataset/fullscale_person/manifest.json"),"validation":value,"mapping":"Custom PersonOnlyValidator filters COCO person class0 during NMS; retains original80classhead, single_cls=False; no test access","measured_at_utc":utc_timestamp(),"wall_seconds":time.perf_counter()-started})
    else:
        pause_flag=project_path("results/optimization/pause_for_protocol_amendment.flag")
        if pause_flag.exists():
            print("Intentional orchestration pause before next GPU job; preserve all completed training records.",flush=True)
            raise SystemExit(75)
        run(a.config,a.seed,a.stage,a.epochs,a.run_tag)
if __name__=="__main__":main()
