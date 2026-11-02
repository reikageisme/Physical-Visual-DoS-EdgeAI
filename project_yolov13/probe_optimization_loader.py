"""CPU-only native augmentation throughput probe for workers0 vs2."""
from pipeline_utils import configure_project_environment,require_local_venv,project_path,write_json,utc_timestamp
configure_project_environment()
def main():
    require_local_venv()
    import sys,time,gc,torch,yaml
    torch.set_num_threads(2)
    sys.path.insert(0,str(project_path("yolov13_src",must_exist=True)))
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset,build_dataloader
    from fullscale_train_one import train_arguments
    from pipeline_utils import materialize_ultralytics_dataset_yaml
    runtime=materialize_ultralytics_dataset_yaml("dataset/data.yaml")
    data=yaml.safe_load(runtime.read_text(encoding="utf-8"));results=[]
    for workers in(0,2):
        args=train_arguments(runtime,"loader-probe",42,project_path("runs/optimization"));args.update(workers=workers)
        cfg=get_cfg(overrides=args)
        dataset=build_yolo_dataset(cfg,str(project_path("dataset/fullscale_person/images/train")),8,data,mode="train",rect=False,stride=32)
        started=time.perf_counter();loader=build_dataloader(dataset,8,workers,shuffle=False,rank=-1)
        batch_times=[];prior=started
        for index,batch in enumerate(loader):
            now=time.perf_counter();batch_times.append(now-prior);prior=now
            if index+1>=128:break
        elapsed=time.perf_counter()-started
        results.append({"workers":workers,"batches":len(batch_times),"images":len(batch_times)*8,"wall_seconds_including_worker_start":elapsed,"steady_batches_per_second_excluding_first16":112/sum(batch_times[16:]),"gpu_used":False})
        del loader,dataset;gc.collect()
    write_json("results/optimization/loader_throughput_probe.json",{"status":"complete","measured_at_utc":utc_timestamp(),"scope":"CPU-only native416px batch8 trainingaugmentation loader; same sorted primaryimages, no modelforward/backward; measured while existing GPU training active; diagnostic, not isolated GPU benchmark","records":results})
if __name__=="__main__":main()
