"""CPU-only audit of original pretrained -> one-class tensor transfer."""
from pipeline_utils import configure_project_environment,require_local_venv,project_path,write_json,utc_timestamp
configure_project_environment();require_local_venv()
import sys,torch
torch.set_num_threads(2)
sys.path.insert(0,str(project_path("yolov13_src",must_exist=True)))
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from copy import deepcopy
original=YOLO(str(project_path("weights/yolov13s.pt",must_exist=True))).model.cpu().float()
converted=DetectionModel(original.yaml,nc=1,verbose=False).cpu().float()
source=original.state_dict();destination=converted.state_dict()
matched=[k for k in source if k in destination and source[k].shape==destination[k].shape]
mismatches=[{"key":k,"pretrained_shape":list(source[k].shape),"one_class_shape":list(destination[k].shape)} for k in source if k in destination and source[k].shape!=destination[k].shape]
corrected_cfg=deepcopy(original.yaml)
hyper_rows=[r for r in corrected_cfg["head"] if r[2]=="HyperACE"]
checkpoint_yaml_edges=[r[3][1] for r in hyper_rows]
actual_edges=original.model[9].branch1.m.hgnn.edge_generator.num_hyperedges
for row in hyper_rows:row[3][1]=actual_edges
restored=DetectionModel(corrected_cfg,nc=1,verbose=False).cpu().float();restored_state=restored.state_dict()
restored_mismatches=[{"key":k,"pretrained_shape":list(source[k].shape),"one_class_shape":list(restored_state[k].shape)} for k in source if k in restored_state and source[k].shape!=restored_state[k].shape]
write_json("results/optimization/head_transfer_audit.json",{"status":"complete","audited_at_utc":utc_timestamp(),"device":"CPU only","pretrained_nc":original.model[-1].nc,"finetuned_nc":converted.model[-1].nc,"checkpoint_yaml_hyperedges":checkpoint_yaml_edges,"actual_pretrained_hyperedges":actual_edges,"native_rebuilt_hyperedges":converted.model[9].branch1.m.hgnn.edge_generator.num_hyperedges,"compatible_state_tensor_count":len(matched),"incompatible_state_tensors":mismatches,"native_transfer":"DetectionModel.load copies only keys with identical shapes; incompatible classifier and HyperACE tensors retain newly initialized values","restored_architecture_incompatible_tensors":restored_mismatches,"person_row_transplant_after_restoring_hyperedges_feasible":all(m["pretrained_shape"][0]==80 and m["one_class_shape"][0]==1 and m["pretrained_shape"][1:]==m["one_class_shape"][1:] for m in restored_mismatches),"pretrained_parameters_m":sum(p.numel() for p in original.parameters())/1e6,"native_oneclass_parameters_m":sum(p.numel() for p in converted.parameters())/1e6,"restored_oneclass_parameters_m":sum(p.numel() for p in restored.parameters())/1e6,"audit_scope":"Tensor shapes and source implementation; no new GPU evaluation or training","matched_parameters_numel":sum(source[k].numel() for k in matched),"unmatched_new_tensors_numel":sum(destination[m["key"]].numel() for m in mismatches)})
original_n=YOLO(str(project_path("weights/yolov13n.pt",must_exist=True))).model.cpu().float()
cfg_n=deepcopy(original_n.yaml);native_n=DetectionModel(deepcopy(cfg_n),nc=1,verbose=False).cpu().float()
actual_n=original_n.model[9].branch1.m.hgnn.edge_generator.num_hyperedges
encoded_n=[r[3][1] for r in cfg_n["head"] if r[2]=="HyperACE"]
for row in cfg_n["head"]:
    if row[2]=="HyperACE":row[3][1]=actual_n*2 # parse_model applies0.5 hyperedge multiplier to n.
restored_n=DetectionModel(cfg_n,nc=1,verbose=False).cpu().float()
src_n=original_n.state_dict();nat_n=native_n.state_dict();res_n=restored_n.state_dict()
bad_n=[{"key":k,"pretrained_shape":list(src_n[k].shape),"one_class_shape":list(nat_n[k].shape)} for k in src_n if k in nat_n and src_n[k].shape!=nat_n[k].shape]
restored_bad_n=[{"key":k,"pretrained_shape":list(src_n[k].shape),"one_class_shape":list(res_n[k].shape)} for k in src_n if k in res_n and src_n[k].shape!=res_n[k].shape]
write_json("results/optimization/head_transfer_audit_yolov13n.json",{"status":"complete","audited_at_utc":utc_timestamp(),"device":"CPU only","model_key":"yolov13n","checkpoint_yaml_hyperedges":encoded_n,"actual_pretrained_hyperedges":actual_n,"native_rebuilt_hyperedges":native_n.model[9].branch1.m.hgnn.edge_generator.num_hyperedges,"restored_hyperedges":restored_n.model[9].branch1.m.hgnn.edge_generator.num_hyperedges,"native_incompatible_state_tensors":bad_n,"restored_incompatible_state_tensors":restored_bad_n,"person_row_transplant_after_restoring_hyperedges_feasible":all(m["pretrained_shape"][0]==80 and m["one_class_shape"][0]==1 and m["pretrained_shape"][1:]==m["one_class_shape"][1:] for m in restored_bad_n),"native_oneclass_parameters_m":sum(p.numel() for p in native_n.parameters())/1e6,"restored_oneclass_parameters_m":sum(p.numel() for p in restored_n.parameters())/1e6,"scope":"CPU tensor-shape audit only; no nano tuning or performance claim"})
