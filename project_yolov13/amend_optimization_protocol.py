"""Record bounded adaptive decisions before new candidate or test access."""
import argparse,json
from pipeline_utils import configure_project_environment,require_local_venv,project_path,sha256_file,utc_timestamp,write_json
configure_project_environment()

def main():
    require_local_venv();parser=argparse.ArgumentParser();parser.add_argument("--freeze-deployment",action="store_true");args=parser.parse_args()
    path=project_path("results/optimization/protocol.json",must_exist=True);protocol=json.loads(path.read_text(encoding="utf-8"))
    if project_path("results/optimization/winner_lock.json").exists():raise RuntimeError("Protocol cannot amend after selection lock")
    if args.freeze_deployment:
        if "deployment_choice_policy" not in protocol:
            protocol["deployment_choice_policy"]={"frozen_at_utc":utc_timestamp(),"criterion":"Highest seed42 full-validation FP32batch8 mAP95 between trained winner's native-selected checkpoint and restorednc1 zero-finetuning checkpoint","tie_rule":"Prefer zero-finetuning on exact equality to avoid extra training cost","test_access":False,"confirmation":"Still3 fresh independent-seed trainedwinner confirmations; zero deterministic checkpoint n1, no seed SD/CI","evaluation":"After validation decision lock, chosen zero checkpoint test and sameFP32batch1 timing if zero wins; original80head person diagnostic also retained"}
            protocol["selection_criterion"]="Stage1 choose CSVnative FP16batch16 maximum mAP95 epoch1..6 per trajectory; stage2 rank ONLYnative-selected checkpoints by FP32batch8 fullval mAP95, seed42; original_sgd excluded from trainedoptimizedpool"
            write_json(path,protocol)
        return
    if "adaptive_sixth_candidate_decision" in protocol:return
    diagnostic_path=project_path("results/optimization/bn_controlled_diagnostic.json",must_exist=True)
    diagnostic=json.loads(diagnostic_path.read_text(encoding="utf-8"))
    record=json.loads(project_path("results/optimization/screening_records/adamw_transfer_restored-seed42.json",must_exist=True).read_text(encoding="utf-8"))
    zero=record["initialization"]["zero_finetuning_validation"]["map50_95"]
    trained=diagnostic["unchanged_fp32_batch8_validation"]["map50_95"]
    activated=zero-trained>.005;freeze_bn=diagnostic["map95_gain_from_only_bn_buffer_replacement"]>=.005
    decision={"decided_at_utc":utc_timestamp(),"diagnostic":str(diagnostic_path.relative_to(project_path("."))).replace("\\","/"),"diagnostic_sha256":sha256_file(diagnostic_path),"zero_fp32_map95":zero,"trained_fp32_map95":trained,"activation_gap":zero-trained,"activated":activated,"bn_buffer_reset_fp32_gain":diagnostic["map95_gain_from_only_bn_buffer_replacement"],"freeze_bn_for_extra_candidate":freeze_bn if activated else None,"test_seen":False}
    if activated:
        from optimization_train_one import CONFIGS
        name="transfer_tiny_lr_bn_frozen" if freeze_bn else "transfer_tiny_lr"
        configs=protocol["candidate_configurations"];configs={key:value for key,value in configs.items()}
        # Execute the one permitted adaptive candidate next, then all remaining original candidates.
        ordered={}
        for key,value in configs.items():
            ordered[key]=value
            if key=="adamw_transfer_restored":ordered[name]=CONFIGS[name]
        if len(ordered)!=6:raise RuntimeError("Frozen bound requires exactly6 including control")
        protocol["candidate_configurations"]=ordered;protocol["maximum_training_runs"]=9;protocol["total_six_epoch_comparison_records"]=12
        decision["candidate_added"]=name;decision["bundled_intervention"]="faithful8edge personrow transfer + AdamW1e-5 + frozen9layerbackbone + mildaugmentation + conditionallyfrozen BNstatistics; not isolated causal ablation"
    protocol["adaptive_sixth_candidate_decision"]=decision;write_json("results/optimization/adaptive_sixth_candidate_decision.json",decision);write_json(path,protocol)
if __name__=="__main__":main()
