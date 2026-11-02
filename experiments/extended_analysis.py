"""Seed-level inference on completed extension; no pseudo-replication of repeats.

The primary ratio is averaged across the same held-out images inside a seed.
Clean denominators are floored at one, specified before evaluation; zero-clean
images are counted explicitly. All confidence intervals are conditional on this
fixed image subset and chosen model/hardware, not population-wide guarantees.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(values, seed=20261002, null=0.):
    values = np.asarray(values,dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise ValueError('Exactly ten finite seed-level observations required')
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    draws = np.random.default_rng(seed).integers(0,len(values),(20000,len(values)))
    bootstrap = values[draws].mean(1)
    ci = np.quantile(bootstrap,[.025,.975]).tolist()
    signs = np.asarray(list(itertools.product([-1.,1.],repeat=len(values))))
    permutation_means = (signs*(values-null)).mean(1)
    p = float(np.mean(np.abs(permutation_means) >= abs(mean-null)-1e-12))
    return dict(seed_values=values.tolist(),n_seeds=len(values),mean=mean,sample_std=sd,
                conditional_paired_seed_bootstrap_95ci=ci,bootstrap_draws=20000,
                exact_two_sided_sign_flip_p=p,sign_flip_assignments=len(signs),
                null_mean=null,paired_standardized_effect_dz=None if sd==0 else (mean-null)/sd,
                assumptions='paired seed effects exchangeable/sign-symmetric under null; fixed held-out images',
                not_an_image_population_confidence_interval=True)


def holm(rows):
    order = sorted(range(len(rows)),key=lambda i:rows[i]['statistics']['exact_two_sided_sign_flip_p'])
    previous = 0.
    for rank,index in enumerate(order):
        value = min(1.,(len(rows)-rank)*rows[index]['statistics']['exact_two_sided_sign_flip_p'])
        previous = max(previous,value)
        rows[index]['holm_p_within_declared_secondary_family'] = previous
    return rows


def observations(case,tau=.01,cap=None):
    return {r['image_id']:r for r in case['records'] if r['threshold']==tau and r['pre_nms_cap']==cap}


def ratio(case,clean,metric='active_candidates',ids=None):
    a,b = observations(case),observations(clean)
    keys = sorted(a.keys() & b.keys()) if ids is None else ids
    if not keys:
        raise ValueError('No common held-out image IDs')
    return float(np.mean([a[i][metric]/max(1.,b[i][metric]) for i in keys]))


def baseline_metric(case,tau=.01,cap=None):
    return next(m for m in case['metrics'] if m['threshold']==tau and m['pre_nms_cap']==cap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    run = ROOT/args.run_dir
    evaluation = run/'evaluation'
    if 'finished_utc' not in load(evaluation/'manifest.json'):
        raise RuntimeError('Analysis needs all completed evaluation cases')
    optimizer = load(run/'manifest.json')
    if optimizer['protocol']['seeds'] != 10:
        raise ValueError('This analysis protocol requires ten seeds')
    sources = {}

    def read_case(name,seed,budget):
        path = evaluation/f'{name}_seed{seed}_q{budget}.json'
        result = load(path)
        sources[str(path.relative_to(ROOT))] = digest(path)
        return result

    clean = [read_case('clean',seed,0) for seed in range(10)]
    all_specs = optimizer['protocol']['specifications']
    results = {}
    for spec in all_specs:
        for budget in spec['budgets']:
            key = (spec['name'],budget)
            results[key] = [read_case(spec['name'],seed,budget) for seed in range(10)]
    sg = results[('sg_ga',2500)]
    center = results[('ga_center',2500)]
    primary_values = [ratio(sg[s],clean[s])-ratio(center[s],clean[s]) for s in range(10)]
    primary = paired(primary_values)
    primary.update(hypothesis=optimizer['protocol']['primary_hypothesis'],threshold=.01,pre_nms_cap=None,
                   minimum_relevant_difference=.1,
                   zero_clean_candidate_denominators=sum(r['active_candidates']==0 for c in clean for r in observations(c).values()),
                   ratio_definition='mean over common images of patched candidates / max(clean candidates, 1)',
                   meaningful_sg_superiority_supported=primary['conditional_paired_seed_bootstrap_95ci'][0]>.1,
                   does_not_establish_availability_failure=True)
    methods = []
    secondary = []
    for (name,budget),cases in results.items():
        values = [ratio(cases[s],clean[s]) for s in range(10)]
        ap = [baseline_metric(c)['coco_subset_bbox']['AP'] for c in cases]
        methods.append(dict(name=name,budget=budget,images_per_seed=cases[0]['images'],
                            candidate_ratio=paired(values,null=1.),
                            candidate_ratio_minus_clean=paired(np.asarray(values)-1),
                            AP_mean_over_patch_seeds=float(np.mean(ap)),AP_sample_std_over_patch_seeds=float(np.std(ap,ddof=1)),
                            AP_is_pooled_over_images_before_averaging_patch_seeds=True))
        if budget == max(next(s['budgets'] for s in all_specs if s['name']==name)) and name!='sg_ga':
            reference = results[('sg_ga',budget)]
            common = sorted(set(cases[0]['image_ids']) & set(reference[0]['image_ids']))
            effect = [ratio(reference[s],clean[s],ids=common)-ratio(cases[s],clean[s],ids=common) for s in range(10)]
            secondary.append(dict(comparison=f'sg_ga_q{budget}_minus_{name}_q{budget}',
                                  equal_budget=True,images=len(common),statistics=paired(effect)))
    holm(secondary)
    curves = []
    ids32 = sorted(clean[0]['image_ids'])[:32]
    for name in ['sg_ga','ga_center','ga_random_location','random_search']:
        for budget in [400,1000,2500]:
            values = [ratio(results[(name,budget)][s],clean[s],ids=ids32) for s in range(10)]
            curves.append(dict(name=name,budget=budget,common_image_ids=ids32,statistics=paired(values,null=1.)))
    defenses = []
    for name,budget in [('clean',0),('sg_ga',400),('sg_ga',2500),('adaptive_cap100',400)]:
        cases = clean if name=='clean' else results[(name,budget)]
        for tau in [.01,.05,.10,.25,.50]:
            for cap in [None,50,100,300]:
                delta_ap = [baseline_metric(c,tau,cap)['coco_subset_bbox']['AP']-baseline_metric(c)['coco_subset_bbox']['AP'] for c in cases]
                delta_ar = [baseline_metric(c,tau,cap)['coco_subset_bbox']['AR100']-baseline_metric(c)['coco_subset_bbox']['AR100'] for c in cases]
                latency = [baseline_metric(c,tau,cap)['postprocess_ms']['mean'] for c in cases]
                defenses.append(dict(name=name,budget=budget,threshold=tau,pre_nms_cap=cap,
                                     delta_AP_vs_same_patch_undefended=paired(delta_ap),
                                     delta_AR100_vs_same_patch_undefended=paired(delta_ar),
                                     postprocess_mean_over_seed_ms=float(np.mean(latency)),
                                     inference_is_exploratory_multiple_grid_comparisons=True))
    eot = []
    eot_specs = all_specs+[dict(name='sg_ga',budgets=[400])]
    for spec in eot_specs:
        budget = max(spec['budgets'])
        effects = []
        for seed in range(10):
            path = evaluation/f'eot_{spec["name"]}_seed{seed}_q{budget}.json'
            row = load(path)
            sources[str(path.relative_to(ROOT))] = digest(path)
            effects.append(float(np.mean([np.mean([s['difference'] for s in r['samples']]) for r in row['records']])))
        eot.append(dict(name=spec['name'],budget=budget,images_per_seed=8,new_paired_draws_per_image=16,
                        independent_fitness_difference=paired(effects)))
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),analysis_source_sha256=digest(__file__),
                  optimizer_manifest_sha256=digest(run/'manifest.json'),
                  evaluator_manifest_sha256=digest(evaluation/'manifest.json'),source_files_sha256=sources,
                  primary=primary,methods=methods,secondary_holm_family=secondary,
                  same_image_budget_curves=curves,defense_tradeoffs=defenses,independent_eot=eot,
                  limits=['fixed 128-image subset, not full COCO', 'confidence intervals conditional on images/model/hardware',
                          'ten seeds resource-limited, no prospective power guarantee',
                          'no physical/camera evidence in these results','adapted baselines not exact published reproductions'],
                  preregistration='local protocol before held-out evaluation, not public registry')
    # The additional suites are separate protocols; keep their units, timing
    # boundaries and image counts separate rather than pool all observations.
    controls = []
    control_folder = run/'controls'
    if (control_folder/'manifest.json').exists() and 'finished_utc' in load(control_folder/'manifest.json'):
        names = load(control_folder/'manifest.json')['protocol']['controls']
        for name in names:
            cases = [load(control_folder/f'{name}_seed{s}'/'evaluation.json') for s in range(10)]
            values = [ratio(cases[s],clean[s])-1 for s in range(10)]
            ap = [baseline_metric(cases[s])['coco_subset_bbox']['AP']-baseline_metric(clean[s])['coco_subset_bbox']['AP'] for s in range(10)]
            fixed = name in ['checkerboard','solid_gray']
            controls.append(dict(name=name,
                                  candidate_ratio_minus_clean=dict(mean=float(np.mean(values)),inference_not_applicable_fixed_patch=True) if fixed else paired(values),
                                  delta_AP_vs_clean=dict(mean=float(np.mean(ap)),inference_not_applicable_fixed_patch=True) if fixed else paired(ap),
                                  independent_patch_realizations=1 if fixed else 10,
                                  optimization_queries=400 if name=='integrity_gradient' else 0,
                                  not_published_attack_reproduction=True))
            for s in range(10):
                path = control_folder/f'{name}_seed{s}'/'evaluation.json'
                sources[str(path.relative_to(ROOT))] = digest(path)
        output['controls_manifest_sha256'] = digest(control_folder/'manifest.json')
    output['additional_controls_exploratory'] = controls
    transfer = []
    transfer_folder = run/'transfer'
    if (transfer_folder/'manifest.json').exists() and 'finished_utc' in load(transfer_folder/'manifest.json'):
        for name in load(transfer_folder/'manifest.json')['protocol']['models']:
            values,ap = [],[]
            for s in range(10):
                path = transfer_folder/f'{name}_seed{s}.json'
                case = load(path)
                sources[str(path.relative_to(ROOT))] = digest(path)
                a = {r['image_id']:r for r in case['records'] if r['condition']=='patched'}
                b = {r['image_id']:r for r in case['records'] if r['condition']=='clean'}
                values.append(float(np.mean([a[i]['total_prepared_tensor_ms']/b[i]['total_prepared_tensor_ms'] for i in a])))
                ap.append(case['coco_subset_bbox']['patched']['AP']-case['coco_subset_bbox']['clean']['AP'])
            transfer.append(dict(model=name,images=16,within_model_paired_latency_ratio=paired(values,null=1.),
                                  subset_delta_AP=paired(ap),has_nms=name!='detr',
                                  interpretation='same prepared input; native model normalization/postprocessing; not cross-model absolute latency'))
        output['transfer_manifest_sha256'] = digest(transfer_folder/'manifest.json')
    output['transfer_exploratory'] = transfer
    qos = []
    qos_folder = run/'backend_qos'
    if (qos_folder/'manifest.json').exists() and 'finished_utc' in load(qos_folder/'manifest.json'):
        for threads in [1,2,4]:
            blocks = [load(qos_folder/f'timing_seed{s}_threads{threads}.json') for s in range(10)]
            for name in ['clean','sg_ga','ga_center','ga_random_location','random_search']:
                rates,medians,tails = [],[],[]
                for s,block in enumerate(blocks):
                    rows = [r for r in block['records'] if r['condition']==name]
                    values = [r['total_prepared_tensor_ms'] for r in rows]
                    rates.append(float(np.mean([r['late_processing_budget'] for r in rows])))
                    medians.append(float(np.median(values)))
                    tails.append(float(np.quantile(values,.95)))
                    sources[str((qos_folder/f'timing_seed{s}_threads{threads}.json').relative_to(ROOT))] = digest(qos_folder/f'timing_seed{s}_threads{threads}.json')
                late_estimate = paired(rates)
                for key in ['exact_two_sided_sign_flip_p','paired_standardized_effect_dz','sign_flip_assignments','null_mean','assumptions']:
                    late_estimate.pop(key)
                qos.append(dict(condition=name,cpu_threads=threads,physical_devices=1,
                                processing_deadline_ms=300,late_fraction=late_estimate,
                                median_ms_mean_over_seeds=float(np.mean(medians)),p95_ms_mean_over_seeds=float(np.mean(tails)),
                                not_camera_deadline_measurement=True))
        output['backend_qos_manifest_sha256'] = digest(qos_folder/'manifest.json')
    output['processing_budget_exploratory'] = qos
    reference = []
    folder = run/'reference'
    if (folder/'manifest.json').exists() and 'finished_utc' in load(folder/'manifest.json'):
        for name in load(folder/'manifest.json')['protocol']['functions']:
            values,ap,forward_counts,norms = [],[],[],[]
            for s in range(10):
                path = folder/f'{name}_seed{s}.json'
                case = load(path)
                sources[str(path.relative_to(ROOT))] = digest(path)
                a = {r['image_id']:r for r in case['records'] if r['condition']=='patched'}
                b = {r['image_id']:r for r in case['records'] if r['condition']=='clean'}
                values.append(float(np.mean([np.median([v['active_candidates'] for v in a[i]['samples']])/max(1,np.median([v['active_candidates'] for v in b[i]['samples']])) for i in a])))
                ap.append(case['coco_subset_bbox']['patched']['AP']-case['coco_subset_bbox']['clean']['AP'])
                forward_counts.append(case['training_image_forwards'])
                norms.append(case['actual_final_l2_norm'])
            reference.append(dict(method=name,images=32,within_reference_candidate_ratio=paired(values,null=1.),
                                   within_reference_delta_AP=paired(ap),training_image_forwards=forward_counts,actual_norms=norms,
                                   reference_implementation_variant_not_original_paper=True,
                                   not_equal_threat_model_budget_training_or_architecture_to_local_patch=True))
        output['reference_manifest_sha256'] = digest(folder/'manifest.json')
    output['public_reference_variants_exploratory'] = reference
    target = run/'analysis.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ANALYSIS COMPLETE: '+str(target),flush=True)


if __name__ == '__main__':
    main()
