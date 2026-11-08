"""Phân tích E5-phone theo PREREG_E5_phone.md: A/A + MDE theo profile/công đoạn (cùng công thức E5 laptop),
CỘNG báo cáo trôi xung nhịp ở ĐẦU kết quả (số đợt bị cờ / tổng, độ lệch lớn nhất, độ dốc trôi) và độ nhạy "bỏ đợt bị cờ".

    python studies/e5_phone/analyze.py RUN_DIR PIN_VERIFY_JSON
RUN_DIR có raw_<profile>.jsonl (hàng burst_start/measure/burst_end), manifest.json. PIN_VERIFY_JSON là pin_verify.json của phần A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
sys.path.insert(0, str(SIM / 'studies' / 'e5_protocol'))
from analyze import BOOT_SEED, PRIMARY_STAGES, analyze_profile  # noqa: E402  (e5_protocol/analyze.py)
from sponge_sim import stats  # noqa: E402


def load_profile(run_dir: Path, profile: str, pin_verify: dict):
    raw = [json.loads(line) for line in (run_dir / f'raw_{profile}.jsonl').read_text(encoding='utf-8').splitlines()]
    v = pin_verify['verify'][profile]
    rows = [dict(kind='pin', profile=profile, allowed=v['allowed'], readback=v['allowed'], torch_threads=v['threads'], verify=v)]
    refs = []
    for r in raw:
        if r['kind'] == 'burst_start':
            refs.append(dict(kind='reference', block=r['burst'], ref_ms=r['ref_ms'], rel_change=r['rel_change'], flagged=r['flagged'],
                             wall_utc=r['wall_utc'], position='start'))
        elif r['kind'] == 'burst_end':
            refs.append(dict(kind='reference', block=f'end{r["burst"]}', ref_ms=r['ref_ms'], rel_change=r['rel_change'],
                             flagged=r['flagged'], wall_utc=r['wall_utc'], position='end', burst=r['burst']))
        elif r['kind'] == 'measure':
            rows.append(dict(r, block=r['burst']))
    return rows + refs, refs


def main():
    run_dir, pin_path = Path(sys.argv[1]), Path(sys.argv[2])
    pin_verify = json.loads(pin_path.read_text(encoding='utf-8'))
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    profiles = manifest['config']['profiles']
    out = dict(experiment='E5-phone', run_dir=str(run_dir), pin_verify=str(pin_path), bootstrap_seed=BOOT_SEED,
               n_boot=stats.N_BOOT, unit='image', config_sha256=manifest['config_sha256'],
               network_state_declared_by_user=manifest['config'].get('network_state_declared_by_user'), profiles={})
    headline = {}
    for prof in profiles:
        rows, refs = load_profile(run_dir, prof, pin_verify)
        res = analyze_profile(rows)
        starts = [r for r in refs if r.get('position') == 'start']
        ends = [r for r in refs if r.get('position') == 'end']
        allr = starts + ends
        flagged = [r for r in allr if r['flagged']]
        ref_series = [r['ref_ms'] for r in sorted(starts, key=lambda r: r['block'])]
        slope = None
        if len(ref_series) >= 3:
            x = np.arange(len(ref_series))
            slope = float(np.polyfit(x, ref_series, 1)[0] / ref_series[0])  # tương đối theo đợt
        res['drift_headline'] = dict(reference_points=len(allr), flagged=len(flagged), flagged_fraction=len(flagged) / max(len(allr), 1),
                                     max_abs_rel_change=max(abs(r['rel_change']) for r in allr),
                                     bursts_with_any_flag=len({r['block'] if r['position'] == 'start' else r.get('burst') for r in flagged}),
                                     total_bursts=len(starts), relative_slope_per_burst=slope)
        headline[prof] = res['drift_headline']
        out['profiles'][prof] = res
    out['drift_headline_first'] = headline          # đặt trước các kết quả khác theo PREREG
    out['primary'] = {p: {s: dict(ratio=v['stages'][s]['aa_ratio'], ci95=v['stages'][s]['aa_ci95_ratio'],
                                  ci_contains_1=v['stages'][s]['aa_ci_contains_1'], mde_pct=v['stages'][s]['mde_ratio_pct'])
                          for s in PRIMARY_STAGES} for p, v in out['profiles'].items()}
    ordered = {'experiment': out.pop('experiment'), 'drift_headline_first': out.pop('drift_headline_first'), **out}
    (run_dir / 'summary.json').write_text(json.dumps(ordered, indent=1, ensure_ascii=False), encoding='utf-8')
    for p, h in headline.items():
        print(p, 'cờ', h['flagged'], '/', h['reference_points'], 'max|Δ|', round(h['max_abs_rel_change'], 3),
              'MDE', {s: round(ordered['primary'][p][s]['mde_pct'], 2) for s in PRIMARY_STAGES})


if __name__ == '__main__':
    main()
