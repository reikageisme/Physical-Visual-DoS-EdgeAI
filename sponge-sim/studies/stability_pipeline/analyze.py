"""All-three-session stability analysis; no selection of a preferred session."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def pool_logs(matrix):
    matrix=np.asarray(matrix,float)
    assert matrix.ndim==2 and matrix.shape[0]==3
    per_image=np.median(matrix,axis=0)
    rng=np.random.default_rng(20261106)
    draws=rng.integers(0,len(per_image),(20000,len(per_image)))
    ci=np.exp(np.quantile(per_image[draws].mean(1),[.025,.975]))
    return dict(n_sessions=3,n_images=len(per_image),geometric_ratio=float(np.exp(per_image.mean())),
                ci95=ci.tolist(),per_image_median_log_ratios=per_image.tolist())


def analyze(root):
    source=Path(__file__).parents[1]/'m4_nms_dispatch/analyze.py'
    spec=importlib.util.spec_from_file_location('paired_analysis',source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    sessions=[];matrix=[];identities=None;lock=None
    for index in (1,2,3):
        run=root/f'session{index}'
        if not (run/'raw.jsonl').exists():
            sessions.append(dict(session=index,complete=False,quality_pass=False,reason='missing raw'))
            continue
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['config']['experiment']=='M6' and manifest['config']['session']==index
        images=manifest['replay_inputs']
        if identities is None:identities=images;lock=manifest['provenance']['lock_sha256']
        assert identities==images and lock==manifest['provenance']['lock_sha256']
        rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
        if rows[-1]['kind']!='done':
            sessions.append(dict(session=index,complete=False,quality_pass=False,reason='missing done marker'))
            continue
        summary=old.analyze(run)
        summary['session']=index
        summary['quality_pass']=(summary['quality_pass']
            and len([r for r in rows if r['kind']=='reference'])==22
            and all(summary['mde'][arm]['mde_frac'] is not None and summary['mde'][arm]['mde_frac']<=.01
                    for arm in ('clean','attacked')))
        sessions.append(summary)
        if not summary['complete']:
            continue
        clean=summary['stages']['clean|total_tv_True']['image_medians_ms']
        attacked=summary['stages']['attacked|total_tv_True']['image_medians_ms']
        matrix.append(np.log(np.asarray(attacked)/np.asarray(clean)))
    return dict(n_sessions_planned=3,stable=all(s['quality_pass'] for s in sessions),
                all_sessions=sessions,aggregate=None if len(matrix)!=3 else pool_logs(matrix),
                note='Aggregate includes all three sessions, including any failing quality; use as primary only if stable=true.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();out=analyze(a.root)
    with (a.root/'confirmation_summary.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(dict(stable=out['stable'],aggregate=out['aggregate']),indent=2))
