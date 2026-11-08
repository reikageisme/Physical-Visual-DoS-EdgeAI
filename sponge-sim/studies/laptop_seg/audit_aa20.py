"""Post-run verification/figure only; no inference, load, or new timings."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO/'sponge-sim/results/laptop_seg_20261107/aa20_01'
RUN = ROOT/'P1_cpu2'
TAG = 'prereg-laptop-aa20-20261107'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = RUN/'raw.jsonl'
    rows = [json.loads(s) for s in raw.read_text().splitlines()]
    m = json.loads((RUN/'manifest.json').read_text())
    saved = json.loads((RUN/'controls_summary.json').read_text())
    lockpath = REPO/'sponge-sim/studies/laptop_seg/lock_aa20.json'
    lock = json.loads(lockpath.read_text())
    assert sha(lockpath) == m['lock_sha256']
    assert rows[-1]['kind'] == 'done' and saved['complete']
    assert saved['raw_sha256'] == sha(raw)
    tagcommit = subprocess.check_output(['git', 'rev-parse', TAG], text=True).strip()
    assert tagcommit == m['prereg_commit']
    for name, expected in lock['source_sha256'].items():
        assert sha(REPO/name) == expected
    primary = ['sponge-sim/studies/laptop_seg/run.py', 'sponge-sim/studies/laptop_seg/analyze.py',
               'sponge-sim/prereg/PREREG_LAPTOP_AA20_20261107.md']
    for p in primary:
        b = subprocess.check_output(['git','show',f'{TAG}:{p}'])
        assert hashlib.sha256(b).hexdigest() == lock['source_sha256'][p]
    assert subprocess.check_output(['git','show',f'{TAG}:sponge-sim/studies/laptop_seg/lock_aa20.json']) == lockpath.read_bytes()
    for r in lock['files']:
        assert sha(REPO/r['path']) == r['sha256']
    refs = []
    for line, r in enumerate(rows, 1):
        if r['kind'] != 'reference':
            continue
        med = statistics.median(r['samples_ns'])
        baseline = med if not refs else refs[0]['median_ns']
        ratio = med/baseline
        assert ratio == r['ratio'] and (abs(ratio-1) > .03) == r['flagged']
        refs.append(dict(raw_line=line, block=r['block'], median_ns=med, ratio=ratio,
                         flagged=r['flagged'], elapsed_s=r['wall_utc']-rows[0]['wall_utc']))
    assert len(refs) == 22 and sum(r['flagged'] for r in refs) == 3
    ids = [Path(p).stem for p in lock['input_paths']]
    assert sorted({r['image'] for r in rows if r['kind']=='timing'}) == ids
    stages = {}
    for stage in ('forward', 'total'):
        medians = {}
        for image in ids:
            for label in ('A', 'B'):
                rs = [r for r in rows if r['kind']=='timing' and r['stage']==stage and r['image']==image and r['label']==label]
                assert sorted(r['rep'] for r in rs) == list(range(8))
                medians[image,label] = statistics.median(r['ns'] for r in rs)
        d = np.array([np.log(medians[i,'A']/medians[i,'B']) for i in ids])
        mde = float(np.expm1((stats.t.ppf(.975,19)+stats.t.ppf(.8,19))*d.std(ddof=1)/np.sqrt(20)))
        assert mde == saved['stages'][stage]['aa_mde_fraction']
        assert float(np.exp(d.mean())) == saved['stages'][stage]['aa_geometric_ratio']
        times = [r['ns'] for r in rows if r['kind']=='timing' and r['stage']==stage]
        stages[stage] = dict(aa_mde_fraction=mde, aa_ratio=float(np.exp(d.mean())),
                             per_image_aa_ratio=np.exp(d).tolist(),
                             all_trial_median_ms=statistics.median(times)/1e6,
                             max_trial_ms=max(times)/1e6)
    assert saved['quality_pass'] is False
    result = dict(status='POSTHOC_OFFLINE_VERIFIED_NO_NEW_TIMINGS', prereg_tag=TAG, prereg_commit=tagcommit,
                  lock_sha256=sha(lockpath), raw_sha256=sha(raw),
                  primary_tagged_source_bytes_match=True, all_runtime_source_and_input_hashes_match=True,
                  raw_rows=len(rows), timing_rows=sum(r['kind']=='timing' for r in rows),
                  reference_samples=sum(len(r['samples_ns']) for r in rows if r['kind']=='reference'),
                  references=refs, stages=stages, quality_pass=False,
                  source_script_sha256=sha(Path(__file__)),
                  artifact_hashes={f.name:sha(f) for f in [raw,RUN/'manifest.json',RUN/'controls_summary.json']})
    with (ROOT/'verification_receipt.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(8,6.4),layout='constrained')
    x = np.array([r['elapsed_s'] for r in refs])/60
    y = 100*(np.array([r['ratio'] for r in refs])-1)
    flags = np.array([r['flagged'] for r in refs])
    axes[0].plot(x,y,'o-',color='#2463a6',markersize=4)
    axes[0].scatter(x[flags],y[flags],color='#c53030',zorder=3,label='3 flagged references / 22')
    for v in (-3,3):
        axes[0].axhline(v,color='gray',ls='--',lw=1)
    axes[0].set(xlabel='Minutes since first raw record',ylabel='Reference change from start (%)',
                title='One P-core clean A/A20: quality gate FAIL')
    axes[0].legend(loc='lower right',fontsize=8)
    for name,color in [('forward','#2463a6'),('total','#c05621')]:
        axes[1].plot(range(1,21),100*(np.array(stages[name]['per_image_aa_ratio'])-1),
                     'o-',label=f'{name}: MDE {100*stages[name]["aa_mde_fraction"]:.2f}%',color=color,markersize=3)
    axes[1].axhline(0,color='gray',lw=1)
    axes[1].set(xlabel='Image index in locked order',ylabel='Median A / median B minus 1 (%)',
                title='Identical clean inputs; A/B are timing labels')
    axes[1].set_xticks(range(1,21,2))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ('png','pdf'):
        p=ROOT/f'aa20_diagnostic.{ext}'
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p,dpi=170)
    plt.close(fig)
    print(json.dumps(dict(verified=True,stages=stages,flags=3,quality_pass=False),indent=2))


if __name__ == '__main__':
    main()
