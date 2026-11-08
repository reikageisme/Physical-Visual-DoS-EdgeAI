"""Known synthetic cost surface and missing-repetition quality rejection."""
import importlib.util
import json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('m2_analysis',Path(__file__).parents[1]/'studies/m2_confirm/analyze.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_known_surface_and_missing_rep(tmp_path):
    images={f'{i}.jpg':str(i) for i in range(4)}
    cfg=dict(device='cpu',n_images=4,repeats=3,K=[0,10,50,100,300],retinas=[False,True])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=cfg,images=images)))
    rows=[dict(kind='reference',block='start',flagged=False)]
    for ix,image in enumerate(images):
        rows.append(dict(kind='reference',block=ix,flagged=False))
        for k in cfg['K']:
            for retina in cfg['retinas']:
                mask=k*(1 if retina else .05)
                for copy in range(2 if k==0 else 1):
                    for rep in range(3):
                        rows.append(dict(kind='cell',image=image,k=k,retina=retina,copy=copy,rep=rep,total_ns=100+mask,mask_ns=mask))
    rows += [dict(kind='reference',block='end',flagged=False),dict(kind='pin',verify=dict(passed=True)),dict(kind='pin_end',verify=dict(passed=True)),dict(kind='done')]
    path=tmp_path/'raw.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
    result=mod.analyze(tmp_path)
    assert result['quality_pass']
    np.testing.assert_allclose(result['ratios']['True|300']['ratio'],400/110,rtol=1e-14)
    assert result['mde']['True']['frac']==0
    del rows[2]
    path.write_text('\n'.join(map(json.dumps,rows)))
    assert not mod.analyze(tmp_path)['quality_pass']
