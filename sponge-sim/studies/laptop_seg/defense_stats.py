"""Prespecified image-unit exact sign flips and Holm correction, offline only."""
import numpy as np


def exact_signflip(values):
    values=np.asarray(values,dtype=float)
    n=len(values)
    if not 1<=n<=20 or not np.isfinite(values).all():
        raise ValueError('requires 1..20 finite image log ratios')
    observed=abs(values.sum());extreme=0
    for start in range(0,2**n,65536):
        indices=np.arange(start,min(start+65536,2**n),dtype=np.uint32)[:,None]
        signs=1-2*((indices>>np.arange(n,dtype=np.uint32))&1).astype(np.int8)
        extreme+=int((np.abs(signs@values)>=observed-1e-12).sum())
    return extreme/2**n


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('invalid p-values')
    order=np.argsort(p);adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted.tolist()
