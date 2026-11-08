"""Frozen whole-image PGD replay jobs; no optimization or input selection.

Balanced order follows studies/stability_pipeline/run.py (bound in new lock).
"""
import hashlib
import random


def balanced_orders(jobs,repeats,seed):
    groups={}
    for j,job in enumerate(jobs):groups.setdefault(job[1],[]).append(j)
    assert all(len(v)==2 for v in groups.values())
    stages=list(groups);random.Random(seed).shuffle(stages)
    orders=[]
    for rep in range(repeats):
        offsets=list(range(len(stages)));random.Random(seed+rep//len(stages)).shuffle(offsets)
        offset=offsets[rep%len(stages)];order=[]
        for stage in stages[offset:]+stages[:offset]:
            pair=groups[stage][:]
            if (rep+stages.index(stage))%2:pair.reverse()
            order.extend(pair)
        orders.append(order)
    return orders


def total_orders(repeats,image_index):
    orders=[]
    for rep in range(repeats):
        arms=[0,1] if (rep//2+image_index)%2==0 else [1,0]
        order=[]
        for arm in arms:
            pair=[2*arm,2*arm+1]
            if (rep+image_index+arm)%2:pair.reverse()
            order.extend(pair)
        orders.append(order)
    return orders


def carryover_orders(repeats,image_index,seed):
    """Balance A/B within previous-arm AND absolute pair-position strata.

    One unmeasured first-arm burn-in initializes the previous-workload state.
    Labels are randomized using counts alone; no observed timing is consulted.
    """
    sequences=[];strata={};previous=0 if image_index%2==0 else 1
    for rep in range(repeats):
        arms=[0,1] if (rep//2+image_index)%2==0 else [1,0]
        sequences.append(arms)
        for position,arm in enumerate(arms):
            strata.setdefault((arm,position,previous),[]).append(rep)
            previous=arm
    rng=random.Random(seed);afirst={}
    for (arm,position,previous),reps in sorted(strata.items()):
        if len(reps)%2:raise ValueError('carryover stratum must be even')
        shuffled=reps[:];rng.shuffle(shuffled)
        for rep in reps:afirst[rep,arm]=rep in shuffled[:len(reps)//2]
    result=[]
    for rep,arms in enumerate(sequences):
        order=[]
        for arm in arms:
            pair=[2*arm,2*arm+1]
            if not afirst[rep,arm]:pair.reverse()
            order.extend(pair)
        result.append(order)
    return result


def jobs(net,clean,attacked,pred_proto,suppress,full,ops,write,image,total_only=False):
    result=[]
    for arm,im in [('clean',clean),('attacked',attacked)]:
        raw=net(im);pred,proto=pred_proto(raw)
        det=suppress(pred.clone())
        write(kind='counts',image=image,arm=arm,n_tau=int((pred[:,4:84].amax(1)>.25).sum()),k=len(det),
              prediction_sha256=hashlib.sha256(pred.numpy().tobytes()).hexdigest(),proto_shape=list(proto.shape))
        if total_only:
            same_function=lambda x=im:full(x)
            result.extend([(arm,'aa_A',same_function,lambda:()),
                           (arm,'aa_B',same_function,lambda:())])
            continue
        def mask(pr=proto,d=det):
            return None if not len(d) else ops.process_mask_native(pr,d[:,6:],d[:,:4],(640,640))
        result.extend([(arm,'forward',lambda x=im:net(x),lambda:()),
                       (arm,'nms_tv',lambda p:suppress(p),lambda p=pred:(p.clone(),)),
                       (arm,'mask_native',mask,lambda:()),
                       (arm,'total',lambda x=im:full(x),lambda:()),
                       (arm,'aa_A',lambda x=im:full(x),lambda:()),
                       (arm,'aa_B',lambda x=im:full(x),lambda:())])
    return result
