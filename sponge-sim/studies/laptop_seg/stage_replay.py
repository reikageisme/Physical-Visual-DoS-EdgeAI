"""Prepared CPU component controls, with fixed captured torchvision inputs.

Kernel endpoint is an amortized cached 64-call block INCLUDING Python facade,
loop and index-output release. Native-mask endpoint is one prepared-feature
call, with output release outside both clocks. Neither is service latency.
"""
import hashlib


def digest(tensor):return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def capture_kernel(pred,suppress,torchvision):
    """Intercept only untimed preparation; restore original facade in finally."""
    original=torchvision.ops.nms;calls=[]
    def observed(boxes,scores,iou):
        frozen=(boxes.clone().contiguous(),scores.clone().contiguous(),float(iou))
        indices=original(boxes,scores,iou)
        calls.append((*frozen,indices.clone()))
        return indices
    torchvision.ops.nms=observed
    try:det=suppress(pred.clone())
    finally:torchvision.ops.nms=original
    if len(calls)!=1:raise RuntimeError('expected exactly one fixed torchvision kernel call')
    return det,calls[0]


def jobs(net,clean,attacked,pred_proto,suppress,ops,torchvision,write,image,cfg):
    endpoint=cfg['isolated_stage'];calls=cfg['calls_per_trial']
    if type(calls) is not int or (endpoint,calls) not in (('nms_kernel',64),('mask_native',1)):
        raise ValueError('registered endpoint/call-count pair required')
    result=[];guards=[]
    for arm,im in [('clean',clean),('attacked',attacked)]:
        pred,proto=pred_proto(net(im))
        det,(boxes,scores,iou,indices)=capture_kernel(pred,suppress,torchvision)
        n_tau=int((pred[:,4:84].amax(1)>.25).sum())
        if len(scores)!=n_tau or len(det)!=min(len(indices),300):
            raise RuntimeError('kernel input/output boundary mismatch')
        meta=dict(kind='counts',image=image,arm=arm,n_tau=n_tau,k=len(det),kernel_n=len(scores),
                  kernel_kept=len(indices),prediction_sha256=digest(pred),proto_shape=list(proto.shape),
                  kernel_boxes_sha256=digest(boxes),kernel_scores_sha256=digest(scores),
                  kernel_indices_sha256=digest(indices),kernel_iou=iou,endpoint=endpoint)
        if endpoint=='nms_kernel':
            original=torchvision.ops.nms
            payload=(boxes,scores);before=[digest(t) for t in payload]
            # Check captured-input semantics once, outside timing.
            check=original(boxes,scores,iou)
            if digest(check)!=digest(indices):raise RuntimeError('captured kernel replay differs from wrapper')
            del check
            def fn(b=boxes,s=scores,t=iou,kernel=original):
                for _ in range(64):
                    selected=kernel(b,s,t)
                    del selected
                return None
        else:
            payload=(proto,det);before=[digest(t) for t in payload]
            def fn(pr=proto,d=det):
                return None if not len(d) else ops.process_mask_native(pr,d[:,6:],d[:,:4],(640,640))
            sample=fn()
            meta.update(mask_dtype=None if sample is None else str(sample.dtype),
                        mask_shape=None if sample is None else list(sample.shape))
            del sample
        if before!=[digest(t) for t in payload]:raise RuntimeError('untimed stage probe mutated its inputs')
        guards.append((arm,payload,before))
        write(**meta)
        result.extend([(arm,'aa_A',fn,lambda:()),(arm,'aa_B',fn,lambda:())])
    def verify():
        for arm,payload,before in guards:
            after=[digest(t) for t in payload]
            write(kind='stage_input_check',image=image,arm=arm,endpoint=endpoint,
                  before_sha256=before,after_sha256=after,unchanged=before==after)
            if before!=after:raise RuntimeError('prepared stage input mutated during replay')
    return result,verify
