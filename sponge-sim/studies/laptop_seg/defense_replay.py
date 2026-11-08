"""Paired defense jobs on the SAME frozen attacked tensor, no optimization."""
import hashlib


def observed_suppression(pred, suppress, torchvision, setting):
    """Count actual compiled-kernel inputs outside timing; restore in finally."""
    original = torchvision.ops.nms
    calls = []

    def recorded(boxes, scores, iou_threshold):
        calls.append(len(scores))
        return original(boxes, scores, iou_threshold)

    torchvision.ops.nms = recorded
    try:
        det = suppress(pred.clone(), **setting)
    finally:
        torchvision.ops.nms = original
    if len(calls) > 1:
        raise RuntimeError('unexpected multiple NMS kernel invocations')
    return det, 0 if not calls else calls[0]


def jobs(net, attacked, pred_proto, suppress, full, torchvision, write, image, setting):
    pred, _ = pred_proto(net(attacked))
    digest = hashlib.sha256(pred.numpy().tobytes()).hexdigest()
    n_tau = int((pred[:, 4:84].amax(1) > .25).sum())
    result = []
    for arm, options in [('baseline', dict(max_det=300, max_nms=30000)), ('defended', setting)]:
        det, kernel_n = observed_suppression(pred, suppress, torchvision, options)
        if kernel_n != min(n_tau, options['max_nms']) or len(det) > options['max_det']:
            raise RuntimeError('defense boundary/count contract failed')
        write(kind='counts', image=image, arm=arm, input_kind='same_frozen_attacked',
              n_tau=n_tau, kernel_n=kernel_n, k=len(det), prediction_sha256=digest,
              detection_sha256=hashlib.sha256(det.numpy().tobytes()).hexdigest(), options=options)
        same_function = lambda options=options: full(attacked, **options)
        result.extend([(arm, 'aa_A', same_function, lambda:()),
                       (arm, 'aa_B', same_function, lambda:())])
    return result
