"""Project-specific, function-preserving switches for YOLOv13 ablations."""

from __future__ import annotations


ABLATION_VARIANTS = ("full", "no-hyperace", "no-fullpad", "no-dsc3k2")


def apply_ablation(yolo_model, variant: str) -> dict[str, int | str]:
    """Apply one architectural ablation to a loaded Ultralytics YOLO model.

    The full model is left unchanged. HyperACE is bypassed by replacing each
    AdaHG operator with an identity while retaining its surrounding projection
    paths. FullPAD tunnels return their original feature only. DSConv operators
    inside DSC3k2 blocks are converted to mathematically equivalent dense
    convolutions initialized from the pretrained depthwise/pointwise weights.
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation {variant!r}; choose one of {ABLATION_VARIANTS}.")

    import torch
    from ultralytics.nn.modules.block import AdaHGComputation, DSC3k2, FullPAD_Tunnel
    from ultralytics.nn.modules.conv import DSConv

    core = getattr(yolo_model, "model", yolo_model)
    modules = list(core.modules())
    counts: dict[str, int] = {"variant": variant, "target_modules": 0, "converted_convolutions": 0}

    if variant == "full":
        return counts

    if variant == "no-hyperace":
        targets = [module for module in modules if isinstance(module, AdaHGComputation)]
        for module in targets:
            module.ablate_hypergraph = True
            module.hgnn = torch.nn.Identity()
        counts["target_modules"] = len(targets)

    elif variant == "no-fullpad":
        targets = [module for module in modules if isinstance(module, FullPAD_Tunnel)]
        for module in targets:
            module.disable_distribution = True
            module.gate = None
        counts["target_modules"] = len(targets)

    elif variant == "no-dsc3k2":
        blocks = [module for module in modules if isinstance(module, DSC3k2)]
        converted = 0
        for block in blocks:
            for module in list(block.modules()):
                if isinstance(module, DSConv):
                    module.convert_to_standard()
                    converted += 1
        counts["target_modules"] = len(blocks)
        counts["converted_convolutions"] = converted

    if counts["target_modules"] == 0:
        raise RuntimeError(f"Ablation {variant!r} found no matching modules in this model.")
    if variant == "no-dsc3k2" and counts["converted_convolutions"] == 0:
        raise RuntimeError("DSC3k2 blocks were found, but contained no DSConv modules to replace.")
    return counts
