"""
sponge_fitness.py — Fitness Functions for Sponge Attack Optimization

Two fitness modes:
  1. GrayBoxFitness   : uses pre-NMS raw confidence scores (requires model access)
  2. ObservableFitness: uses only observable system outputs (latency, FPS, post-NMS count)
                        → fully black-box, usable with any detection API

Reference: Major Issue 1 in REVIEWED.md
"""

import time
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Gray-box fitness (original approach, renamed for clarity)
# Requires access to pre-NMS raw anchor scores
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sponge_fitness(
    batch_scores: torch.Tensor,
    conf_thresh: float = 0.01,
    lambda_weight: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [GRAY-BOX] Vectorized fitness over a batch of raw pre-NMS scores.

    Args:
        batch_scores  : GPU tensor [B, 8400] — max class confidence per anchor
        conf_thresh   : threshold to count a box as "active"
        lambda_weight : weight on box count term (DoS emphasis)

    Returns:
        fitness_scores : [B] tensor — higher = more resource pressure
        num_boxes      : [B] tensor — number of active boxes (NMS input count)

    Fitness formula (gray-box):
        F(x_adv) = Σ C_i(x_adv) + λ · N_active(x_adv)
    where:
        C_i(x_adv) : confidence score of anchor i for adversarial input x_adv
        N_active   : count of anchors with C_i > conf_thresh (fed into NMS)
        λ          : weighting factor (default 1.5)
    """
    # 1. Identify active anchors
    active_mask = batch_scores > conf_thresh                       # [B, 8400] bool

    # 2. Count active boxes per image
    num_boxes_per_image = active_mask.sum(dim=1).float()          # [B]

    # 3. Sum confidence of active anchors per image
    total_conf_per_image = (batch_scores * active_mask).sum(dim=1)  # [B]

    # 4. Combined fitness
    fitness_scores = total_conf_per_image + lambda_weight * num_boxes_per_image

    return fitness_scores, num_boxes_per_image


# ─────────────────────────────────────────────────────────────────────────────
# Observable fitness (strict black-box alternative)
# Only uses observable system outputs: latency, FPS drop, post-NMS count
# ─────────────────────────────────────────────────────────────────────────────

class ObservableFitness:
    """
    [BLACK-BOX] Fitness estimation using only externally observable metrics.

    This mode is compatible with strict black-box threat model where attacker
    can only observe:
      - Number of final detections (after NMS)
      - Inference latency (ms)
      - FPS

    Usage:
        obs_fitness = ObservableFitness(baseline_latency_ms=30.0)
        score = obs_fitness.compute(victim_model, adversarial_image_tensor)
    """

    def __init__(
        self,
        baseline_latency_ms: float = None,
        latency_weight: float = 2.0,
        det_count_weight: float = 1.0,
        n_warmup_queries: int = 5,
    ):
        self.baseline_latency_ms = baseline_latency_ms
        self.latency_weight      = latency_weight
        self.det_count_weight    = det_count_weight
        self.n_warmup_queries    = n_warmup_queries

    def calibrate_baseline(self, victim_model, clean_image: torch.Tensor):
        """
        Measure baseline latency on clean image (no patch).
        Call this once before optimization starts.
        """
        latencies = []
        for _ in range(self.n_warmup_queries):
            result = victim_model.get_predictions_with_nms(
                clean_image, conf_thresh=0.25, profile_latency=True
            )
            latencies.append(result['latency_ms']['total_ms'])

        self.baseline_latency_ms = sum(latencies) / len(latencies)
        print(f"[ObservableFitness] Baseline latency: {self.baseline_latency_ms:.1f} ms")
        return self.baseline_latency_ms

    def compute(
        self,
        victim_model,
        adv_image: torch.Tensor,
        conf_thresh: float = 0.25,
        n_queries: int = 3,
    ) -> float:
        """
        Compute observable fitness for a single adversarial image.

        Fitness = latency_weight * (adv_latency / baseline_latency) +
                  det_count_weight * num_final_detections

        Args:
            victim_model : VictimModel instance
            adv_image    : [1, 3, H, W] adversarial image tensor
            conf_thresh  : confidence threshold (controls NMS input)
            n_queries    : number of repeated queries to average latency

        Returns:
            float fitness score (higher = more DoS impact)
        """
        assert self.baseline_latency_ms is not None, \
            "Call calibrate_baseline() first!"

        latencies = []
        num_dets  = []

        for _ in range(n_queries):
            result = victim_model.get_predictions_with_nms(
                adv_image, conf_thresh=conf_thresh, profile_latency=True
            )
            latencies.append(result['latency_ms']['total_ms'])
            num_dets.append(result['num_final_boxes'])

        avg_latency = sum(latencies) / len(latencies)
        avg_dets    = sum(num_dets)  / len(num_dets)

        latency_ratio = avg_latency / max(self.baseline_latency_ms, 1.0)

        fitness = (self.latency_weight * latency_ratio +
                   self.det_count_weight * avg_dets)

        return fitness, {
            'avg_latency_ms'   : avg_latency,
            'latency_ratio'    : latency_ratio,
            'avg_final_dets'   : avg_dets,
        }