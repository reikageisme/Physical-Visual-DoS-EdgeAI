# Ablation analysis

No measured ablation data was found. Fill results/ablation.csv with controlled runs listed in configs/ablation_template.csv. No component-level performance attribution can be made yet.

# Published YOLOv13 ablation reference

These are the original paper's MS COCO validation ablations, not measurements on this project's dataset.

| Experiment | Variant | Dataset | mAP@0.5:0.95 (%) | mAP@0.5 (%) | Params (M) | FLOPs (G) |
|---|---|---|---:|---:|---:|---:|
| FullPAD-HyperACE | No distribution (paper treats as HyperACE removed) | MS COCO val | 47.1 | 64.1 | -- | -- |
| FullPAD-HyperACE | Backbone-to-neck only | MS COCO val | 47.8 | 65.0 | -- | -- |
| FullPAD-HyperACE | In-neck only | MS COCO val | 47.6 | 64.8 | -- | -- |
| FullPAD-HyperACE | Neck-to-head only | MS COCO val | 47.7 | 64.9 | -- | -- |
| FullPAD-HyperACE | All three FullPAD paths | MS COCO val | 48.0 | 65.2 | -- | -- |
| DS-blocks | YOLOv13-N with DS blocks | MS COCO val | 41.6 | 57.8 | 2.5 | 6.4 |
| DS-blocks | YOLOv13-N with vanilla blocks | MS COCO val | 41.6 | 57.9 | 3.1 | 7.9 |
| DS-blocks | YOLOv13-S with DS blocks | MS COCO val | 48.0 | 65.2 | 9.0 | 20.8 |
| DS-blocks | YOLOv13-S with vanilla blocks | MS COCO val | 48.0 | 65.3 | 11.7 | 27.1 |
| Hyperedge-count | M=2 | MS COCO val | 47.8 | 64.7 | 8.6 | 20.4 |
| Hyperedge-count | M=4 | MS COCO val | 47.9 | 65.0 | 8.8 | 20.5 |
| Hyperedge-count | M=8 | MS COCO val | 48.0 | 65.2 | 9.0 | 20.8 |
| Hyperedge-count | M=16 | MS COCO val | 48.1 | 65.2 | 9.6 | 21.5 |

## Quantitative reading

- On YOLOv13-S, all three FullPAD paths score +0.9 percentage points in AP50:95 and +1.1 points in AP50 over the paper's no-distribution setting, which the authors treat as removing HyperACE.
- The paper's Table V gives DS-block savings of 1.5 GFLOPs and 0.6M parameters for Nano, and 6.3 GFLOPs and 2.7M for Small, with unchanged AP50:95 and a 0.1-point lower AP50 in both tabulated comparisons.
- The paper's prose reports different Small-model DS savings (4.2 GFLOPs and 2.2M parameters) than the Table V arithmetic (6.3 GFLOPs and 2.7M); this report uses the table values and flags the discrepancy.
- For YOLOv13-S, increasing hyperedges from M=2 to M=8 changes AP50:95 by +0.2 points; moving from M=8 to M=16 adds +0.1 points while adding 0.7 GFLOPs and 0.6M parameters.

Use these results to motivate the project-specific ablations; do not present them as outcomes on the project's own data.

Source: [Lei et al., YOLOv13, arXiv:2506.17733](https://arxiv.org/abs/2506.17733), Tables III-V and accompanying discussion.
