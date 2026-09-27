# Segmentation Robustness Lab

## Research Question

How does segmentation performance change under a natural shift toward smaller objects in the evaluation distribution?

## Motivation

Object-size distribution shifts are a practical robustness concern for object detection and segmentation systems. A model may perform well on a standard evaluation distribution while degrading when the input distribution contains a substantially larger proportion of small objects.

My previous work involved YOLO-based segmentation for detecting components in architectural drawings. That work uses proprietary data and code, so this repository is an independent public experiment using COCO data.

## Hypothesis

A shift toward smaller objects will reduce segmentation performance, with the largest degradation occurring in small-object recall and mask quality.

## Experimental Design

The experiment uses COCO 2017 validation images.

Two evaluation cohorts are constructed:

### Baseline cohort

- 200 images
- Small-object fraction <= 0.35
- Randomly sampled with a fixed seed

### Small-object-shift cohort

- 200 images
- Small-object fraction >= 0.50
- Selected to approximately match the baseline category distribution

The cohorts contain no overlapping images.

The category-distribution distance is measured using total variation distance.

Current cohort statistics:

| Property | Baseline | Small-object shift |
|---|---:|---:|
| Images | 200 | 200 |
| Mean small-object fraction | 0.063 | 0.643 |
| Mean medium-object fraction | 0.367 | 0.162 |
| Mean large-object fraction | 0.570 | 0.196 |
| Mean objects/image | 4.91 | 5.86 |
| Mean categories/image | 2.50 | 3.06 |
| Category TV distance | | 0.0578 |

The cohorts therefore create a substantial shift toward smaller objects while keeping category composition reasonably similar. The remaining differences in object count, number of categories, and image dimensions will be treated as potential confounders rather than ignored.

## Evaluation

The planned model is a pretrained YOLO segmentation model evaluated with fixed inference settings:

- Image size: 640
- Confidence threshold: 0.25
- IoU threshold: 0.70

### Primary metric

**Mask mAP@50:95**

### Secondary metrics

- Mask mAP@50
- Mask mAP@75
- Box mAP@50:95
- Precision
- Recall
- Small-object recall
- Size-conditioned AP/recall

## Failure Rule

The primary failure criterion is:

> Failure is declared if mask mAP@50:95 decreases by more than 15% relative on the small-object-shift cohort compared with the baseline cohort.

Relative change is calculated as:

```text
(shifted_metric - baseline_metric) / baseline_metric
