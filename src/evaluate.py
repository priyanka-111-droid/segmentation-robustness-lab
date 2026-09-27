from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


def load_cohort(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_coco_image_ids(
    coco: COCO,
    cohort: pd.DataFrame,
) -> list[int]:
    """Verify that cohort image IDs exist in COCO annotations."""
    available_ids = set(coco.getImgIds())

    image_ids = cohort["image_id"].astype(int).tolist()

    missing = set(image_ids) - available_ids

    if missing:
        raise ValueError(
            f"{len(missing)} cohort image IDs are missing from COCO."
        )

    return image_ids


def run_predictions(
    model: YOLO,
    image_paths: list[str],
):
    """Run identical inference on a cohort."""
    return model.predict(
        source=image_paths,
        imgsz=640,
        conf=0.25,
        iou=0.70,
        verbose=False,
        save=False,
    )

def build_coco_predictions(
    results,
    image_ids: list[int],
    model_names: dict,
    coco: COCO,
):
    """
    Convert Ultralytics predictions into COCO result format.

    Includes bounding boxes and segmentation masks.
    """
    category_name_to_id = {
        category["name"]: category["id"]
        for category in coco.dataset["categories"]
    }

    predictions = []

    for result, image_id in zip(results, image_ids):
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        mask_data = None

        if result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            class_index = int(classes[i])
            class_name = model_names[class_index]

            if class_name not in category_name_to_id:
                raise ValueError(
                    f"Class '{class_name}' not found in COCO categories."
                )

            category_id = category_name_to_id[class_name]

            prediction = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(width),
                    float(height),
                ],
                "score": float(scores[i]),
            }

            if mask_data is not None:
                mask = mask_data[i]

                orig_height, orig_width = result.orig_shape

                if mask.shape != (orig_height, orig_width):
                    import cv2

                    mask = cv2.resize(
                        mask,
                        (orig_width, orig_height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                binary_mask = (mask > 0.5).astype("uint8")

                # COCO requires Fortran-contiguous masks for RLE encoding.
                binary_mask = binary_mask.astype("uint8")
                binary_mask = binary_mask[:, :, None]

                from pycocotools import mask as mask_utils

                rle = mask_utils.encode(
                    binary_mask[:, :, 0]
                )

                # pycocotools returns bytes for counts.
                rle["counts"] = rle["counts"].decode("utf-8")

                prediction["segmentation"] = rle

            predictions.append(prediction)

    return predictions


def evaluate_coco(
    coco: COCO,
    predictions: list[dict],
    image_ids: list[int],
    iou_type: str,
):
    """Run COCO evaluation for bbox or segmentation."""
    if not predictions:
        raise ValueError("No predictions were generated.")

    prediction_api = coco.loadRes(predictions)

    evaluator = COCOeval(
        coco,
        prediction_api,
        iouType=iou_type,
    )

    evaluator.params.imgIds = image_ids

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "AP50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR_1": float(evaluator.stats[6]),
        "AR_10": float(evaluator.stats[7]),
        "AR_100": float(evaluator.stats[8]),
        "AR_small": float(evaluator.stats[9]),
        "AR_medium": float(evaluator.stats[10]),
        "AR_large": float(evaluator.stats[11]),
    }


def relative_change(
    baseline: float,
    shifted: float,
) -> float:
    """Return relative change from baseline to shifted."""
    if baseline == 0:
        return float("nan")

    return (shifted - baseline) / baseline


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO segmentation robustness on fixed cohorts."
    )

    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
    )

    parser.add_argument(
        "--annotations",
        default="data/annotations/instances_val2017.json",
    )

    parser.add_argument(
        "--image-dir",
        default="data/images/val2017",
    )

    parser.add_argument(
        "--baseline",
        default="data/cohorts/baseline.csv",
    )

    parser.add_argument(
        "--shifted",
        default="data/cohorts/small_object_shift.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations...")
    coco = COCO(args.annotations)

    print("Loading cohorts...")
    baseline = load_cohort(args.baseline)
    shifted = load_cohort(args.shifted)

    baseline_ids = get_coco_image_ids(
        coco,
        baseline,
    )

    shifted_ids = get_coco_image_ids(
        coco,
        shifted,
    )

    baseline_paths = [
        str(Path(args.image_dir) / filename)
        for filename in baseline["file_name"]
    ]

    shifted_paths = [
        str(Path(args.image_dir) / filename)
        for filename in shifted["file_name"]
    ]

    for path in baseline_paths + shifted_paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    print("\nExperiment configuration")
    print("-" * 40)
    print(f"Model:              {args.model}")
    print(f"Image size:         640")
    print(f"Confidence:         0.25")
    print(f"IoU:                0.70")
    print(f"Baseline images:    {len(baseline_ids)}")
    print(f"Small-shift images: {len(shifted_ids)}")

    print("\nLoading model...")
    model = YOLO(args.model)

    print("\nRunning baseline inference...")
    baseline_results = run_predictions(
        model,
        baseline_paths,
    )

    print("Running small-object-shift inference...")
    shifted_results = run_predictions(
        model,
        shifted_paths,
    )

    print("\nConverting predictions...")
    baseline_predictions = build_coco_predictions(
        baseline_results,
        baseline_ids,
        model.names,
        coco,
    )
    
    shifted_predictions = build_coco_predictions(
        shifted_results,
        shifted_ids,
        model.names,
        coco,
    )

    print("\nEvaluating bounding boxes...")
    baseline_box_metrics = evaluate_coco(
        coco,
        baseline_predictions,
        baseline_ids,
        "bbox",
    )

    shifted_box_metrics = evaluate_coco(
        coco,
        shifted_predictions,
        shifted_ids,
        "bbox",
    )

    print("\nEvaluating segmentation masks...")

    baseline_mask_metrics = evaluate_coco(
        coco,
        baseline_predictions,
        baseline_ids,
        "segm",
    )

    shifted_mask_metrics = evaluate_coco(
        coco,
        shifted_predictions,
        shifted_ids,
        "segm",
    )

    results = {
        "experiment": {
            "model": args.model,
            "imgsz": 640,
            "confidence": 0.25,
            "iou": 0.70,
            "baseline_images": len(baseline_ids),
            "shifted_images": len(shifted_ids),
        },
        "baseline": {
            "bbox": baseline_box_metrics,
            "segm": baseline_mask_metrics,
        },
        "small_object_shift": {
            "bbox": shifted_box_metrics,
            "segm": shifted_mask_metrics,
        },
        "relative_change": {
            "bbox_AP50_95": relative_change(
                baseline_box_metrics["AP50_95"],
                shifted_box_metrics["AP50_95"],
            ),
            "segm_AP50_95": relative_change(
                baseline_mask_metrics["AP50_95"],
                shifted_mask_metrics["AP50_95"],
            ),
        },
        "failure_rule": {
    "primary_metric": "segm_AP50_95",
    "relative_degradation_threshold": -0.15,
}
    }

    output_file = output_dir / "results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
