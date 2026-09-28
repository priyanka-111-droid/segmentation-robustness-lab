from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def load_cohort(path: str) -> pd.DataFrame:
    """Load a cohort CSV."""
    return pd.read_csv(path)


def resolve_image_paths(
    cohort: pd.DataFrame,
    image_dir: str,
) -> list[str]:
    """Resolve COCO image filenames to local paths."""
    image_root = Path(image_dir)

    paths = []

    for filename in cohort["file_name"]:
        path = image_root / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing image: {path}"
            )

        paths.append(str(path))

    return paths


def run_evaluation(
    model: YOLO,
    image_paths: list[str],
    output_dir: str,
):
    """
    Run YOLO validation/inference on a fixed image cohort.

    This function is intentionally isolated so that inference
    settings remain identical across cohorts.
    """

    results = model.predict(
        source=image_paths,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        verbose=False,
        save=False,
    )

    output = {
        "num_images": len(image_paths),
        "num_results": len(results),
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on predefined robustness cohorts."
    )

    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="YOLO segmentation model.",
    )

    parser.add_argument(
        "--image-dir",
        default="data/images/val2017",
        help="COCO val2017 image directory.",
    )

    parser.add_argument(
        "--baseline",
        default="protocol_manifests/baseline.csv",
         help="Frozen baseline cohort manifest."
    )

    parser.add_argument(
        "--shifted",
        default="protocol_manifests/small_object_shift.csv",
        help="Frozen small-object-shift cohort manifest.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for evaluation outputs.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = YOLO(args.model)

    baseline = load_cohort(args.baseline)
    shifted = load_cohort(args.shifted)

    baseline_paths = resolve_image_paths(
        baseline,
        args.image_dir,
    )

    shifted_paths = resolve_image_paths(
        shifted,
        args.image_dir,
    )

    print("\nCohort summary")
    print("-" * 40)
    print(f"Baseline images:    {len(baseline_paths)}")
    print(f"Small-shift images: {len(shifted_paths)}")

    print("\nInference configuration")
    print("-" * 40)
    print("Image size: 640")
    print("Confidence: 0.25")
    print("IoU:        0.70")

    print("\nNOTE:")
    print("Evaluation has NOT been executed.")
    print("This script currently validates cohort/image setup only.")

    manifest = {
        "model": args.model,
        "baseline": {
            "cohort_file": args.baseline,
            "num_images": len(baseline_paths),
        },
        "small_object_shift": {
            "cohort_file": args.shifted,
            "num_images": len(shifted_paths),
        },
        "inference": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.70,
        },
    }

    manifest_path = output_dir / "evaluation_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
