from pathlib import Path
from collections import Counter

import json
import pandas as pd


COCO_SMALL_AREA = 32 ** 2
COCO_MEDIUM_AREA = 96 ** 2


def load_coco_annotations(annotation_file: str) -> dict:
    """Load a COCO annotation JSON file."""
    annotation_path = Path(annotation_file)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_object_size(area: float) -> str:
    """Classify an object using COCO's standard area definitions."""
    if area < COCO_SMALL_AREA:
        return "small"

    if area < COCO_MEDIUM_AREA:
        return "medium"

    return "large"


def build_image_metadata(annotation_data: dict) -> pd.DataFrame:
    """
    Build one row per image containing object-size statistics
    and category information.
    """

    images = {
        image["id"]: image
        for image in annotation_data["images"]
    }

    image_annotations = {
        image_id: []
        for image_id in images
    }

    for annotation in annotation_data["annotations"]:
        image_id = annotation["image_id"]

        if image_id in image_annotations:
            image_annotations[image_id].append(annotation)

    rows = []

    for image_id, image in images.items():
        annotations = image_annotations[image_id]

        size_counts = Counter()
        categories = set()

        for annotation in annotations:
            area = annotation.get("area", 0)
            size = classify_object_size(area)

            size_counts[size] += 1
            categories.add(annotation["category_id"])

        total_objects = sum(size_counts.values())

        if total_objects == 0:
            small_fraction = 0.0
            medium_fraction = 0.0
            large_fraction = 0.0
        else:
            small_fraction = size_counts["small"] / total_objects
            medium_fraction = size_counts["medium"] / total_objects
            large_fraction = size_counts["large"] / total_objects

        rows.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "num_objects": total_objects,
                "num_small": size_counts["small"],
                "num_medium": size_counts["medium"],
                "num_large": size_counts["large"],
                "small_fraction": small_fraction,
                "medium_fraction": medium_fraction,
                "large_fraction": large_fraction,
                "num_categories": len(categories),
                "categories": sorted(categories),
            }
        )

    return pd.DataFrame(rows)