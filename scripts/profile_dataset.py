import argparse
from pathlib import Path

from src.dataset import load_coco_annotations, build_image_metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to COCO instances_val2017.json",
    )

    parser.add_argument(
        "--output",
        default="data/image_metadata.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    annotation_data = load_coco_annotations(args.annotations)

    metadata = build_image_metadata(annotation_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_path, index=False)

    print(f"Processed {len(metadata):,} images")
    print(f"Saved metadata to: {output_path}")


if __name__ == "__main__":
    main()