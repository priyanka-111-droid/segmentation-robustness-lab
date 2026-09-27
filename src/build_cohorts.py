from pathlib import Path

import pandas as pd


def build_baseline_cohort(
    metadata: pd.DataFrame,
    n_images: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample a representative baseline cohort.

    Images are sampled randomly using a fixed seed.
    Images without annotations are excluded.
    """

    eligible = metadata[metadata["num_objects"] > 0].copy()

    if n_images > len(eligible):
        raise ValueError(
            f"Requested {n_images} images, "
            f"but only {len(eligible)} annotated images are available."
        )

    return eligible.sample(
        n=n_images,
        random_state=seed,
    ).reset_index(drop=True)


def build_small_object_cohort(
    metadata: pd.DataFrame,
    n_images: int,
    min_small_fraction: float = 0.50,
) -> pd.DataFrame:
    """
    Select images in which at least `min_small_fraction` of
    annotated objects are small.

    Images are ranked by their small-object fraction.
    """

    eligible = metadata[
        (metadata["num_objects"] > 0)
        & (metadata["small_fraction"] >= min_small_fraction)
    ].copy()

    if len(eligible) < n_images:
        raise ValueError(
            f"Only {len(eligible)} images satisfy the small-object "
            f"criterion, but {n_images} were requested."
        )

    return (
        eligible
        .sort_values(
            ["small_fraction", "num_objects"],
            ascending=[False, False],
        )
        .head(n_images)
        .reset_index(drop=True)
    )


def save_cohort(
    cohort: pd.DataFrame,
    output_path: str,
) -> None:
    """Save a cohort definition to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cohort.to_csv(path, index=False)