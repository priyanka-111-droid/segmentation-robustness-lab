from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd


def parse_category_counts(value) -> dict[int, int]:
    """Convert the CSV representation back into a category-count dictionary."""
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}

    parsed = literal_eval(value)
    return {int(k): int(v) for k, v in parsed.items()}


def category_distribution(cohort: pd.DataFrame) -> dict[int, float]:
    """Return the fraction of all instances belonging to each category."""
    counts: dict[int, int] = {}

    for value in cohort["category_counts"]:
        for category_id, count in parse_category_counts(value).items():
            counts[category_id] = counts.get(category_id, 0) + count

    total = sum(counts.values())

    if total == 0:
        return {}

    return {
        category_id: count / total
        for category_id, count in counts.items()
    }


def distribution_distance(
    baseline: pd.DataFrame,
    shifted: pd.DataFrame,
) -> float:
    """
    Calculate total variation distance between category distributions.

    0.0 means identical distributions.
    Larger values mean greater category mismatch.
    """
    baseline_dist = category_distribution(baseline)
    shifted_dist = category_distribution(shifted)

    categories = set(baseline_dist) | set(shifted_dist)

    return 0.5 * sum(
        abs(
            baseline_dist.get(category_id, 0.0)
            - shifted_dist.get(category_id, 0.0)
        )
        for category_id in categories
    )


def build_candidate_pools(
    metadata: pd.DataFrame,
    min_small_fraction: float = 0.50,
    max_baseline_small_fraction: float = 0.35,
):
    """
    Separate candidate images into baseline and small-object-shift pools.
    """
    usable = metadata[metadata["num_objects"] > 0].copy()

    baseline = usable[
        usable["small_fraction"] <= max_baseline_small_fraction
    ].copy()

    shifted = usable[
        usable["small_fraction"] >= min_small_fraction
    ].copy()

    return baseline, shifted


def build_matched_cohorts(
    metadata: pd.DataFrame,
    n_images: int = 200,
    min_small_fraction: float = 0.50,
    max_baseline_small_fraction: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:

    rng = np.random.default_rng(seed)

    baseline_pool, shifted_pool = build_candidate_pools(
        metadata,
        min_small_fraction=min_small_fraction,
        max_baseline_small_fraction=max_baseline_small_fraction,
    )

    if len(baseline_pool) < n_images:
        raise ValueError(
            f"Only {len(baseline_pool)} baseline candidates available."
        )

    if len(shifted_pool) < n_images:
        raise ValueError(
            f"Only {len(shifted_pool)} shifted candidates available."
        )

    # Randomize candidates so we don't systematically select
    # particular COCO image IDs.
    baseline_pool = baseline_pool.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    shifted_pool = shifted_pool.sample(
        frac=1,
        random_state=seed + 1,
    ).reset_index(drop=True)

    # Start with a random baseline cohort.
    baseline = baseline_pool.head(n_images).copy()

    # Greedily construct the shifted cohort.
    #
    # At each step, choose the candidate whose category distribution
    # keeps the two cohorts as similar as possible while maintaining
    # the desired small-object shift.
    selected_indices = []

    current_shifted = shifted_pool.iloc[0:0].copy()

    for _ in range(n_images):
        best_idx = None
        best_distance = float("inf")

        remaining = shifted_pool.drop(index=selected_indices)

        # Evaluate a random subset of candidates for efficiency.
        candidate_count = min(100, len(remaining))

        candidate_indices = rng.choice(
            remaining.index.to_numpy(),
            size=candidate_count,
            replace=False,
        )

        for idx in candidate_indices:
            candidate = remaining.loc[[idx]]

            trial = pd.concat(
                [current_shifted, candidate],
                ignore_index=True,
            )

            distance = distribution_distance(
                baseline,
                trial,
            )

            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        selected_indices.append(best_idx)

        current_shifted = shifted_pool.loc[
            selected_indices
        ].reset_index(drop=True)

    shifted = current_shifted

    distance = distribution_distance(
        baseline,
        shifted,
    )

    return (
        baseline.reset_index(drop=True),
        shifted.reset_index(drop=True),
        distance,
    )


def save_cohort(cohort: pd.DataFrame, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(path, index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build category-matched COCO object-size cohorts."
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to image metadata CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/cohorts",
        help="Directory for generated cohorts.",
    )

    parser.add_argument(
        "--n-images",
        type=int,
        default=200,
        help="Number of images per cohort.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)

    baseline, shifted, distance = build_matched_cohorts(
        metadata,
        n_images=args.n_images,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cohort(
        baseline,
        output_dir / "baseline.csv",
    )

    save_cohort(
        shifted,
        output_dir / "small_object_shift.csv",
    )

    print("\nCohort construction complete")
    print("-" * 40)

    print(f"Baseline images:      {len(baseline)}")
    print(f"Small-shift images:   {len(shifted)}")

    print(
        f"\nBaseline small fraction: "
        f"{baseline['small_fraction'].mean():.3f}"
    )

    print(
        f"Shifted small fraction:  "
        f"{shifted['small_fraction'].mean():.3f}"
    )

    print(
        f"\nCategory distribution distance: "
        f"{distance:.4f}"
    )

    print(
        "\nSaved:"
        f"\n  {output_dir / 'baseline.csv'}"
        f"\n  {output_dir / 'small_object_shift.csv'}"
    )


if __name__ == "__main__":
    main()