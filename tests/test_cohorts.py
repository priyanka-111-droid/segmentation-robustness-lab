import pandas as pd


def test_cohorts_have_expected_size():
    baseline = pd.read_csv("data/cohorts/baseline.csv")
    shifted = pd.read_csv("data/cohorts/small_object_shift.csv")

    assert len(baseline) == 200
    assert len(shifted) == 200


def test_cohorts_do_not_overlap():
    baseline = pd.read_csv("data/cohorts/baseline.csv")
    shifted = pd.read_csv("data/cohorts/small_object_shift.csv")

    baseline_ids = set(baseline["image_id"])
    shifted_ids = set(shifted["image_id"])

    assert baseline_ids.isdisjoint(shifted_ids)


def test_shift_has_more_small_objects():
    baseline = pd.read_csv("data/cohorts/baseline.csv")
    shifted = pd.read_csv("data/cohorts/small_object_shift.csv")

    assert shifted["small_fraction"].mean() > baseline["small_fraction"].mean()