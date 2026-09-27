from src.evaluate import relative_change


def test_relative_change():
    baseline = 0.80
    shifted = 0.68

    change = relative_change(baseline, shifted)

    assert abs(change - (-0.15)) < 1e-6


def test_relative_change_improvement():
    baseline = 0.50
    shifted = 0.60

    change = relative_change(baseline, shifted)

    assert abs(change - 0.20) < 1e-6