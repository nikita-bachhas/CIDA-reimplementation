"""Support selection uses class-local positions and does not require Torch."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('digit_support_picker',
    Path(__file__).resolve().parents[1] / 'scripts/choose_digit_supports.py')
picker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(picker)


def test_mixed_seed_choices_resolve_to_class_positions():
    ds = SimpleNamespace(labels=[0, 5, 6, 5, 0, 6, 5, 6], class_names=list(map(str, range(10))))
    draws = [{5: [1], 6: [2]}, {5: [6], 6: [7]}]
    selected = picker.support_positions(ds, [5, 6], draws, [1, 0])
    assert selected == {'5': 2, '6': 0}
    for c, expected in [(5, 6), (6, 2)]:
        members = [i for i, label in enumerate(ds.labels) if label == c]
        assert members[selected[str(c)]] == expected


@pytest.mark.parametrize('picks', [[], [0], [0, 1, 0], [-1, 0], [0, 2]])
def test_reject_incomplete_or_out_of_range_choices(picks):
    ds = SimpleNamespace(labels=[5, 6], class_names=list(map(str, range(10))))
    with pytest.raises(ValueError):
        picker.support_positions(ds, [5, 6], [{5: [0], 6: [1]}] * 2, picks)
