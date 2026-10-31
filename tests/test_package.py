"""Smoke tests: the package and its modules import, and constants are sane."""

import importlib

import pytest

import djmath

MODULES = ["signals", "harmony", "key", "tempo", "mixing", "setlist"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"djmath.{name}")


def test_twelve_unique_pitch_classes():
    assert len(djmath.PITCH_CLASSES) == 12
    assert len(set(djmath.PITCH_CLASSES)) == 12


def test_sample_rate_covers_audible_musical_range():
    # Nyquist frequency must exceed the highest piano note (C8 ~ 4186 Hz).
    assert djmath.SAMPLE_RATE / 2 > 4186
