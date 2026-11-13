
"""Tests for the Camelot wheel, harmonic distance and key scoring."""

from itertools import product

import pytest

from djmath.key import (
    camelot_code,
    harmonic_distance,
    key_label,
    key_profiles,
    key_score,
    parse_key,
    scale_pitch_classes,
)

ALL_KEYS, _ = key_profiles()


@pytest.mark.parametrize("label", ALL_KEYS)
def test_parse_key_inverts_key_label(label):
    assert key_label(*parse_key(label)) == label


def test_scale_pitch_classes():
    assert scale_pitch_classes("C major") == {0, 2, 4, 5, 7, 9, 11}
    assert scale_pitch_classes("A minor") == scale_pitch_classes("C major")


@pytest.mark.parametrize(
    "label, code",
    [
        ("C major", "8B"),
        ("A minor", "8A"),
        ("G major", "9B"),
        ("F major", "7B"),
        ("E minor", "9A"),
        ("F# major", "2B"),
        ("B major", "1B"),
        ("G# minor", "1A"),
    ],
)
def test_known_camelot_codes(label, code):
    assert camelot_code(label) == code


def test_camelot_codes_are_a_bijection():
    codes = {camelot_code(k) for k in ALL_KEYS}
    assert len(codes) == 24


def test_distance_is_a_metric():
    for a in ALL_KEYS:
        assert harmonic_distance(a, a) == 0
    for a, b in product(ALL_KEYS, repeat=2):
        assert harmonic_distance(a, b) == harmonic_distance(b, a)
        if a != b:
            assert harmonic_distance(a, b) > 0
    for a, b, c in product(ALL_KEYS, repeat=3):
        assert harmonic_distance(a, c) <= harmonic_distance(a, b) + harmonic_distance(b, c)


def test_neighbouring_keys_are_at_distance_one():
    assert harmonic_distance("C major", "G major") == 1   # fifth up
    assert harmonic_distance("C major", "F major") == 1   # fifth down
    assert harmonic_distance("C major", "A minor") == 1   # relative
    assert harmonic_distance("C major", "F# major") == 6  # opposite side


def test_largest_distance_is_seven():
    assert max(harmonic_distance(a, b) for a, b in product(ALL_KEYS, repeat=2)) == 7


def test_neighbours_on_the_wheel_share_six_notes():
    for label in ALL_KEYS:
        for other in ALL_KEYS:
            same_letter = camelot_code(label)[-1] == camelot_code(other)[-1]
            if same_letter and harmonic_distance(label, other) == 1:
                assert len(scale_pitch_classes(label) & scale_pitch_classes(other)) == 6


@pytest.mark.parametrize(
    "reference, estimate, score",
    [
        ("C major", "C major", 1.0),
        ("C major", "G major", 0.5),
        ("C major", "A minor", 0.3),
        ("A minor", "C major", 0.3),
        ("C major", "C minor", 0.2),
        ("C major", "F major", 0.0),   # fifth below is not rewarded
        ("C minor", "G major", 0.0),
    ],
)
def test_mirex_key_score(reference, estimate, score):
    assert key_score(reference, estimate) == score
