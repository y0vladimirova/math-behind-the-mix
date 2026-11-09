
"""Tests for djmath.harmony: tuning, chords and dissonance."""

import numpy as np
import pytest

from djmath.harmony import (
    JUST_RATIOS,
    cents,
    chord_dissonance,
    chord_frequencies,
    dissonance_curve,
    equal_tempered_ratio,
    frequency_to_midi,
    midi_to_frequency,
    note_name,
    pair_dissonance,
)


def test_a4_is_midi_69_and_440_hz():
    assert midi_to_frequency(69) == pytest.approx(440.0)
    assert frequency_to_midi(440.0) == pytest.approx(69)


def test_octave_doubles_frequency():
    assert midi_to_frequency(81) == pytest.approx(2 * midi_to_frequency(69))


@pytest.mark.parametrize("midi", [21, 60, 64.5, 108])
def test_midi_frequency_round_trip(midi):
    assert frequency_to_midi(midi_to_frequency(midi)) == pytest.approx(midi)


@pytest.mark.parametrize(
    "frequency, name", [(440.0, "A"), (261.63, "C"), (392.0, "G"), (880.0, "A")]
)
def test_note_name(frequency, name):
    assert note_name(frequency) == name


def test_cents_of_octave_and_fifth():
    assert cents(2) == pytest.approx(1200)
    assert cents(JUST_RATIOS[7]) == pytest.approx(701.955, abs=1e-3)


def test_equal_temperament_errors():
    # Known deviations of equal temperament from just intonation, in cents
    assert cents(equal_tempered_ratio(7)) - cents(JUST_RATIOS[7]) == pytest.approx(-1.955, abs=1e-3)
    assert cents(equal_tempered_ratio(4)) - cents(JUST_RATIOS[4]) == pytest.approx(13.686, abs=1e-3)


def test_pythagorean_comma():
    # Twelve pure fifths overshoot seven octaves by about 23.46 cents
    assert cents((3 / 2) ** 12 / 2**7) == pytest.approx(23.46, abs=1e-2)


def test_just_major_triad_is_4_5_6():
    assert np.allclose(chord_frequencies(440, "major", "just"), [440, 550, 660])


def test_unknown_chord_type_is_rejected():
    with pytest.raises(ValueError):
        chord_frequencies(440, "sus4")


def test_pair_dissonance_properties():
    assert pair_dissonance(440, 440) == pytest.approx(0)
    assert pair_dissonance(440, 460) == pytest.approx(pair_dissonance(460, 440))
    assert pair_dissonance(440, 460) > pair_dissonance(440, 880)


def test_dissonance_curve_has_minima_at_simple_ratios():
    ratios = np.linspace(1, 2.3, 1301)
    curve = dissonance_curve(261.63, ratios)
    is_min = (curve[1:-1] < curve[:-2]) & (curve[1:-1] < curve[2:])
    minima = ratios[1:-1][is_min]
    for simple in [5 / 4, 4 / 3, 3 / 2, 2]:
        assert np.min(np.abs(minima - simple)) < 0.002


def test_major_triad_less_dissonant_than_cluster():
    root = 261.63
    major = chord_dissonance(chord_frequencies(root, "major"))
    cluster = chord_dissonance([root * equal_tempered_ratio(s) for s in (0, 1, 2)])
    assert major < cluster
    
