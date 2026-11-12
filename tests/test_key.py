

    
"""Tests for key detection with the Krumhansl-Schmuckler algorithm."""

import numpy as np
import pytest

from djmath.harmony import chord_frequencies, chromagram, midi_to_frequency
from djmath.key import (
    MAJOR_PROFILE,
    correlations,
    estimate_key,
    key_label,
    key_profiles,
)
from djmath.signals import harmonic_tone, stft


def synthesize_progression(chords, duration=0.5):
    """Concatenate triads given as (root MIDI note, chord type) pairs."""
    segments = []
    for root_midi, chord_type in chords:
        notes = chord_frequencies(midi_to_frequency(root_midi), chord_type)
        segments.append(sum(harmonic_tone(f, duration, n_harmonics=6) for f in notes))
    return np.concatenate(segments)


def chroma_of(signal):
    freqs, _, mags = stft(signal)
    return chromagram(freqs, mags)


def test_key_labels():
    assert key_label(0, "major") == "C major"
    assert key_label(21, "minor") == "A minor"
    with pytest.raises(ValueError):
        key_label(0, "dorian")


def test_profiles_are_rotations():
    labels, profiles = key_profiles()
    assert profiles.shape == (24, 12)
    g_major = profiles[labels.index("G major")]
    assert g_major[7] == MAJOR_PROFILE[0]          # tonic of G is pitch class 7
    assert np.allclose(g_major, np.roll(MAJOR_PROFILE, 7))


def test_correlation_with_itself_is_one():
    _, profiles = key_profiles()
    scores = correlations(profiles[0], profiles)
    assert scores[0] == pytest.approx(1)
    assert np.all(scores <= 1 + 1e-12)


def test_correlation_ignores_scale_and_offset():
    _, profiles = key_profiles()
    v = np.random.default_rng(0).random(12)
    assert np.allclose(correlations(v, profiles), correlations(3 * v + 5, profiles))


def test_constant_vector_is_rejected():
    _, profiles = key_profiles()
    with pytest.raises(ValueError):
        correlations(np.ones(12), profiles)


@pytest.mark.parametrize("tonic", range(12))
def test_detects_major_keys_from_I_IV_V_I(tonic):
    root = 60 + tonic
    x = synthesize_progression(
        [(root, "major"), (root + 5, "major"), (root + 7, "major"), (root, "major")]
    )
    detected, _ = estimate_key(chroma_of(x))
    assert detected == key_label(tonic, "major")


@pytest.mark.parametrize("tonic", range(12))
def test_detects_minor_keys_from_i_iv_V_i(tonic):
    root = 60 + tonic
    x = synthesize_progression(
        [(root, "minor"), (root + 5, "minor"), (root + 7, "major"), (root, "minor")]
    )
    detected, _ = estimate_key(chroma_of(x))
    assert detected == key_label(tonic, "minor")

