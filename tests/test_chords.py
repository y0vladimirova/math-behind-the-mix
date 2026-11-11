"""Tests for chord recognition by template matching."""

import numpy as np
import pytest

from djmath.harmony import (
    chord_frequencies,
    chord_label,
    chord_templates,
    chromagram,
    midi_to_frequency,
    recognize_chords,
    smooth_chroma,
)
from djmath.signals import harmonic_tone, stft


def synthesize_chord(root_midi, chord_type, duration=0.5):
    notes = chord_frequencies(midi_to_frequency(root_midi), chord_type)
    return sum(harmonic_tone(f, duration, n_harmonics=6) for f in notes) / len(notes)


def test_chord_labels():
    assert chord_label(0, "major") == "C"
    assert chord_label(9, "minor") == "Am"
    assert chord_label(13, "major") == "C#"


def test_templates_are_unit_vectors_with_correct_notes():
    labels, templates = chord_templates()
    assert templates.shape == (24, 12)
    assert np.allclose(np.linalg.norm(templates, axis=1), 1)
    c_major = templates[labels.index("C")]
    assert set(np.flatnonzero(c_major)) == {0, 4, 7}
    a_minor = templates[labels.index("Am")]
    assert set(np.flatnonzero(a_minor)) == {9, 0, 4}


@pytest.mark.parametrize("chord_type", ["major", "minor"])
@pytest.mark.parametrize("root", range(12))
def test_recognizes_every_major_and_minor_chord(root, chord_type):
    x = synthesize_chord(48 + root, chord_type)
    freqs, _, mags = stft(x)
    frame_labels, _ = recognize_chords(chromagram(freqs, mags))
    expected = chord_label(root, chord_type)
    assert all(label == expected for label in frame_labels)


def test_silence_is_labelled_no_chord():
    freqs, _, mags = stft(np.zeros(8192))
    frame_labels, _ = recognize_chords(chromagram(freqs, mags))
    assert set(frame_labels) == {"N"}


def test_smoothing_keeps_shape_and_constant_rows():
    chroma = np.tile(np.linspace(0, 1, 12)[:, None], (1, 20))
    smoothed = smooth_chroma(chroma, width=5)
    assert smoothed.shape == chroma.shape
    assert np.allclose(smoothed, chroma)


def test_smoothing_reduces_an_isolated_spike():
    chroma = np.zeros((12, 11))
    chroma[0, 5] = 1
    assert smooth_chroma(chroma, width=5)[0, 5] == pytest.approx(0.2)
    
