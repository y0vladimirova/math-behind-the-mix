"""Tests for the short-time Fourier transform and the chromagram."""

import numpy as np
import pytest

from djmath.harmony import chord_frequencies, chromagram, pitch_class_of_bins
from djmath.signals import hann_window, harmonic_tone, sine_wave, stft


def test_hann_window_starts_at_zero_and_peaks_in_the_middle():
    w = hann_window(8)
    assert w[0] == pytest.approx(0)
    assert w[4] == pytest.approx(1)


def test_stft_shapes():
    x = np.zeros(22_050)
    freqs, times, mags = stft(x, frame_length=2048, hop_length=512)
    assert freqs.shape == (1025,)
    assert mags.shape == (1025, len(times))
    assert len(times) == 1 + (22_050 - 2048) // 512


def test_stft_rejects_signal_shorter_than_a_frame():
    with pytest.raises(ValueError):
        stft(np.zeros(100), frame_length=2048)


def test_stft_of_sine_peaks_at_its_frequency_in_every_frame():
    x = sine_wave(1000, duration=1.0)
    freqs, _, mags = stft(x)
    peak_freqs = freqs[np.argmax(mags, axis=0)]
    resolution = freqs[1] - freqs[0]
    assert np.all(np.abs(peak_freqs - 1000) <= resolution)


def test_pitch_class_of_bins():
    classes = pitch_class_of_bins(np.array([0.0, 261.63, 440.0, 880.0, 10_000.0]))
    assert list(classes) == [-1, 0, 9, 9, -1]


def test_chromagram_of_a4_peaks_at_pitch_class_a():
    freqs, _, mags = stft(sine_wave(440, duration=1.0))
    chroma = chromagram(freqs, mags)
    assert chroma.shape[0] == 12
    assert np.all(np.argmax(chroma, axis=0) == 9)
    assert np.allclose(chroma.max(axis=0), 1)


def test_chromagram_of_c_major_chord_highlights_c_e_g():
    notes = chord_frequencies(261.63, "major")
    x = sum(harmonic_tone(f, duration=1.0, n_harmonics=4) for f in notes)
    freqs, _, mags = stft(x)
    mean_chroma = chromagram(freqs, mags).mean(axis=1)
    top_three = set(np.argsort(mean_chroma)[-3:])
    assert top_three == {0, 4, 7}


def test_chromagram_of_silence_is_zero():
    freqs, _, mags = stft(np.zeros(8192))
    assert np.all(chromagram(freqs, mags) == 0)
    
