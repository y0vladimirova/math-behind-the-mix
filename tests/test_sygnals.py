"""Tests for djmath.signals: synthesis and Fourier transforms."""

import numpy as np
import pytest

from djmath.signals import (
    dft,
    fft,
    harmonic_tone,
    magnitude_spectrum,
    sine_wave,
    time_axis,
)

def test_time_axis_length_and_spacing():
    t = time_axis(0.5, sample_rate=1000)
    assert len(t) == 500
    assert t[1] - t[0] == pytest.approx(1 / 1000)
    

@pytest.mark.parametrize("n", [1, 2, 7, 64, 100])
def test_dft_matches_numpy(n):
    x = np.random.default_rng(0).standard_normal(n)
    assert np.allclose(dft(x), np.fft.fft(x))


@pytest.mark.parametrize("n", [1, 2, 8, 64, 1024])
def test_fft_matches_numpy(n):
    x = np.random.default_rng(0).standard_normal(n)
    assert np.allclose(fft(x), np.fft.fft(x))

def test_fft_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fft(np.zeros(100))

def test_sine_has_single_peak_at_its_frequency():
    x = sine_wave(440, duration=1.0, amplitude=0.5)
    freqs, amps = magnitude_spectrum(x)
    peak = np.argmax(amps)
    assert freqs[peak] == pytest.approx(440)
    assert amps[peak] == pytest.approx(0.5, abs=1e-6)

def test_harmonic_tone_amplitudes_fall_as_one_over_k():
    f0 = 220
    x = harmonic_tone(f0, duration=1.0, n_harmonics=5)
    freqs, amps = magnitude_spectrum(x)
    first = amps[np.argmin(np.abs(freqs - f0))]
    for k in range(2, 6):
        kth = amps[np.argmin(np.abs(freqs - k * f0))]
        assert kth / first == pytest.approx(1 / k, rel=1e-3)
