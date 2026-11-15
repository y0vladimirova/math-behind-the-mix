"""Tests for drum synthesis, onset detection and tempo estimation."""

import numpy as np
import pytest

from djmath.constants import SAMPLE_RATE
from djmath.signals import drum_loop, kick_drum
from djmath.tempo import (
    autocorrelation,
    estimate_tempo,
    lag_to_bpm,
    onset_strength,
    tempo_prior,
)


def test_kick_drum_decays():
    kick = kick_drum()
    first, last = np.abs(kick[:500]).max(), np.abs(kick[-500:]).max()
    assert last < 0.1 * first


def test_drum_loop_length():
    assert len(drum_loop(120, duration=2.0)) == 2 * SAMPLE_RATE


def test_autocorrelation_matches_direct_definition():
    x = np.random.default_rng(0).random(200)
    direct = np.correlate(x, x, mode="full")[len(x) - 1:]
    assert np.allclose(autocorrelation(x), direct / direct[0])


def test_autocorrelation_of_impulse_train_peaks_at_period():
    x = np.zeros(400)
    x[::25] = 1
    r = autocorrelation(x)
    assert r[0] == pytest.approx(1)
    assert np.argmax(r[1:60]) + 1 == 25


def test_lag_to_bpm():
    # At 100 frames per second, a period of 50 frames is 0.5 s = 120 BPM
    assert lag_to_bpm(50, frame_rate=100) == pytest.approx(120)


def test_tempo_prior_is_symmetric_in_octaves():
    assert tempo_prior(120) == pytest.approx(1)
    assert tempo_prior(60) == pytest.approx(tempo_prior(240))
    assert tempo_prior(100) > tempo_prior(200)


def test_onset_strength_peaks_at_every_drum_hit():
    bpm = 120
    times, novelty = onset_strength(drum_loop(bpm, duration=4.0))
    assert novelty.max() == pytest.approx(1)
    hit_times = np.arange(2, 14) * 0.5 * 60 / bpm    # kicks and hi-hats, skipping the edges
    for t in hit_times:
        near = np.abs(times - t) < 0.03
        assert novelty[near].max() > 0.2


@pytest.mark.parametrize("bpm", [80, 95, 110, 120, 128, 140, 150, 160])
def test_estimates_tempo_of_drum_loops(bpm):
    estimate = estimate_tempo(drum_loop(bpm, duration=20.0))
    assert estimate == pytest.approx(bpm, rel=0.01)


def test_prior_folds_fast_tempos_into_one_octave():
    # Above 120 * sqrt(2) ~ 170 BPM, half the tempo is closer to the prior centre
    assert estimate_tempo(drum_loop(180, duration=20.0)) == pytest.approx(90, rel=0.01)


def test_too_short_signal_is_rejected():
    with pytest.raises(ValueError):
        estimate_tempo(np.random.default_rng(0).random(4096))
