"""Onset detection and tempo estimation.

The tempo of a track is estimated in two steps:

1. an onset strength (novelty) curve from the spectral flux: how much the
   log-compressed spectrum increases from one frame to the next;
2. the autocorrelation of that curve: a beat repeats with a fixed period,
   so the autocorrelation peaks at the beat period. A log-normal tempo
   prior resolves the ambiguity between multiples of the period.
"""

import numpy as np

from djmath.constants import SAMPLE_RATE
from djmath.signals import stft


def onset_strength(
    x: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 256,
    sample_rate: int = SAMPLE_RATE,
    compression: float = 100.0,
    smoothing: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral-flux novelty curve.

    novelty[m] = sum_k max(0, Y[k, m] - Y[k, m-1]),  Y = log(1 + gamma |X|)

    Only increases in energy count (onsets), and the logarithm makes quiet
    and loud events comparable. A local average is subtracted, negative
    values are clipped, and the

    curve is smoothed with a short Hann window
    of `smoothing` frames, so that each onset peak is a few frames wide
    instead of one. Finally the curve is scaled to a maximum of 1.

    Returns the time of each frame's centre (s) and the novelty curve.
    """
    _, starts, mags = stft(x, frame_length, hop_length, sample_rate)
    times = starts + frame_length / (2 * sample_rate)
    log_mags = np.log1p(compression * mags)
    flux = np.maximum(0.0, np.diff(log_mags, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])   # frame 0 has no predecessor

    # Subtract a local average over about half a second
    width = max(1, int(round(0.5 * sample_rate / hop_length)))
    local_mean = np.convolve(flux, np.ones(width) / width, mode="same")
    novelty = np.maximum(0.0, flux - local_mean)

    if smoothing > 1:
        kernel = np.hanning(smoothing + 2)[1:-1]   # drop the zero end points
        novelty = np.convolve(novelty, kernel / kernel.sum(), mode="same")

    peak = novelty.max()
    return times, novelty / peak if peak > 0 else novelty

def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation r[l] = sum_n x[n] x[n+l] / sum_n x[n]^2.

    Computed with the FFT (Wiener-Khinchin theorem): the autocorrelation is
    the inverse transform of the power spectrum |X|^2. Zero-padding to twice
    the length prevents the circular wrap-around of the DFT.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(np.abs(spectrum) ** 2)[:n]
    return r / r[0] if r[0] > 0 else r

def lag_to_bpm(lag: float, frame_rate: float) -> float:
    """Tempo in BPM for a beat period of `lag` novelty frames."""
    return 60 * frame_rate / lag

def tempo_prior(bpm, center: float = 120.0, width: float = 1.0) -> np.ndarray:
    """Log-normal preference for typical tempos.

    w(bpm) = exp(-0.5 * (log2(bpm / center) / width)^2)

    The distance is measured in octaves (doublings), so 60 and 240 BPM are
    penalised equally around a centre of 120 BPM.
    """
    octaves = np.log2(np.asarray(bpm, dtype=float) / center)
    return np.exp(-0.5 * (octaves / width) ** 2)

def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample position of a peak at index i, from a parabola through 3 points."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    left, centre, right = y[i - 1], y[i], y[i + 1]
    denominator = left - 2 * centre + right
    if denominator == 0:
        return float(i)
    return i + 0.5 * (left - right) / denominator

def estimate_tempo(
    x: np.ndarray,
    bpm_range: tuple[float, float] = (60.0, 200.0),
    use_prior: bool = True,
    refine: bool = True,
    hop_length: int = 256,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Estimate the tempo of a signal in BPM.

    Searches the autocorrelation of the onset curve for the strongest peak
    whose lag lies in `bpm_range`, optionally weighted by the tempo prior,
    and (if `refine`) locates the peak between frames with parabolic
    interpolation.
    """
    frame_rate = sample_rate / hop_length
    _, novelty = onset_strength(x, hop_length=hop_length, sample_rate=sample_rate)
    r = autocorrelation(novelty)

    min_lag = int(np.floor(lag_to_bpm(bpm_range[1], frame_rate)))
    max_lag = int(np.ceil(lag_to_bpm(bpm_range[0], frame_rate)))
    lags = np.arange(max(min_lag, 1), min(max_lag, len(r) - 2) + 1)
    if len(lags) == 0:
        raise ValueError("The signal is too short for the requested tempo range.")

    scores = r[lags]
    if use_prior:
        scores = scores * tempo_prior(lag_to_bpm(lags, frame_rate))
    best_lag = lags[int(np.argmax(scores))]
    lag = _parabolic_peak(r, best_lag) if refine else float(best_lag)
    return lag_to_bpm(lag, frame_rate)
