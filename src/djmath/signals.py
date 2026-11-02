"""Sound as mathematics: synthesis and frequency analysis.

This module builds test signals with known frequency content and
implements the discrete Fourier transform (DFT) in two ways:

- ``dft``: straight from the definition, O(N^2) operations
- ``fft``: the recursive radix-2 Cooley-Tukey algorithm, O(N log N)

Both are verified against ``numpy.fft.fft`` in the tests.
"""

import numpy as np

from djmath.constants import SAMPLE_RATE

def time_axis(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return the sample times t_n = n / sample_rate, in seconds.

    Parameters
    ----------
    duration : float
        Length of the signal in seconds.
    sample_rate : int
        Number of samples per second (Hz).
    """
    n_samples = int(round(duration * sample_rate))
    return np.arange(n_samples) / sample_rate

def sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Sample the pure tone x(t) = A sin(2 pi f t + phi)."""
    t = time_axis(duration, sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def harmonic_tone(
    fundamental: float,
    duration: float,
    n_harmonics: int = 10,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Sum of harmonics k * f0 with amplitudes 1/k (a sawtooth approximation).

    Harmonics at or above the Nyquist frequency are skipped, because they
    cannot be represented at this sample rate and would alias.
    The result is normalised to a peak amplitude of 1.
    """
    nyquist = sample_rate / 2
    if fundamental >= nyquist:
        raise ValueError("The fundamental must be below the Nyquist frequency.")

    t = time_axis(duration, sample_rate)
    tone = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        if k * fundamental >= nyquist:
            break
        tone += np.sin(2 * np.pi * k * fundamental * t) / k
    return tone / np.max(np.abs(tone))

def dft(x: np.ndarray) -> np.ndarray:
    """Discrete Fourier transform computed directly from its definition.

    X[k] = sum_{n=0}^{N-1} x[n] exp(-2 pi i k n / N),  k = 0, ..., N-1

    Each X[k] is the inner product of the signal with a complex sinusoid
    that completes k cycles over the N samples. Cost: O(N^2).
    """
    x = np.asarray(x, dtype=complex)
    n_samples = x.shape[0]
    n = np.arange(n_samples)
    spectrum = np.empty(n_samples, dtype=complex)
    for k in range(n_samples):
        spectrum[k] = np.sum(x * np.exp(-2j * np.pi * k * n / n_samples))
    return spectrum

def _is_power_of_two(n: int) -> bool:
    """True if n is 1, 2, 4, 8, ... (a power of two has a single 1 bit)."""
    return n > 0 and (n & (n - 1)) == 0

def fft(x: np.ndarray) -> np.ndarray:
    """Recursive radix-2 Cooley-Tukey fast Fourier transform.

    Splits the signal into even- and odd-indexed samples, transforms each
    half recursively, and combines them with "twiddle factors":

        X[k]       = E[k] + w^k O[k]
        X[k + N/2] = E[k] - w^k O[k],   w = exp(-2 pi i / N)

    Cost: O(N log N). The length of x must be a power of two.
    """
    x = np.asarray(x, dtype=complex)
    n_samples = x.shape[0]
    if not _is_power_of_two(n_samples):
        raise ValueError(f"Length must be a power of two, got {n_samples}.")
    if n_samples == 1:
        return x.copy()

    even = fft(x[0::2])
    odd = fft(x[1::2])
    twiddled = np.exp(-2j * np.pi * np.arange(n_samples // 2) / n_samples) * odd
    return np.concatenate([even + twiddled, even - twiddled])

def magnitude_spectrum(
    x: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and amplitudes of the non-negative-frequency spectrum.

    For a real signal the spectrum is symmetric, so only bins 0..N/2 carry
    information. Scaling by 2/N makes a sine of amplitude A show a peak of
    height A (except at 0 Hz and at the Nyquist frequency).
    Uses numpy's FFT for speed; our own implementations are validated
    against it in the tests.
    """
    n_samples = len(x)
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
    amplitudes = 2 * np.abs(spectrum) / n_samples
    return frequencies, amplitudes

