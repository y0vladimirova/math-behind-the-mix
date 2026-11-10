"""Tuning, chords and consonance.

This module connects musical notes to frequencies and measures how
consonant combinations of notes are:

- conversions between frequency, MIDI note number and pitch class
- interval sizes in cents; just intonation vs. 12-tone equal temperament
- chord construction from stacked intervals
- sensory dissonance (Plomp-Levelt curve, Sethares parametrisation)
- chromagrams: folding a spectrogram into the 12 pitch classes

Chord recognition is added in the next commit.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np

from djmath.constants import A4_FREQUENCY, PITCH_CLASSES

A4_MIDI: int = 69
"""MIDI note number of A4, the tuning reference."""

JUST_RATIOS: dict[int, Fraction] = {
    0: Fraction(1, 1),     # unison
    1: Fraction(16, 15),   # minor second
    2: Fraction(9, 8),     # major second
    3: Fraction(6, 5),     # minor third
    4: Fraction(5, 4),     # major third
    5: Fraction(4, 3),     # perfect fourth
    6: Fraction(45, 32),   # tritone
    7: Fraction(3, 2),     # perfect fifth
    8: Fraction(8, 5),     # minor sixth
    9: Fraction(5, 3),     # major sixth
    10: Fraction(9, 5),    # minor seventh
    11: Fraction(15, 8),   # major seventh
    12: Fraction(2, 1),    # octave
}
"""Just-intonation frequency ratio for each interval, keyed by semitones."""

CHORD_TYPES: dict[str, tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
}
"""Triads as semitone offsets from the root (stacked major/minor thirds)."""

def midi_to_frequency(midi: float, a4: float = A4_FREQUENCY) -> float:
    """Frequency in Hz of a MIDI note: f = a4 * 2^((m - 69) / 12)."""
    return a4 * 2 ** ((midi - A4_MIDI) / 12)


def frequency_to_midi(frequency: float, a4: float = A4_FREQUENCY) -> float:
    """(Fractional) MIDI note number of a frequency: m = 69 + 12 log2(f / a4)."""
    return A4_MIDI + 12 * np.log2(frequency / a4)

def note_name(frequency: float, a4: float = A4_FREQUENCY) -> str:
    """Name of the pitch class nearest to a frequency, e.g. 440 -> 'A'."""
    midi = int(round(frequency_to_midi(frequency, a4)))
    return PITCH_CLASSES[midi % 12]

def cents(ratio: float) -> float:
    """Size of a frequency ratio in cents: 1200 log2(ratio).

    An octave (ratio 2) is 1200 cents; an equal-tempered semitone is 100.
    """
    return 1200 * np.log2(float(ratio))

def equal_tempered_ratio(semitones: float) -> float:
    """Frequency ratio of an interval in 12-tone equal temperament: 2^(s/12)."""
    return 2 ** (semitones / 12)

def chord_frequencies(
    root: float, chord_type: str = "major", tuning: str = "equal"
) -> np.ndarray:
    """Frequencies of a triad built on `root` (Hz).

    Parameters
    ----------
    root : float
        Frequency of the chord's root note in Hz.
    chord_type : str
        One of the keys of CHORD_TYPES.
    tuning : str
        "equal" for 12-tone equal temperament, "just" for just intonation.
    """
    if chord_type not in CHORD_TYPES:
        raise ValueError(f"Unknown chord type {chord_type!r}.")
    offsets = CHORD_TYPES[chord_type]
    if tuning == "equal":
        ratios = [equal_tempered_ratio(s) for s in offsets]
    elif tuning == "just":
        ratios = [float(JUST_RATIOS[s]) for s in offsets]
    else:
        raise ValueError("tuning must be 'equal' or 'just'.")
    return root * np.array(ratios)

def harmonic_spectrum(
    fundamental: float, n_harmonics: int = 6, decay: float = 0.88
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies k*f0 and amplitudes decay^(k-1) of an idealised tone."""
    k = np.arange(1, n_harmonics + 1)
    return fundamental * k, decay ** (k - 1)

# Constants of the Plomp-Levelt curve as fitted by Sethares (1993).
_D_STAR, _S1, _S2, _B1, _B2 = 0.24, 0.0207, 18.96, 3.51, 5.75

def pair_dissonance(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    """Sensory dissonance between two pure tones (Sethares' model).

    d = min(a1, a2) * (exp(-b1 s |df|) - exp(-b2 s |df|)),
    s = d* / (s1 * min(f1, f2) + s2)

    It is zero for identical frequencies, peaks when the frequencies are
    roughly a quarter of a critical band apart, and fades for wide intervals.
    Sethares multiplies this by a constant 5; we omit it, since only
    relative dissonance values matter here.
    """
    s = _D_STAR / (_S1 * min(f1, f2) + _S2)
    diff = abs(f2 - f1)
    return min(a1, a2) * (np.exp(-_B1 * s * diff) - np.exp(-_B2 * s * diff))

def dissonance(frequencies, amplitudes) -> float:
    """Total sensory dissonance of a set of partials: sum over all pairs."""
    partials = list(zip(frequencies, amplitudes))
    return sum(
        pair_dissonance(f1, f2, a1, a2)
        for (f1, a1), (f2, a2) in combinations(partials, 2)
    )

def chord_dissonance(fundamentals, n_harmonics: int = 6) -> float:
    """Dissonance of several harmonic tones sounding together."""
    freqs, amps = [], []
    for f0 in fundamentals:
        f, a = harmonic_spectrum(f0, n_harmonics)
        freqs.extend(f)
        amps.extend(a)
    return dissonance(freqs, amps)

def dissonance_curve(
    fundamental: float, ratios: np.ndarray, n_harmonics: int = 6
) -> np.ndarray:
    """Dissonance of two harmonic tones at f0 and r*f0, for each ratio r."""
    return np.array(
        [chord_dissonance([fundamental, r * fundamental], n_harmonics) for r in ratios]
    )

def pitch_class_of_bins(
    frequencies: np.ndarray,
    fmin: float = 55.0,
    fmax: float = 5000.0,
    a4: float = A4_FREQUENCY,
) -> np.ndarray:
    """Pitch class (0-11) of each frequency bin, or -1 outside [fmin, fmax].

    Each bin is assigned to the nearest equal-tempered note, then folded
    into one octave with modulo 12. Bins below fmin (including 0 Hz) or
    above fmax are ignored: they contain little pitch information.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    classes = np.full(frequencies.shape, -1)
    valid = (frequencies >= fmin) & (frequencies <= fmax)
    midi = np.round(A4_MIDI + 12 * np.log2(frequencies[valid] / a4))
    classes[valid] = midi.astype(int) % 12
    return classes


def chromagram(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    fmin: float = 55.0,
    fmax: float = 5000.0,
) -> np.ndarray:
    """Fold a magnitude spectrogram into a 12 x n_frames chromagram.

    The energy (squared magnitude) of all bins with the same pitch class is
    summed, and each frame is normalised so that its largest entry is 1.
    Silent frames stay all-zero.
    """
    classes = pitch_class_of_bins(frequencies, fmin, fmax)
    energy = np.asarray(magnitudes) ** 2
    chroma = np.zeros((12, energy.shape[1]))
    for pc in range(12):
        chroma[pc] = energy[classes == pc].sum(axis=0)

    peak = chroma.max(axis=0)
    return np.divide(chroma, peak, out=np.zeros_like(chroma), where=peak > 0)




