"""djmath: the mathematics behind DJing, implemented from first principles.

Modules
-------
signals  : sound synthesis, DFT/FFT, short-time Fourier transform
harmony  : tuning systems, consonance, chromagrams
key      : key detection and harmonic (Camelot) distance
tempo    : onset detection and tempo estimation
mixing   : crossfades and time-stretching
setlist  : track graph and set-ordering algorithms
"""

from djmath.constants import A4_FREQUENCY, PITCH_CLASSES, SAMPLE_RATE

__version__ = "0.1.0"
__all__ = ["A4_FREQUENCY", "PITCH_CLASSES", "SAMPLE_RATE"]
