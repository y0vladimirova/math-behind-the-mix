"""Shared constants used across the project."""

SAMPLE_RATE: int = 22_050
"""Default sample rate in Hz. Half of CD quality; enough for music analysis
(Nyquist frequency 11 025 Hz) while keeping computations fast."""

A4_FREQUENCY: float = 440.0
"""Reference tuning pitch in Hz (ISO 16)."""

PITCH_CLASSES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)
"""The 12 pitch classes of 12-tone equal temperament, indexed 0-11."""
