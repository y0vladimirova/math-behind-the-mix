"""Musical key detection.

The key of a piece is estimated with the Krumhansl-Schmuckler algorithm:
the average chroma vector of the piece is correlated with 24 key profiles
(12 major, 12 minor), and the key with the highest correlation wins.

Harmonic distance between keys (circle of fifths, Camelot wheel) is added
in the next commit.
"""

import numpy as np

from djmath.constants import PITCH_CLASSES

MAJOR_PROFILE: np.ndarray = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
"""Krumhansl-Kessler probe-tone ratings for a major key, starting at the tonic."""

MINOR_PROFILE: np.ndarray = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
"""Krumhansl-Kessler probe-tone ratings for a minor key, starting at the tonic."""

MODES: tuple[str, str] = ("major", "minor")

def key_label(tonic: int, mode: str) -> str:
    """Readable key name, e.g. (0, 'major') -> 'C major'."""
    if mode not in MODES:
        raise ValueError(f"mode must be 'major' or 'minor', got {mode!r}.")
    return f"{PITCH_CLASSES[tonic % 12]} {mode}"

def key_profiles() -> tuple[list[str], np.ndarray]:
    """Profiles of all 24 keys (shape 24 x 12) and their labels.

    The profile of the key with tonic t is the C profile cyclically shifted
    by t positions, so that its first value lands on pitch class t.
    """
    labels, rows = [], []
    for mode, profile in zip(MODES, (MAJOR_PROFILE, MINOR_PROFILE)):
        for tonic in range(12):
            labels.append(key_label(tonic, mode))
            rows.append(np.roll(profile, tonic))
    return labels, np.array(rows)

def correlations(vector: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    """Pearson correlation of `vector` (length 12) with each row of `profiles`.

    Pearson correlation is the cosine similarity of the two vectors after
    subtracting their means, so it ignores both scale and constant offset.
    """
    v = np.asarray(vector, dtype=float)
    v = v - v.mean()
    p = profiles - profiles.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(v)
    if denominator.min() == 0:
        raise ValueError("Correlation is undefined for a constant vector.")
    return p @ v / denominator

def estimate_key(chroma: np.ndarray) -> tuple[str, np.ndarray]:
    """Estimate the key of a chromagram (12 x n_frames) or chroma vector.

    The frames are averaged into one 12-dimensional vector describing how
    much each pitch class is used, which is then correlated with the
    24 key profiles. Returns the best key and all 24 correlations, in the
    order of `key_profiles()`.
    """
    chroma = np.asarray(chroma, dtype=float)
    vector = chroma.mean(axis=1) if chroma.ndim == 2 else chroma
    labels, profiles = key_profiles()
    scores = correlations(vector, profiles)
    return labels[int(np.argmax(scores))], scores







