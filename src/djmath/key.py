"""Musical key detection.

The key of a piece is estimated with the Krumhansl-Schmuckler algorithm:
the average chroma vector of the piece is correlated with 24 key profiles
(12 major, 12 minor), and the key with the highest correlation wins.

Keys are compared with the circle of fifths: each key gets a position on
the Camelot wheel (Z12 x {A, B}), and the harmonic distance between two
keys is their shortest-path distance on the wheel. Key estimates are
scored with the standard MIREX weighting.
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

SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),   # natural minor
}
"""Scale degrees of each mode as semitone offsets from the tonic."""

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

def parse_key(label: str) -> tuple[int, str]:
    """Inverse of key_label: 'C# major' -> (1, 'major')."""
    name, mode = label.split()
    if name not in PITCH_CLASSES or mode not in MODES:
        raise ValueError(f"Cannot parse key {label!r}.")
    return PITCH_CLASSES.index(name), mode

def scale_pitch_classes(label: str) -> set[int]:
    """The seven pitch classes of a key's scale, e.g. 'C major' -> {0, 2, 4, 5, 7, 9, 11}."""
    tonic, mode = parse_key(label)
    return {(tonic + step) % 12 for step in SCALES[mode]}

def camelot(label: str) -> tuple[int, str]:
    """Position of a key on the Camelot wheel, e.g. 'C major' -> (8, 'B').

    Multiplying a pitch class by 7 (mod 12) gives its position on the circle
    of fifths. Major keys use letter B; a minor key shares the number of its
    relative major (tonic + 3 semitones) and uses letter A. The offset is
    chosen so that C major is 8B, as in the DJ convention.
    """
    tonic, mode = parse_key(label)
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    number = (7 * major_tonic + 7) % 12 + 1
    return number, "B" if mode == "major" else "A"

def camelot_code(label: str) -> str:
    """Camelot code as text, e.g. 'A minor' -> '8A'."""
    number, letter = camelot(label)
    return f"{number}{letter}"

def harmonic_distance(label_a: str, label_b: str) -> int:
    """Shortest-path distance between two keys on the Camelot wheel.

    One step changes the number by +-1 (a fifth up or down) or switches
    between A and B with the same number (relative major/minor):

        d = min(|n1 - n2|, 12 - |n1 - n2|) + [letters differ]
    """
    n1, l1 = camelot(label_a)
    n2, l2 = camelot(label_b)
    diff = abs(n1 - n2)
    return min(diff, 12 - diff) + int(l1 != l2)

def key_score(reference: str, estimate: str) -> float:
    """MIREX weighted score of a key estimate.

    1.0 correct, 0.5 a perfect fifth above (same mode), 0.3 relative key,
    0.2 parallel key (same tonic, other mode), 0.0 otherwise.
    """
    ref_tonic, ref_mode = parse_key(reference)
    est_tonic, est_mode = parse_key(estimate)
    interval = (est_tonic - ref_tonic) % 12
    if est_mode == ref_mode:
        if interval == 0:
            return 1.0
        if interval == 7:
            return 0.5
        return 0.0
    if ref_mode == "major" and interval == 9:   # e.g. C major -> A minor
        return 0.3
    if ref_mode == "minor" and interval == 3:   # e.g. A minor -> C major
        return 0.3
    if interval == 0:
        return 0.2
    return 0.0
