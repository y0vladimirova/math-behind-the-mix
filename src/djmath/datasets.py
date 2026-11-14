"""Access to annotated datasets stored in the git-ignored data/ folder.

The files are downloaded by ``scripts/download_data.py``; this module only
reads what is already on disk.
"""

from pathlib import Path

from djmath.key import normalize_key_label

DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"
"""The project's data/ folder (src/djmath/datasets.py -> project root)."""

GIANTSTEPS_KEY_DIR: Path = DATA_DIR / "giantsteps_key"


def load_giantsteps_key(root: Path = GIANTSTEPS_KEY_DIR) -> list[tuple[Path, str]]:
    """Downloaded GiantSteps Key tracks with their annotated keys.

    Returns (audio path, key label) pairs, sorted by file name, for every
    annotation whose audio file is present. Keys are normalised to this
    project's spelling, e.g. 'Eb minor' -> 'D# minor'.
    """
    annotation_dir = Path(root) / "annotations" / "key"
    audio_dir = Path(root) / "audio"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"No annotations in {annotation_dir}. "
            "Run: python scripts/download_data.py giantsteps-key"
        )

    tracks = []
    for key_file in sorted(annotation_dir.glob("*.key")):
        audio_file = audio_dir / (key_file.stem + ".mp3")
        if audio_file.exists():
            tracks.append((audio_file, normalize_key_label(key_file.read_text())))
    return tracks
