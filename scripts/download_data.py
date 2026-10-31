"""Prepare the local data/ folder.

Audio is never committed to the repository (see .gitignore). This script
creates the folder layout; dataset downloads are added in a later commit.

Usage:
    python scripts/download_data.py
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUBFOLDERS = ("synthetic", "giantsteps_key", "giantsteps_tempo", "fma")


def main() -> None:
    for name in SUBFOLDERS:
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)
    print(f"Data folders ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
