"""Download datasets into the git-ignored data/ folder.

Usage:
    python scripts/download_data.py                              # create folders only
    python scripts/download_data.py giantsteps-key               # annotations + 100 tracks
    python scripts/download_data.py giantsteps-key --limit 604   # all tracks (~850 MB)

GiantSteps Key (Knees et al., ISMIR 2015): 604 two-minute previews of
electronic dance music with key annotations. The audio is only stored
locally for research use and is never committed to the repository.
"""

import argparse
import hashlib
import io
import random
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUBFOLDERS = ("synthetic", "giantsteps_key", "giantsteps_tempo", "fma")

GIANTSTEPS_KEY_REPO = (
    "https://github.com/GiantSteps/giantsteps-key-dataset/archive/refs/heads/master.zip"
)
GIANTSTEPS_AUDIO_URLS = (
    "https://www.cp.jku.at/datasets/giantsteps/backup/",
    "https://geo-samples.beatport.com/lofi/",
)


def make_folders() -> None:
    for name in SUBFOLDERS:
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)
    print(f"Data folders ready in {DATA_DIR}")


def fetch(url: str, timeout: int = 60) -> bytes:
    """Download the content of a URL."""
    with urlopen(url, timeout=timeout) as response:
        return response.read()


def md5_of(data: bytes) -> str:
    """MD5 checksum, used to verify that a download is complete and unchanged."""
    return hashlib.md5(data).hexdigest()


def download_giantsteps_annotations(target: Path) -> None:
    """Extract the key annotations and checksums from the dataset repository."""
    print("Downloading GiantSteps Key annotations ...")
    archive = zipfile.ZipFile(io.BytesIO(fetch(GIANTSTEPS_KEY_REPO)))
    for member in archive.namelist():
        # Members look like "giantsteps-key-dataset-master/annotations/key/x.key"
        parts = member.split("/", 1)
        if len(parts) < 2 or member.endswith("/"):
            continue
        relative = parts[1]
        if relative.startswith(("annotations/key/", "md5/")):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
    n_keys = len(list((target / "annotations" / "key").glob("*.key")))
    print(f"  {n_keys} key annotations saved.")


def download_giantsteps_audio(target: Path, limit: int, seed: int = 0) -> None:
    """Download `limit` randomly chosen previews, verifying each MD5 checksum."""
    md5_files = sorted((target / "md5").glob("*.md5"))
    if limit < len(md5_files):
        md5_files = sorted(random.Random(seed).sample(md5_files, limit))
    audio_dir = target / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = 0, 0, []
    for i, md5_file in enumerate(md5_files, start=1):
        name = md5_file.stem + ".mp3"
        destination = audio_dir / name
        expected = md5_file.read_text().strip()

        if destination.exists() and md5_of(destination.read_bytes()) == expected:
            skipped += 1
            continue

        for base_url in GIANTSTEPS_AUDIO_URLS:
            try:
                data = fetch(base_url + name)
            except (URLError, OSError):
                continue
            if md5_of(data) == expected:
                destination.write_bytes(data)
                downloaded += 1
                break
        else:
            failed.append(name)
        print(f"  [{i}/{len(md5_files)}] {name}", end="\r")

    print(f"\n  Downloaded {downloaded}, already present {skipped}, failed {len(failed)}.")
    if failed:
        print("  Failed files:", ", ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", choices=["giantsteps-key"],
                        help="dataset to download (omit to only create folders)")
    parser.add_argument("--limit", type=int, default=100,
                        help="number of audio tracks to download (default: 100)")
    args = parser.parse_args()

    make_folders()
    if args.dataset == "giantsteps-key":
        target = DATA_DIR / "giantsteps_key"
        download_giantsteps_annotations(target)
        download_giantsteps_audio(target, args.limit)


if __name__ == "__main__":
    main()
