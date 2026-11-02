"""Download the DFDC training sample videos listed in metadata.json.

Kaggle serves each video as a .zip, so we unzip it and delete the archive.
Safe to stop and rerun: finished videos are skipped, and partial or
corrupt archives are re-downloaded.
"""
import json
import subprocess
import zipfile
from pathlib import Path

COMP = "deepfake-detection-challenge"
DATA_DIR = Path(r"C:\data\dfdc")
VIDEO_DIR = DATA_DIR / "train_sample_videos"
META = VIDEO_DIR / "metadata.json"


def unzip_and_remove(zip_path: Path) -> bool:
    """Extract a downloaded archive. Returns False if it was corrupt."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(VIDEO_DIR)
    except zipfile.BadZipFile:
        zip_path.unlink()  # partial download, will be fetched again
        return False
    zip_path.unlink()
    return True


def main():
    with open(META) as f:
        names = sorted(json.load(f).keys())

    for i, name in enumerate(names, 1):
        video = VIDEO_DIR / name
        zip_path = VIDEO_DIR / f"{name}.zip"

        if video.exists():
            continue  # already done
        if zip_path.exists() and unzip_and_remove(zip_path):
            continue  # zip was downloaded earlier, now extracted

        print(f"[{i}/{len(names)}] {name}")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMP,
             "-f", f"train_sample_videos/{name}", "-p", str(DATA_DIR)],
            check=False,
        )
        if zip_path.exists():
            unzip_and_remove(zip_path)


if __name__ == "__main__":
    main()