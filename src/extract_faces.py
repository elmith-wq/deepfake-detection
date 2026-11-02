"""Extract face crops from DFDC videos.

For each video: sample N evenly spaced frames, detect the most confident
face with MTCNN, save a 224x224 crop, and log it in faces.csv.
Resumable: videos that already have crops are skipped.
"""
import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(r"C:\data\dfdc")
VIDEO_DIR = DATA_DIR / "train_sample_videos"
FACE_DIR = DATA_DIR / "faces"
CSV_PATH = DATA_DIR / "faces.csv"
DETECT_SCALE = 0.5   # detect on half-size frames for speed
MARGIN = 0.3         # extra context around the face box
SIZE = 224


def read_frames(path, n_frames):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    frames = []
    for idx in (int(x) for x in torch.linspace(0, total - 1, n_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def crop_face(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return Image.fromarray(frame[y1:y2, x1:x2]).resize((SIZE, SIZE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N videos (for testing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(device=device, keep_all=False, post_process=False)

    with open(VIDEO_DIR / "metadata.json") as f:
        meta = json.load(f)
    names = sorted(meta)[: args.limit]
    FACE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in tqdm(names):
        stem = Path(name).stem
        if list(FACE_DIR.glob(f"{stem}_*.jpg")):
            continue  # already processed
        path = VIDEO_DIR / name
        if not path.exists():
            continue
        frames = read_frames(path, args.frames)
        if not frames:
            continue
        small = [Image.fromarray(cv2.resize(
            f, None, fx=DETECT_SCALE, fy=DETECT_SCALE)) for _, f in frames]
        boxes, _ = mtcnn.detect(small)
        for (idx, frame), box in zip(frames, boxes):
            if box is None:
                continue  # no face found in this frame
            areas = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            face = crop_face(frame, box[areas.argmax()] / DETECT_SCALE)
            out = FACE_DIR / f"{stem}_{idx:04d}.jpg"
            face.save(out, quality=95)

    # Rebuild the CSV from whatever crops exist on disk
    for p in sorted(FACE_DIR.glob("*.jpg")):
        video = p.stem.rsplit("_", 1)[0] + ".mp4"
        if video in meta:
            rows.append({"path": str(p), "video": video,
                         "label": meta[video]["label"]})
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"{len(rows)} face crops saved. CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()