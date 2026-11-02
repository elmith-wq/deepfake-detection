"""Group-aware train/val split and PyTorch dataset for face crops."""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

CSV_PATH = Path(r"C:\data\dfdc\faces.csv")
META_PATH = Path(r"C:\data\dfdc\train_sample_videos\metadata.json")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats


def make_splits(csv_path=CSV_PATH, fold=0, n_splits=5, seed=42):
    """Split by VIDEO GROUP: a real video and every fake made from it stay together."""
    df = pd.read_csv(csv_path)
    df["target"] = (df["label"] == "FAKE").astype(int)  # FAKE = 1
    with open(META_PATH) as f:
        meta = json.load(f)
    # reals have no "original", so they form their own group
    df["group"] = df["video"].map(lambda v: meta[v].get("original") or v)
    videos = df.groupby("video").agg(
        target=("target", "first"), group=("group", "first")).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = list(sgkf.split(videos, videos.target, videos.group))[fold]
    train_v = set(videos.video.iloc[tr_idx])
    val_v = set(videos.video.iloc[va_idx])
    train_df = df[df.video.isin(train_v)].reset_index(drop=True)
    val_df = df[df.video.isin(val_v)].reset_index(drop=True)
    return train_df, val_df


def get_transforms(train: bool):
    steps = [transforms.Resize((224, 224))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class FaceDataset(Dataset):
    def __init__(self, df, train: bool):
        self.df = df
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), torch.tensor(row["target"], dtype=torch.float32)


if __name__ == "__main__":
    train_df, val_df = make_splits()
    overlap = set(train_df["group"]) & set(val_df["group"])
    assert not overlap, "Leak: a source video and its fakes are split apart!"
    for name, d in [("train", train_df), ("val", val_df)]:
        print(f"{name}: {d['video'].nunique()} videos, {len(d)} crops, "
              f"{d['target'].mean():.1%} fake")
    x, y = FaceDataset(train_df, train=True)[0]
    print("sample:", tuple(x.shape), y.item())