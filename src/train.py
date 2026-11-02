"""Fine-tune EfficientNet-B0 on face crops; evaluate per crop and per video."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from dataset import FaceDataset, make_splits

OUT = Path("outputs")
EPS = 1e-4  # clip probabilities, as the competition did, to avoid infinite log loss


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def evaluate(df, probs):
    """Metrics per crop and per video (mean of a video's crop probabilities)."""
    df = df.assign(prob=np.clip(probs, EPS, 1 - EPS))
    vid = df.groupby("video").agg(target=("target", "first"), prob=("prob", "mean"))
    return {
        "crop_logloss": log_loss(df.target, df.prob),
        "video_logloss": log_loss(vid.target, vid.prob),
        "video_auc": roc_auc_score(vid.target, vid.prob),
    }, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    train_df, val_df = make_splits()

    train_dl = DataLoader(FaceDataset(train_df, train=True), args.batch_size,
                          shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(FaceDataset(val_df, train=False), args.batch_size,
                        num_workers=args.workers, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    base = train_df.target.mean()
    const = log_loss(val_df.target, np.full(len(val_df), base))
    print(f"Baseline (constant {base:.2f} prediction) crop log loss: {const:.3f}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        sched.step()

        metrics, scored = evaluate(val_df, predict(model, val_dl, device))
        print(f"epoch {epoch}: train_loss={total / len(train_df):.4f} "
              + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if metrics["video_logloss"] < best:
            best = metrics["video_logloss"]
            torch.save(model.state_dict(), OUT / "best.pth")
            scored.to_csv(OUT / "val_predictions.csv", index=False)
            print("  saved best model")


if __name__ == "__main__":
    main()