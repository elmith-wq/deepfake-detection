"""Evaluate saved validation predictions: metrics, plots, worst errors."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             confusion_matrix, log_loss, roc_auc_score)

PRED = Path("outputs/val_predictions.csv")
OUT = Path("results")
BASE_RATE = 0.80  # fake rate in the training set


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(PRED)
    vid = (df.groupby("video")
             .agg(target=("target", "first"), prob=("prob", "mean"),
                  n_crops=("prob", "size"))
             .reset_index())

    pred = (vid.prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(vid.target, pred).ravel()
    metrics = {
        "videos": len(vid),
        "real_videos": int((vid.target == 0).sum()),
        "video_logloss": round(log_loss(vid.target, vid.prob), 4),
        "constant_baseline_logloss": round(
            log_loss(vid.target, np.full(len(vid), BASE_RATE)), 4),
        "video_auc": round(roc_auc_score(vid.target, vid.prob), 4),
        "accuracy_at_0.5": round((tn + tp) / len(vid), 4),
        "real_recall": round(tn / (tn + fp), 4),
        "fake_recall": round(tp / (tp + fn), 4),
    }
    print(json.dumps(metrics, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    RocCurveDisplay.from_predictions(vid.target, vid.prob)
    plt.title("Validation ROC (per video)")
    plt.savefig(OUT / "roc.png", dpi=150, bbox_inches="tight")
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        vid.target, pred, display_labels=["REAL", "FAKE"])
    plt.title("Confusion matrix (threshold 0.5)")
    plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.hist([vid.prob[vid.target == 0], vid.prob[vid.target == 1]],
             bins=10, range=(0, 1), label=["REAL", "FAKE"])
    plt.xlabel("Predicted probability of FAKE (video mean)")
    plt.ylabel("Videos")
    plt.legend()
    plt.savefig(OUT / "prob_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    vid["error"] = (vid.target - vid.prob).abs()
    worst = vid.sort_values("error", ascending=False).head(10)
    worst.to_csv(OUT / "worst_videos.csv", index=False)
    print("\nWorst-predicted videos:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()