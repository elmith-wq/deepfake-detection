"""Pool out-of-fold predictions from all folds and report metrics."""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("outputs/val_predictions_fold*.csv")))
vid = df.groupby("video").agg(target=("target", "first"), prob=("prob", "mean")).reset_index()
pred = (vid.prob >= 0.5).astype(int)
m = {
    "videos": len(vid),
    "real_videos": int((vid.target == 0).sum()),
    "video_logloss": round(log_loss(vid.target, vid.prob), 4),
    "constant_baseline_logloss": round(log_loss(vid.target, np.full(len(vid), vid.target.mean())), 4),
    "video_auc": round(roc_auc_score(vid.target, vid.prob), 4),
    "real_recall": round(((pred == 0) & (vid.target == 0)).sum() / (vid.target == 0).sum(), 4),
    "fake_recall": round(((pred == 1) & (vid.target == 1)).sum() / (vid.target == 1).sum(), 4),
}
print(json.dumps(m, indent=2))
open("results/metrics_cv.json", "w").write(json.dumps(m, indent=2))