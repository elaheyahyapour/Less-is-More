from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score,precision_score,recall_score
import numpy as np
import pandas as pd

CLASS_ORDER = ["stop", "slow down", "proceed"]
YOLO_CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog'
}
def safety_weighted_f1(per_class_f1, weights=None):
    if weights is None:
        weights = {"stop": 0.6, "slow down": 0.3, "proceed": 0.1}
    return sum(per_class_f1.get(c, 0.0) * weights.get(c, 0.0) for c in CLASS_ORDER)

def expected_risk_cost(y_true, y_pred, cost=None):
    if cost is None:
        cost = {
            ("stop","proceed"): 10.0,
            ("stop","slow down"): 4.0,
            ("slow down","proceed"): 3.0,
            ("slow down","stop"): 2.0,
            ("proceed","stop"): 1.0,
            ("proceed","slow down"): 0.5,
        }
    total = 0.0
    for t, p in zip(y_true, y_pred):
        total += cost.get((t, p), 0.0) if t != p else 0.0
    return total / max(1, len(y_true))

def catastrophic_rates(y_true, y_pred):
    n = len(y_true)
    stop_idx = [i for i,t in enumerate(y_true) if t == "stop"]
    miss_stop = sum(1 for i in stop_idx if y_pred[i] == "proceed")
    per_stop_miss_rate = miss_stop / max(1, len(stop_idx))
    overall_cat_rate = miss_stop / max(1, n)
    near = sum(1 for t,p in zip(y_true, y_pred)
               if (t == "stop" and p == "slow down") or (t == "slow down" and p == "proceed"))
    near_rate = near / max(1, n)
    return {
        "missed_stop_per_stop": per_stop_miss_rate,
        "catastrophic_overall": overall_cat_rate,
        "near_catastrophic_overall": near_rate
    }

def compute_full_metrics(df, pred_col="predicted"):
    y_true = df["ground_truth"].tolist()
    y_pred = df[pred_col].tolist()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    per_class_f1 = {c: report.get(c, {}).get("f1-score", 0.0) for c in CLASS_ORDER}
    swf1 = safety_weighted_f1(per_class_f1)

    exp_cost = expected_risk_cost(y_true, y_pred)

    cats = catastrophic_rates(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_row_norm = (cm.T / np.maximum(1, cm.sum(axis=1))).T

    return {
        "acc": acc, "macro_f1": f1m, "macro_prec": prec, "macro_rec": rec,
        "per_class": {
            c: {
                "precision": report.get(c, {}).get("precision", 0.0),
                "recall": report.get(c, {}).get("recall", 0.0),
                "f1": report.get(c, {}).get("f1-score", 0.0),
                "support": int(report.get(c, {}).get("support", 0))
            } for c in CLASS_ORDER
        },
        "safety_weighted_f1": swf1,
        "expected_risk_cost": exp_cost,
        "catastrophe": cats,
        "cm_counts": cm,
        "cm_row_norm": cm_row_norm
    }

def bootstrap_ci(df, pred_col="predicted", B=1000, seed=123):
    rng = np.random.default_rng(seed)
    if "scene_id" in df.columns:
        groups = df.groupby("scene_id").indices
        scene_keys = list(groups.keys())
        accs, f1s = [], []
        for _ in range(B):
            samp_scenes = rng.choice(scene_keys, size=len(scene_keys), replace=True)
            idx = np.concatenate([groups[s] for s in samp_scenes])
            yt = df.loc[idx, "ground_truth"].to_numpy()
            yp = df.loc[idx, pred_col].to_numpy()
            accs.append(accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, labels=CLASS_ORDER, average="macro", zero_division=0))
    else:
        n = len(df); idx_all = np.arange(n)
        accs, f1s = [], []
        for _ in range(B):
            samp = rng.choice(idx_all, size=n, replace=True)
            yt = df["ground_truth"].to_numpy()[samp]
            yp = df[pred_col].to_numpy()[samp]
            accs.append(accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, labels=CLASS_ORDER, average="macro", zero_division=0))

    def pct(a, lo, hi): return float(np.percentile(a, lo)), float(np.percentile(a, hi))
    return {"acc_ci": pct(accs, 2.5, 97.5), "macro_f1_ci": pct(f1s, 2.5, 97.5)}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(1.0, (ax2-ax1)*(ay2-ay1)) + max(1.0, (bx2-bx1)*(by2-by1)) - inter
    return inter / ua



