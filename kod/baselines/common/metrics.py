"""
common/metrics.py — Sınıflandırma (patch) ve piksel-düzeyi (segmentasyon) metrikleri (TR).

Sınıflandırma: accuracy, balanced accuracy, AUROC, F1 (eşik 0.5) + alt küme AUROC'ları.
Segmentasyon (crop başına): Pixel-F1@0.5, IoU@0.5, Dice, Oracle-F1 (crop başına en iyi eşik),
global en iyi eşik F1; crop'lar üzerinde birleştirilmiş Pixel-AUPRC / Pixel-AUROC (alt örneklenmiş piksel);
temiz (GT'siz) crop'larda yanlış-pozitif piksel oranı.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score,
                             balanced_accuracy_score, confusion_matrix)

THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def safe_auroc(y, s) -> float:
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def classification_metrics(y: np.ndarray, score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y).astype(int); score = np.asarray(score, dtype=float)
    pred = (score >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(
        n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        auroc=safe_auroc(y, score),
        auprc=float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        f1=float(f1_score(y, pred, zero_division=0)),
        tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]),
    )


def subset_auroc(y, score, groups: np.ndarray, keep_fn) -> float:
    """keep_fn(group_value)->bool olan örnekler üzerinde AUROC."""
    mask = np.array([bool(keep_fn(g)) for g in groups])
    if mask.sum() == 0:
        return float("nan")
    return safe_auroc(np.asarray(y)[mask], np.asarray(score)[mask])


# ----------------------------------------------------------------- piksel metrikleri
def _f1_from_counts(tp, fp, fn):
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


def pixel_f1_iou(prob: np.ndarray, gt: np.ndarray, thr: float = 0.5):
    p = prob >= thr
    g = gt > 0
    tp = np.logical_and(p, g).sum(); fp = np.logical_and(p, ~g).sum(); fn = np.logical_and(~p, g).sum()
    f1 = _f1_from_counts(tp, fp, fn)
    union = tp + fp + fn
    iou = (tp / union) if union > 0 else 0.0
    return float(f1), float(iou), int(tp), int(fp), int(fn)


def per_crop_curve(prob: np.ndarray, gt: np.ndarray, grid=THRESH_GRID) -> np.ndarray:
    """Eşik ızgarasında crop'un F1 eğrisi (oracle ve global eşik seçimi için)."""
    g = gt > 0
    npos = g.sum()
    out = np.zeros(len(grid))
    for i, t in enumerate(grid):
        p = prob >= t
        tp = np.logical_and(p, g).sum()
        fp = p.sum() - tp
        fn = npos - tp
        out[i] = _f1_from_counts(tp, fp, fn)
    return out


def segmentation_summary(records: List[Dict], grid=THRESH_GRID) -> Dict:
    """
    records: her val crop için dict(prob=float32 HxW, gt=uint8 HxW, meta=dict).
    GT içeren crop'lar: ortalama F1@0.5, IoU@0.5, Dice(=F1), Oracle-F1, global-eşik F1; birleşik AUPRC/AUROC.
    GT'siz crop'lar: FP piksel oranı (@0.5).
    """
    rng = np.random.default_rng(0)
    pos_recs = [r for r in records if (r["gt"] > 0).any()]
    neg_recs = [r for r in records if not (r["gt"] > 0).any()]
    f1s, ious, curves = [], [], []
    sub_p, sub_g = [], []
    for r in pos_recs:
        f1, iou, *_ = pixel_f1_iou(r["prob"], r["gt"])
        f1s.append(f1); ious.append(iou)
        curves.append(per_crop_curve(r["prob"], r["gt"], grid))
        # birleşik AUPRC/AUROC için piksel alt örnekleme (crop başına en çok 20k piksel; tüm pozitifler dahil)
        flat_p = r["prob"].ravel(); flat_g = (r["gt"].ravel() > 0)
        pos_idx = np.flatnonzero(flat_g)
        neg_idx = np.flatnonzero(~flat_g)
        take = min(len(neg_idx), 20000)
        sel = np.concatenate([pos_idx, rng.choice(neg_idx, take, replace=False)])
        sub_p.append(flat_p[sel]); sub_g.append(flat_g[sel])
    out = dict(n_crops_with_gt=len(pos_recs), n_crops_clean=len(neg_recs))
    if pos_recs:
        curves = np.stack(curves)
        gbest = int(curves.mean(0).argmax())
        out.update(
            pixel_f1_at_0p5=float(np.mean(f1s)), iou_at_0p5=float(np.mean(ious)), dice=float(np.mean(f1s)),
            oracle_f1=float(curves.max(1).mean()),
            global_best_thr=float(grid[gbest]), global_best_thr_f1=float(curves[:, gbest].mean()),
        )
        P = np.concatenate(sub_p); G = np.concatenate(sub_g)
        out["pixel_auprc_pooled_subsampled"] = float(average_precision_score(G, P))
        out["pixel_auroc_pooled_subsampled"] = float(roc_auc_score(G, P))
        out["pixel_positive_rate_in_gt_crops"] = float(np.mean([(r["gt"] > 0).mean() for r in pos_recs]))
    if neg_recs:
        out["clean_crop_fp_pixel_rate_at_0p5"] = float(np.mean([(r["prob"] >= 0.5).mean() for r in neg_recs]))
        out["clean_crop_frac_any_fp_gt_1pct"] = float(np.mean([((r["prob"] >= 0.5).mean() > 0.01) for r in neg_recs]))
    return out


def grouped_segmentation(records: List[Dict], key: str, grid=THRESH_GRID) -> Dict[str, Dict]:
    groups = sorted({r["meta"][key] for r in records})
    return {g: segmentation_summary([r for r in records if r["meta"][key] == g], grid) for g in groups}
