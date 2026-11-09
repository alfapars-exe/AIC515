"""
common/p2metrics.py — Tam-görüntü piksel metrikleri, çoklu GT (TR).

Temel yapı: görüntü × GT başına iki 256-bin histogram — h_pos (GT-pozitif piksellerin uint8 olasılık dağılımı) ve
h_all (tüm piksellerin). Tahminler uint8 olduğundan bu histogramlar F1/IoU eğrisi, Pixel-AP ve Pixel-AUROC'u TÜM
pikseller üzerinde KESİN olarak verir (alt-örnekleme gerekmez); havuzlanmış (dataset) ve bootstrap (grup) metrikleri
histogram toplamıyla hesaplanır (results/hists/*.npz).
  * F1@0.5, IoU@0.5, F1 eğrisi (ızgara 0.05..0.95), Oracle-F1, F1@τ* (τ* = val ortalama F1 eğrisinin argmax'ı)
  * ap_img / auroc_img (görüntü içi), auprc_pooled / auroc_pooled (histogram toplamı)
  * pred_rectangularity (alan / bbox alanı) ve 8-komşuluk bileşen sayısı @0.5 ve @τ*
  * image-level skorlar: max prob, top-1% ortalama, pozitif piksel oranı
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
GRID_U8 = np.round(THRESH_GRID * 255).astype(np.int32)


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def hists(prob_u8: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = prob_u8.ravel(); g = gt.ravel() > 0
    h_all = np.bincount(p, minlength=256).astype(np.int64)
    h_pos = np.bincount(p[g], minlength=256).astype(np.int64)
    return h_pos, h_all


def curves_from_hist(h_pos: np.ndarray, h_all: np.ndarray):
    """Eşik ızgarasında (F1, IoU, tahmin alanı)."""
    c_pos = np.cumsum(h_pos[::-1])[::-1]; c_all = np.cumsum(h_all[::-1])[::-1]
    npos = int(h_pos.sum())
    f1s, ious, areas = np.zeros(len(GRID_U8)), np.zeros(len(GRID_U8)), np.zeros(len(GRID_U8), np.int64)
    for i, t in enumerate(GRID_U8):
        tp = int(c_pos[t]); pred = int(c_all[t]); fp = pred - tp; fn = npos - tp
        f1s[i] = _f1(tp, fp, fn); u = tp + fp + fn
        ious[i] = (tp / u) if u > 0 else 0.0; areas[i] = pred
    return f1s, ious, areas


def ap_auroc_from_hist(h_pos: np.ndarray, h_all: np.ndarray) -> Tuple[float, float]:
    """Kesin AP (sklearn average_precision step tanımı) ve AUROC (trapez; eşit skorlar trapezle) — 256 seviyeli skorlar."""
    h_neg = h_all - h_pos
    P = h_pos.sum(); N = h_neg.sum()
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(h_pos[::-1]); fp = np.cumsum(h_neg[::-1])  # eşik 255,254,...,0 için prob>=t
    recall = tp / P; prec = tp / np.maximum(tp + fp, 1)
    r_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - r_prev) * prec))
    fpr = fp / N; tpr = recall
    fpr0 = np.concatenate([[0.0], fpr]); tpr0 = np.concatenate([[0.0], tpr])
    auroc = float(np.trapezoid(tpr0, fpr0))
    return ap, auroc


def rectangularity(binary: np.ndarray) -> Tuple[float, int]:
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return float("nan"), 0
    area = len(xs); bb = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    n, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return float(area / bb), int(n - 1)


def image_level_scores(prob_u8: np.ndarray) -> Dict[str, float]:
    p = prob_u8.ravel()
    k = max(1, int(round(0.01 * p.size)))
    top = np.partition(p, p.size - k)[p.size - k:]
    return dict(score_max=float(p.max()) / 255.0, score_top1pct=float(top.mean()) / 255.0,
                pos_frac_0p5=float((p >= 128).mean()), pos_frac_0p3=float((p >= 77).mean()))


def per_image_metrics(prob_u8: np.ndarray, gt: Optional[np.ndarray], tau: Optional[float] = None,
                      with_shape: bool = True) -> Tuple[Dict, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Tek görüntü, tek GT (gt None → yalnız image-level/şekil metrikleri). Dönüş (metrikler, (h_pos,h_all)|None)."""
    out = image_level_scores(prob_u8)
    if with_shape:
        out["rect_0p5"], out["ncc_0p5"] = rectangularity(prob_u8 >= 128)
        if tau is not None:
            out["rect_tau"], out["ncc_tau"] = rectangularity(prob_u8 >= int(round(tau * 255)))
    if gt is None:
        out["n_pos"] = -1
        return out, None
    h_pos, h_all = hists(prob_u8, gt)
    out["n_pos"] = int(h_pos.sum()); out["gt_frac"] = out["n_pos"] / max(int(h_all.sum()), 1)
    if out["n_pos"] == 0:
        return out, (h_pos, h_all)
    f1s, ious, _ = curves_from_hist(h_pos, h_all)
    i05 = int(np.argmin(np.abs(THRESH_GRID - 0.5)))
    out["f1_0p5"], out["iou_0p5"] = float(f1s[i05]), float(ious[i05])
    out["oracle_f1"] = float(f1s.max()); out["oracle_thr"] = float(THRESH_GRID[int(f1s.argmax())])
    out["f1_curve"] = f1s.tolist()
    if tau is not None:
        it = int(np.argmin(np.abs(THRESH_GRID - tau)))
        out["f1_tau"], out["iou_tau"] = float(f1s[it]), float(ious[it])
    out["ap_img"], out["auroc_img"] = ap_auroc_from_hist(h_pos, h_all)
    return out, (h_pos, h_all)


def pooled_from_hists(H_pos: np.ndarray, H_all: np.ndarray) -> Dict[str, float]:
    """(N,256) histogramlar → havuzlanmış AUPRC/AUROC (tüm pikseller)."""
    ap, au = ap_auroc_from_hist(H_pos.sum(0), H_all.sum(0))
    return dict(auprc_pooled=ap, auroc_pooled=au)


def choose_tau(curves: np.ndarray) -> Tuple[float, float]:
    m = curves.mean(0); i = int(m.argmax())
    return float(THRESH_GRID[i]), float(m[i])


def rect_fit(prob_u8: np.ndarray, tau: float) -> np.ndarray:
    """τ eşik → en büyük 8-komşuluk bileşen → bbox'ı doldur → ikili maske (uint8 0/1)."""
    b = (prob_u8 >= int(round(tau * 255))).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = np.zeros_like(b)
    if n <= 1:
        return out
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(stats[k, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    out[y:y + h, x:x + w] = 1
    return out


def binary_f1_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    p = pred > 0; g = gt > 0
    tp = int(np.logical_and(p, g).sum()); fp = int(p.sum()) - tp; fn = int(g.sum()) - tp
    u = tp + fp + fn
    return _f1(tp, fp, fn), (tp / u if u > 0 else 0.0)
