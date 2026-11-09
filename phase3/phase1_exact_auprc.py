"""phase1_exact_auprc.py — W6: PHASE 1 pilot U-Net Pixel-AUPRC'sinin TÜM piksellerle kesin yeniden hesabı (TR).
PHASE 1 raporu, crop başına ≤20k negatif alt-örneklemeyle AUPRC verdi (negatif oranı düşürüldüğü için şişkin).
Burada cache/unet_<v>_val_probs.npz (200 crop × 512², uint8 olasılık + GT) üzerinden 256-bin histogramla kesin AUPRC/AUROC.
Çıktı: phase3/results/phase1_exact_auprc.csv"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parents[1]
rows = []
rng = np.random.default_rng(0)
for v, rep in [("rgb", 0.947), ("rgbfreq", 0.939)]:
    z = np.load(ROOT / "baselines" / "cache" / f"unet_{v}_val_probs.npz")
    p, g = z["prob"].ravel().astype(np.int64), z["gt"].ravel() > 0
    hp = np.bincount(p[g], minlength=256); ha = np.bincount(p, minlength=256)
    tp = np.cumsum(hp[::-1]); al = np.cumsum(ha[::-1]); fp = al - tp
    P, N = tp[-1], fp[-1]
    prec = tp / np.maximum(al, 1); rec = tp / max(P, 1)
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    fpr = fp / max(N, 1); auroc = float(np.trapezoid(np.r_[0, rec], np.r_[0, fpr]))
    # PHASE 1 protokolünün yeniden üretimi: crop başına tüm pozitif + ≤20k negatif
    S, Y = [], []
    for k in range(z["prob"].shape[0]):
        pk, gk = z["prob"][k].ravel(), z["gt"][k].ravel() > 0
        if not gk.any():
            continue
        pos, neg = np.flatnonzero(gk), np.flatnonzero(~gk)
        sel = np.concatenate([pos, rng.choice(neg, min(len(neg), 20000), replace=False)])
        S.append(pk[sel]); Y.append(gk[sel])
    ap_sub = float(average_precision_score(np.concatenate(Y), np.concatenate(S)))
    rows.append(dict(variant=v, reported_phase1_auprc=rep, reproduced_subsampled_auprc=round(ap_sub, 4),
                     exact_auprc_all_pixels=round(ap, 4), exact_auroc=round(auroc, 4), pos_rate=round(P / (P + N), 5),
                     n_crops=int(z["prob"].shape[0]), n_pixels=int(P + N)))
df = pd.DataFrame(rows)
out = Path(__file__).resolve().parent / "results"; out.mkdir(exist_ok=True)
df.to_csv(out / "phase1_exact_auprc.csv", index=False)
print(df.to_string(index=False))
