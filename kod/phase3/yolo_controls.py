"""
yolo_controls.py — Kutu dedektörünün zararsız kontrol kümelerindeki görüntü düzeyi ayrımı (TR; T7'nin dedektör karşılığı).

synthesize_p3.py §7 kontrolleri yalnız U-Net ailesi için özetler (T7_controls.csv). Dedektörün aynı üç kümedeki
(authentic, benign_pasteback, fullimage_resample) çıkarımları q3b_eval_yolo.ps1 ile zaten üretilmiştir:
    phase3_yolo/results/infer_val_<kaynak>_<run>.csv   (sütunlar: new_id, n_boxes, top_conf, H, W)
Bu betik onları okur; görüntü skoru = en yüksek kutu güveni (top_conf). AUROC: sahte val görüntüleri (pozitif)
ile aynı kimliklerin kontrol görüntüleri (negatif). Güven aralığı: kaynak-belge grubu üzerinden 1000 örneklemli
bootstrap (tohum 0). U-Net ailesinde karşılığı T7_controls.csv içindeki `auroc_max` sütunudur (en yüksek piksel skoru);
bildirideki segmenter cümlesi ise %99 yüzdelik skoru (`auroc_top1`) kullanır — iki skor tanımı karıştırılmamalıdır.

Çıktı : phase3/results/T7b_yolo_controls.csv
Kullanım: python kod/phase3/yolo_controls.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

YRES = ROOT / "phase3_yolo" / "results"
OUT = ROOT / "phase3" / "results" / "T7b_yolo_controls.csv"
SOURCES = ["authentic", "benign_pasteback", "fullimage_resample"]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney U ile AUROC (eşitlikler ortalama sırayla)."""
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    grp = pd.read_csv(ROOT / "phase2" / "experiments" / "results" / "per_image_val.csv", usecols=["new_id", "group"], dtype={"new_id": str})
    grp = grp.drop_duplicates("new_id").set_index("new_id").group
    rows = []
    for fa in sorted(YRES.glob("infer_val_authentic_*.csv")):
        run = re.sub(r"^infer_val_authentic_", "", fa.stem)
        ff = YRES / f"infer_val_{run}.csv"
        if not ff.is_file():
            continue
        f = pd.read_csv(ff, dtype={"new_id": str}).set_index("new_id").top_conf
        base = None
        for src in SOURCES:
            fn = YRES / f"infer_val_{src}_{run}.csv"
            if not fn.is_file():
                continue
            n = pd.read_csv(fn, dtype={"new_id": str}).set_index("new_id").top_conf
            ids = f.index.intersection(n.index)
            au = auroc(f[ids].to_numpy(), n[ids].to_numpy())
            g = grp.reindex(ids).fillna(pd.Series(ids, index=ids)).to_numpy()          # grubu bilinmeyen kimlik kendi grubudur
            ug = np.unique(g); members = {k: np.flatnonzero(g == k) for k in ug}
            rng = np.random.default_rng(0); bs = []
            fv, nv = f[ids].to_numpy(), n[ids].to_numpy()
            for _ in range(1000):
                idx = np.concatenate([members[k] for k in rng.choice(ug, len(ug), replace=True)])
                bs.append(auroc(fv[idx], nv[idx]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            if src == "authentic":
                base = au
            rows.append(dict(run=run, source=src, n_pairs=len(ids), n_groups=len(ug), auroc_maxconf=round(au, 4), ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                             delta_vs_authentic_pts=None if base is None else round(100 * (au - base), 2),
                             control_rate_conf_ge_0p5=round(float((nv >= 0.5).mean()), 4)))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    t7 = ROOT / "phase3" / "results" / "T7_controls.csv"
    if t7.is_file():
        u = pd.read_csv(t7)
        u = u[(u.family == "U-Net") & (u.variant == "main") & (u.train_gt == "box")][["source", "seeds", "auroc_top1", "auroc_max"]]
        print("\nKarşılaştırma — U-Net kutu modeli (T7_controls.csv):\n" + u.to_string(index=False))
    print(f"\nyazıldı: {OUT}")


if __name__ == "__main__":
    main()
