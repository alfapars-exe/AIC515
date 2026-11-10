"""seed_extra_stats.py — ek tohum (43, 44) koşularının öngörü değerlendirmesi için tamamlayıcı istatistikler (TR).

Girdi (yalnız okunur): phase2/experiments/results/{summary_val,summary_test,per_image_val,per_image_test}.csv
Çıktı: phase3/results/extra_seed_stats.json ve extra_seed_stats.md

1) Tohum bazında havuzlanmış AUPRC (doğrulama ve test) — standart / işaretçi-onarımlı kutu U-Net,
   standart / birleşik kayıplı piksel U-Net.
2) Eşli görüntü başı AP farkları: görüntü başı AP önce tohumlar üzerinden ortalanır, fark alınır,
   %95 güven aralığı kaynak belge grupları üzerinden bootstrap ile hesaplanır (synthesize_p3.py ile aynı: B=1000).
Kullanım: python phase3/seed_extra_stats.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT  # proje kökü
ROOT = _ROOT
R2 = ROOT / "phase2" / "experiments" / "results"; OUT = ROOT / "phase3" / "results"
B, SEED = 1000, 20261109
SEEDS = (42, 43, 44)
BASES = {"std_box": "unet_bbox", "mr_box": "unet_bbox_mr", "std_pix": "unet_pixel_eps8_morph", "joint_pix": "unetjoint_pixel_eps8_morph"}
GS = {"bbox": "box", "pixel_eps8_morph": "pix", "glyph_v2": "glyph"}


def names(base):
    return [f"{base}_s{s}" for s in SEEDS]


def gboot(v, g):
    v = np.asarray(v, float); g = np.asarray(g).astype(str)
    ug, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(ug)); c = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(ug), size=(B, len(ug)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(v)), int(len(ug))


res: dict = {"seeds": list(SEEDS), "bootstrap": {"B": B, "seed": SEED, "unit": "kaynak belge grubu"}, "per_seed": {}, "paired": {}}
md = ["# Ek tohum istatistikleri (otomatik; seed_extra_stats.py)\n"]

# ---------------------------------------------------------------- 1) tohum bazında havuzlanmış AUPRC
sv = pd.read_csv(R2 / "summary_val.csv"); st = pd.read_csv(R2 / "summary_test.csv")
sv = sv[sv.condition == "clean"]; st = st[st.condition == "clean"]
rows = []
for key, base in BASES.items():
    for split, df, subs in (("val", sv, [("all", g) for g in GS]), ("test", st, [("all", "bbox"), ("clean_test", "bbox"), ("paired_89", "pixel_eps8_morph")])):
        for sub, gt in subs:
            d = df[df.model.isin(names(base)) & (df["gt"] == gt)]
            if "subset" in d.columns: d = d[d.subset == sub]
            d = d.set_index("model").reindex(names(base))
            raw = [float(x) for x in d.auprc_pooled if not pd.isna(x)]
            vals = [None if pd.isna(x) else round(float(x), 4) for x in d.auprc_pooled]
            ok = raw
            if not ok: continue
            m, sd = float(np.mean(ok)), (float(np.std(ok, ddof=1)) if len(ok) > 1 else float("nan"))
            res["per_seed"][f"{key}|{split}|{sub}|{GS[gt]}"] = dict(values=vals, mean=round(m, 4), sd=round(sd, 4), n=len(ok))
            rows.append(dict(model=key, split=split, subset=sub, gt=GS[gt], s42=vals[0], s43=vals[1], s44=vals[2], mean=f"{m:.3f}", sd=f"{sd:.3f}"))
T = pd.DataFrame(rows)
md.append("## 1. Tohum bazında havuzlanmış AUPRC\n\n" + T.to_markdown(index=False))

# ---------------------------------------------------------------- 2) eşli görüntü başı AP farkları
def seed_avg(df, base, gt, extra=None):
    d = df[df.model.isin(names(base)) & (df["gt"] == gt) & (df.condition == "clean") & (df.n_pos > 0)]
    if extra is not None: d = d[extra(d)]
    n_seed = d.model.nunique()
    a = d.groupby("new_id").agg(ap=("ap_img", "mean"), k=("model", "nunique"), group=("group", "first"))
    return a[a.k == n_seed], n_seed


cols = ["new_id", "model", "condition", "gt", "group", "ap_img", "n_pos"]
pv = pd.read_csv(R2 / "per_image_val.csv", dtype={"new_id": str}, usecols=cols, low_memory=False)
pt = pd.read_csv(R2 / "per_image_test.csv", dtype={"new_id": str}, usecols=cols + ["clean_test"], low_memory=False)
rows = []
for lab, a_key, b_key in (("birleşik piksel − standart piksel", "joint_pix", "std_pix"), ("işaretçi-onarımlı kutu − standart kutu", "mr_box", "std_box")):
    for split, df in (("val", pv), ("test", pt)):
        for gt in ("bbox", "pixel_eps8_morph"):
            A, ka = seed_avg(df, BASES[a_key], gt); Bm, kb = seed_avg(df, BASES[b_key], gt)
            j = A.join(Bm, lsuffix="_a", rsuffix="_b", how="inner")
            if len(j) < 10: continue
            m, lo, hi, n, ng = gboot(j.ap_a - j.ap_b, j.group_a)
            ma, mb = float(j.ap_a.mean()), float(j.ap_b.mean())
            res["paired"][f"{a_key}-{b_key}|{split}|{GS[gt]}"] = dict(n=n, n_groups=ng, seeds_a=ka, seeds_b=kb, ap_a=round(ma, 4), ap_b=round(mb, 4),
                                                                     delta=round(m, 4), lo=round(lo, 4), hi=round(hi, 4))
            rows.append(dict(karşılaştırma=lab, bölme=split, gt=GS[gt], n=n, grup=ng, AP_a=f"{ma:.3f}", AP_b=f"{mb:.3f}", fark=f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"))
P = pd.DataFrame(rows)
md.append("\n## 2. Eşli görüntü başı AP farkı (tohum ortalamalı, grup-bootstrap %95 GA)\n\n" + P.to_markdown(index=False))

json.dump(res, open(OUT / "extra_seed_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
(OUT / "extra_seed_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
print("\n".join(md))
