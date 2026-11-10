"""v2_marker_split.py — aynı spesifikasyonun v1 (test) → v2 kutu-AP farkı, v1 görüntüsünde yeşil işaretçi kalıntısı var/yok kırılımıyla.
Kalıntısız v1 görüntülerinde fark yalnız üretici değişimini (Gemini/Ideogram → GPT-Image-2) yansıtır; kalıntılılarda üretici + işaretçi kaybı."""
import os
from pathlib import Path
import numpy as np, pandas as pd
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
VP = Path(str(_ROOT))
R = VP / "phase3_v2" / "results"
tab = pd.read_csv(VP / "phase3_v2" / "v2_table.csv", dtype={"new_id": str, "spec_id": str}); prim = tab[tab.primary].copy()
v1 = pd.read_csv(VP / "dataset_audit" / "samples_forged.csv", dtype=str, low_memory=False)
v1t = v1[v1.split == "testing"].drop_duplicates("spec_id").set_index("spec_id")["new_id"]
prim["v1_id"] = prim.spec_id.map(v1t); prim["key"] = [("tr_" if s == "training" else "te_") + n for s, n in zip(prim.split, prim.new_id)]
gm = pd.read_csv(VP / "phase2" / "gt" / "green_marker_all_forged.csv", dtype={"new_id": str})
gm = gm[gm.split == "testing"].set_index("new_id")["has_marker"]
prim["v1_marker"] = prim.v1_id.map(gm).astype("boolean")
P = pd.read_csv(R / "per_image_v2_primary.csv", usecols=["model", "key", "gt", "ap_img", "n_pos"]); P = P[(P["gt"] == "bbox") & (P.n_pos > 0)]
P1 = pd.read_csv(VP / "phase2" / "experiments" / "results" / "per_image_test.csv", dtype={"new_id": str}, usecols=["model", "gt", "new_id", "ap_img"], low_memory=False)
P1 = P1[P1["gt"] == "bbox"].rename(columns={"new_id": "v1_id", "ap_img": "ap_v1"})
def ci(x):
    # Her çağrıda aynı tohumla yeni üreteç: bir satırın aralığı dosyadaki diğer modellere bağlı olmaz (11.11.2026 düzeltmesi;
    # önceden tek üreteç döngü boyunca paylaşılıyordu ve model eklenince sonraki satırların üçüncü ondalığı oynuyordu).
    rng = np.random.default_rng(0)
    x = np.asarray(x); v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"


def gci(x, g, B=1000):
    """Kaynak belge grupları üzerinden bootstrap (tohum ortalamalı satırlar için)."""
    x = np.asarray(x, float); ug, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(ug)); c = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(ug), size=(B, len(ug))); bs = s[idx].sum(1) / c[idx].sum(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ug)
rows = []; J = []
for m in sorted(P.model.unique()):
    j = P[P.model == m].merge(prim[["key", "v1_id", "v1_marker", "v1_tool", "v1_group"]], on="key").merge(P1[P1.model == m][["v1_id", "ap_v1"]], on="v1_id")
    J.append(j)
    for lab, s in (("v1 kalıntılı", j[j.v1_marker == True]), ("v1 kalıntısız", j[j.v1_marker == False])):
        if len(s) >= 10:
            rows.append(dict(model=m, v1_subset=lab, n=len(s), ap_v1=round(s.ap_v1.mean(), 3), ap_v2=round(s.ap_img.mean(), 3), delta_ci=ci(s.ap_img - s.ap_v1)))
T = pd.DataFrame(rows); T.to_csv(R / "T_v2_marker_split.csv", index=False)

# Tohum ortalamalı satırlar: görüntü başı AP önce tohumlar üzerinden ortalanır, sonra v2 − v1 farkı alınır;
# güven aralığı kaynak belge grupları (v1_group) üzerinden bootstrap. Tek tohumlu tabanlar da (n_seeds=1) yazılır.
import re as _re
A = pd.concat(J, ignore_index=True); A["base"] = A.model.map(lambda m: _re.sub(r"[-_]s\d+$", "", m))
arows = []
for b, d in A.groupby("base"):
    k = d.model.nunique()
    a = d.groupby("key").agg(ap_v2=("ap_img", "mean"), ap_v1=("ap_v1", "mean"), n_seed=("model", "nunique"),
                             v1_marker=("v1_marker", "first"), v1_group=("v1_group", "first"))
    a = a[a.n_seed == k]
    for lab, s in (("v1 kalıntılı", a[a.v1_marker == True]), ("v1 kalıntısız", a[a.v1_marker == False])):
        if len(s) >= 10:
            mu, lo, hi, ng = gci(s.ap_v2 - s.ap_v1, s.v1_group.fillna("").where(s.v1_group.fillna("") != "", s.index.to_series()))
            arows.append(dict(base=b, n_seeds=k, v1_subset=lab, n=len(s), n_groups=ng, ap_v1=round(s.ap_v1.mean(), 3), ap_v2=round(s.ap_v2.mean(), 3),
                              delta=round(mu, 4), lo=round(lo, 4), hi=round(hi, 4), delta_ci=f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"))
TA = pd.DataFrame(arows); TA.to_csv(R / "T_v2_marker_split_agg.csv", index=False)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
print(prim.v1_marker.value_counts(dropna=False).to_dict()); print(T.to_string(index=False))
print("\n== tohum ortalamalı (grup-bootstrap) =="); print(TA.drop(columns=["delta", "lo", "hi"]).to_string(index=False))
