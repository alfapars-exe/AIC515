"""make_space_meta.py — Faz 3 sonuçlarını HF Space'in küçük meta dosyalarına aktarır (TR).
  * tau.json (38 model: Faz 2 + Faz 3 segmentasyon + YOLO)            → hf_space/phase2/experiments/results/, hf_space/meta/
  * summary_test_agg.csv (güncel)                                      → hf_space/phase2/experiments/results/
  * görüntü-düzeyi eşikler (Youden-J, skor = en yüksek %1 piksel ort.) → hf_space/meta/image_tau.json (mevcut girdiler korunur)
      sahte: per_image_val.csv (gt=bbox, clean) · authentic: per_image_controls.csv (condition=authentic)
      YOLO s42: authentic skorları preds/<run>/val_authentic/*.png haritalarından (aynı skor tanımı)
  * phase3/results/tables_p3.md                                        → hf_space/meta/phase3_tables.md
Kullanım: python phase3/make_space_meta.py
"""
import glob, json, shutil
from pathlib import Path
import numpy as np, pandas as pd, cv2
from sklearn.metrics import roc_curve, roc_auc_score

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
ROOT = _ROOT
R = ROOT / "phase2" / "experiments" / "results"; PRED = ROOT / "phase2" / "experiments" / "preds"
SP = ROOT / "hf_space"; META = _KOD / "app" / "meta"  # asıl kaynak; hf_space/meta türetilir


def top1(png):
    u = cv2.imread(png, cv2.IMREAD_GRAYSCALE).ravel(); k = max(1, int(0.01 * u.size))
    return float(np.partition(u, -k)[-k:].mean() / 255.0)


def youden(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]
    fpr, tpr, thr = roc_curve(y, s); i = int(np.argmax(tpr - fpr)); t = float(min(thr[i], 1.0))
    return dict(thr=round(t, 4), tpr=round(float(tpr[i]), 3), fpr=round(float(fpr[i]), 3),
                auroc=round(float(roc_auc_score(y, s)), 3), n_pos=int(len(pos)), n_neg=int(len(neg)))


for d in (SP / "phase2" / "experiments" / "results", META):
    d.mkdir(parents=True, exist_ok=True); shutil.copy2(R / "tau.json", d / "tau.json")
shutil.copy2(R / "summary_test_agg.csv", SP / "phase2" / "experiments" / "results" / "summary_test_agg.csv")
shutil.copy2(ROOT / "phase3" / "results" / "tables_p3.md", META / "phase3_tables.md")

it = json.load(open(META / "image_tau.json", encoding="utf-8"))
val = pd.read_csv(R / "per_image_val.csv", dtype={"new_id": str}, low_memory=False)
val = val[(val["gt"] == "bbox") & (val["condition"] == "clean")]
ctl = pd.read_csv(R / "per_image_controls.csv", dtype={"new_id": str}, low_memory=False)
ctl = ctl[ctl["condition"] == "authentic"]
new = []
for m in sorted(set(val["model"]) - set(it)):
    pos = val.loc[val["model"] == m, "score_top1pct"].to_numpy()
    neg = ctl.loc[ctl["model"] == m, "score_top1pct"].to_numpy()
    if len(neg) == 0:
        pngs = sorted(glob.glob(str(PRED / m / "val_authentic" / "*.png")))
        neg = np.array([top1(p) for p in pngs])
    if len(pos) and len(neg):
        it[m] = youden(pos, neg); new.append(m)
json.dump(it, open(META / "image_tau.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("image_tau eklenen:", len(new)); [print(" ", m, it[m]) for m in new]
