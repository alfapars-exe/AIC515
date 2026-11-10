"""dataset_curation.py — Lab Rat dataset-curation: AIForge-Doc v1 dağılım/bias/karıştırıcı analizi (TR).
Girdi: dataset_audit/samples_forged.csv, splits.json, phase3/results/marker_residue.csv
Çıktı: phase3/results/dataset_curation.md + dataset_crosstabs.csv"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
R = _ROOT
sf = pd.read_csv(R / "dataset_audit" / "samples_forged.csv", dtype={"new_id": str}, low_memory=False)
sf["new_id"] = sf["new_id"].astype(int).map(lambda x: f"{x:09d}")
sf["tool"] = sf.assigned_tool.map({"gemini-nano": "gemini", "qwen-inpaint": "ideogram"})
sp = json.load(open(R / "dataset_audit" / "splits.json"))
lab = {}
for k in ("train", "val", "test"):
    for i in sp[k]:
        lab[(("testing" if k == "test" else "training"), f"{int(i):09d}")] = k
sf["p_split"] = [lab.get((s, i), "excluded?") for s, i in zip(sf.split, sf.new_id)]
r = sf.tamper_ratio
sf["area_bin"] = pd.cut(r * 100, [-1, 1, 5, 20, 101], labels=["<1%", "1-5%", "5-20%", ">20%"])
mr = pd.read_csv(R / "phase3" / "results" / "marker_residue.csv", dtype={"new_id": str})
mr["key"] = mr.split.map({"train": "training", "val": "training", "test": "testing"}) + ":" + mr.new_id
sf["key2"] = sf.split + ":" + sf.new_id
sf = sf.merge(mr[["key", "ring_green_px", "has_residue"]], left_on="key2", right_on="key", how="left")
sf["marker"] = np.where(sf.ring_green_px.fillna(0) > 0, "yes", "no")


def cramers_v(t):
    chi2, p, dof, _ = chi2_contingency(t)
    n = t.values.sum(); k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * max(k, 1)))), float(p)


md = ["# AIForge-Doc v1 — veri kürasyonu ve bias raporu (Lab Rat dataset-curation)\n",
      f"Toplam sahte görüntü: {len(sf)}; split: " + ", ".join(f"{k} {v}" for k, v in sf.p_split.value_counts().items()) + "\n"]
rows = []
md.append("## 1. Dağılımlar ve dengesizlik\n")
for col in ["tool", "source_dataset", "doc_type", "language", "area_bin", "marker"]:
    vc = sf[col].value_counts()
    ratio = vc.max() / max(vc.min(), 1)
    md.append(f"- **{col}**: " + ", ".join(f"{k} {v} ({v / len(sf):.1%})" for k, v in vc.items()) + f" — dengesizlik oranı max/min = {ratio:.1f}" +
              (" — <%5 sınıflar: " + ", ".join(str(k) for k, v in vc.items() if v < 0.05 * vc.max()) if (vc < 0.05 * vc.max()).any() else ""))
md.append("\n## 2. Karıştırıcılar (Cramér's V, ki-kare)\n\n| değişken çifti | V | p | yorum |\n|---|---|---|---|")
pairs = [("tool", "source_dataset"), ("tool", "area_bin"), ("tool", "marker"), ("source_dataset", "area_bin"), ("marker", "area_bin"), ("tool", "language")]
for a, b in pairs:
    t = pd.crosstab(sf[a], sf[b]); v, p = cramers_v(t)
    sev = "güçlü" if v >= 0.5 else "orta" if v >= 0.3 else "zayıf" if v >= 0.1 else "ihmal edilebilir"
    md.append(f"| {a} × {b} | {v:.3f} | {p:.1e} | {sev} |")
    t2 = t.stack().reset_index(); t2.columns = ["a_val", "b_val", "n"]; t2["a"], t2["b"] = a, b; rows.append(t2)
md.append("\n**tool × source_dataset (n)**\n\n" + pd.crosstab(sf.tool, sf.source_dataset, margins=True).to_markdown())
md.append("\n**tool × area_bin (satır %)**\n\n" + (pd.crosstab(sf.tool, sf.area_bin, normalize="index") * 100).round(1).to_markdown())
md.append("\n**tool × marker (satır %)**\n\n" + (pd.crosstab(sf.tool, sf.marker, normalize="index") * 100).round(1).to_markdown())
md.append("\n## 3. Split dengesi (ki-kare, train/val/test)\n\n| değişken | χ² p | yorum |\n|---|---|---|")
S = sf[sf.p_split.isin(["train", "val", "test"])]
for col in ["tool", "source_dataset", "area_bin", "marker"]:
    t = pd.crosstab(S.p_split, S[col]); _, p, _, _ = chi2_contingency(t)
    md.append(f"| {col} | {p:.3f} | {'dağılımlar benzer (p≥0.05)' if p >= 0.05 else 'FARKLI — sonuçlar bu değişkenle tabakalanmalı'} |")
md.append("\n**split × tool (satır %)**\n\n" + (pd.crosstab(S.p_split, S.tool, normalize="index") * 100).round(1).to_markdown())
md.append("\n**split × source (satır %)**\n\n" + (pd.crosstab(S.p_split, S.source_dataset, normalize="index") * 100).round(1).to_markdown())
# Ideogram alt grubu için küçük-örnek uyarısı
ide = S[S.tool == "ideogram"].p_split.value_counts()
md.append(f"\nIdeogram örnek sayısı split başına: {ide.to_dict()} — held-out üretici analizinin istatistiksel gücü sınırlı (val {ide.get('val', 0)}).")
pd.concat(rows).to_csv(R / "phase3" / "results" / "dataset_crosstabs.csv", index=False)
(R / "phase3" / "results" / "dataset_curation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
