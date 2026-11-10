"""v2_report.py — AIForge-Doc v2 harici test tabloları (TR). Girdi: results/*_v2_{primary,all}.csv + v1 test sonuçları.
Çıktı: results/v2_tables.md, results/T_v2.csv, results/T_v2_image.csv, results/v2_numbers.json, ../paper/sec/tab_v2.tex"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
VP = Path(str(_ROOT))
R = VP / "phase3_v2" / "results"; R1 = VP / "phase2" / "experiments" / "results"
tab = pd.read_csv(VP / "phase3_v2" / "v2_table.csv", dtype={"new_id": str, "spec_id": str})
prim = tab[tab.primary].copy()
v1 = pd.read_csv(VP / "dataset_audit" / "samples_forged.csv", dtype=str, low_memory=False)
v1t = v1[v1.split == "testing"].drop_duplicates("spec_id").set_index("spec_id")["new_id"]
prim["v1_test_id"] = prim.spec_id.map(v1t)
prim["key"] = [("tr_" if s == "training" else "te_") + n for s, n in zip(prim.split, prim.new_id)]
k2v1 = dict(zip(prim.key, prim.v1_test_id))

S = pd.read_csv(R / "summary_v2_primary.csv"); I = pd.read_csv(R / "image_level_v2_primary.csv")
P = pd.read_csv(R / "per_image_v2_primary.csv", usecols=["model", "key", "gt", "ap_img", "n_pos"])
P1 = pd.read_csv(R1 / "per_image_test.csv", dtype={"new_id": str}, usecols=["model", "gt", "new_id", "ap_img"], low_memory=False)
P1 = P1[P1["gt"] == "bbox"]

LABEL = {"unet_bbox": "U-Net box", "unet_pixel_eps8_morph": "U-Net pix", "unet_glyph_v2": "U-Net glyph",
         "dlv3p_bbox": "DLV3+ box", "dlv3p_pixel_eps8_morph": "DLV3+ pix", "dlv3p_glyph_v2": "DLV3+ glyph",
         "unet_bbox_mr": "U-Net box, marker-repaired", "unet_pixel_eps8_morph_nm_mr": "U-Net pix, marker-repaired",
         "unet_glyph_v2_nm_mr": "U-Net glyph, marker-repaired", "unet_bbox_geminionly": "U-Net box, Gemini-only",
         "unet_pixel_eps8_morph_geminionly": "U-Net pix, Gemini-only", "unet_glyph_v2_geminionly": "U-Net glyph, Gemini-only",
         "aiforge-tiles-yolo11s-img1024-conserv": "YOLO11s tiles", "aiforge-full-yolo11s-img1280-conserv": "YOLO11s full 1280",
         "aiforge-tiles_geminionly-yolo11s-img1024-conserv": "YOLO11s, Gemini-only",
         "aiforge-tiles_authneg-yolo11s-img1024-conserv": "YOLO11s, +authentic tiles"}


def base(m):
    for s in ("_s42", "_s43", "_s44", "-s42", "-s43", "-s44"):
        if m.endswith(s):
            return m[: -len(s)]
    return m


def boot_mean(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; rng = np.random.default_rng(seed)
    v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rows = []
for m in sorted(S.model.unique()):
    s = S[(S.model == m) & (S.subset == "primary")].set_index("gt")
    il = I[(I.model == m) & (I.subset == "primary")]
    # aynı spesifikasyonun v1 (test) ve v2 sürümü: kutu GT görüntü-AP farkı
    a = P[(P.model == m) & (P["gt"] == "bbox") & (P.n_pos > 0)][["key", "ap_img"]].copy()
    a["v1_id"] = a.key.map(k2v1)
    b = P1[P1.model == m][["new_id", "ap_img"]].rename(columns={"new_id": "v1_id", "ap_img": "ap_v1"})
    j = a.merge(b, on="v1_id", how="inner")
    d = boot_mean(j.ap_img - j.ap_v1) if len(j) else (np.nan,) * 3
    rows.append(dict(model=m, base=base(m), n=int(s.loc["bbox", "n"]),
                     auprc_box=s.loc["bbox", "auprc_pooled"], auprc_pix=s.loc["pixel_eps8_morph", "auprc_pooled"],
                     auprc_glyph=s.loc["glyph_v2", "auprc_pooled"], f1tau_box=s.loc["bbox", "f1_tau"],
                     iou_tau_box=s.loc["bbox", "iou_tau"], f1tau_pix=s.loc["pixel_eps8_morph", "f1_tau"],
                     img_auroc=float(il.auroc_top1pct.iloc[0]) if len(il) else np.nan,
                     img_ci=il.ci95.iloc[0] if len(il) else "", tpr_v1thr=float(il.tpr_at_v1_thr.iloc[0]) if len(il) else np.nan,
                     fpr_v1thr=float(il.fpr_at_v1_thr.iloc[0]) if len(il) else np.nan,
                     n_matched=len(j), ap_v1_matched=float(j.ap_v1.mean()) if len(j) else np.nan,
                     ap_v2_matched=float(j.ap_img.mean()) if len(j) else np.nan, d_ap=d[0], d_ap_lo=d[1], d_ap_hi=d[2]))
T = pd.DataFrame(rows); T.to_csv(R / "T_v2.csv", index=False)
num = ["auprc_box", "auprc_pix", "auprc_glyph", "f1tau_box", "f1tau_pix", "img_auroc", "ap_v1_matched", "ap_v2_matched", "d_ap"]
G = T.groupby("base")[num].agg(["mean", "std", "count"])

# alt kümeler (primary içi): kaynak, v1 üreticisi
sub = S[S.subset.str.startswith(("source_dataset=", "v1_tool=", "area_bin="))]
SUB = sub.pivot_table(index=["model", "gt"], columns="subset", values="auprc_pooled")

L = ["# AIForge-Doc v2 (GPT-Image-2) harici test — v1'de eğitilmiş modeller, yeniden eğitim yok", "",
     f"Birincil küme: {len(prim)} çift (v1 karşılığı TestingSet'te, kaynak belge v1 eğitiminde görülmemiş). "
     f"v2'de renk kuralıyla işaretçi kalıntısı: {int(tab.green_residue.sum())}/{len(tab)} görüntü (eşli kural ve konum: `v2_marker_paired_check.json`); "
     f"kutu dışı değişen piksel: {int(tab.pixel_eps8_n_out.sum())} (patch-back).", "",
     "## Tohum ortalamaları (birincil küme)", "", "| model | n tohum | box AUPRC | pix AUPRC | glyph AUPRC | F1@τ* box | görüntü AUROC | v1→v2 eşleşik AP (box) | ΔAP |",
     "|---|---|---|---|---|---|---|---|---|"]
for bse, r in G.iterrows():
    def f(c):
        mu, sd, n = r[(c, "mean")], r[(c, "std")], r[(c, "count")]
        return f"{mu:.3f}" + (f"±{sd:.3f}" if n > 1 and not np.isnan(sd) else "")
    L.append(f"| {LABEL.get(bse, bse)} | {int(r[('auprc_box', 'count')])} | {f('auprc_box')} | {f('auprc_pix')} | {f('auprc_glyph')} | "
             f"{f('f1tau_box')} | {f('img_auroc')} | {r[('ap_v1_matched', 'mean')]:.3f} → {r[('ap_v2_matched', 'mean')]:.3f} | {f('d_ap')} |")
L += ["", "## Tek tek modeller", "", T.round(4).to_markdown(index=False), "", "## Alt kümeler (pooled AUPRC)", "", SUB.round(3).to_markdown()]
if (R / "summary_v2_all.csv").exists():
    SA = pd.read_csv(R / "summary_v2_all.csv"); IA = pd.read_csv(R / "image_level_v2_all.csv")
    L += ["", "## Tüm v2 (3066 çift; 'seen' = kaynak belge v1 eğitiminde) — README taban çizgileriyle karşılaştırma için", "",
          SA[SA.subset.isin(["all", "primary", "seen"])].pivot_table(index=["model", "gt"], columns="subset", values="auprc_pooled").round(3).to_markdown(), "",
          IA[IA.subset.isin(["all", "primary", "seen"])][["model", "subset", "n_pairs", "auroc_top1pct", "ci95", "tpr_at_v1_thr", "fpr_at_v1_thr"]].round(3).to_markdown(index=False),
          "", "README (zero-shot, tüm 3066 çift): TruFor AUC 0.599, DocTamper 0.585, GPT-Image-2 öz-yargı 0.532, insan 2AFC 0.501."]
(R / "v2_tables.md").write_text("\n".join(L), encoding="utf-8")
json.dump({"n_primary": int(len(prim)), "green_residue_share": float(tab.green_residue.mean()),
           "seed_means": {b: {c: float(G.loc[b, (c, "mean")]) for c in num} for b in G.index}},
          open(R / "v2_numbers.json", "w"), indent=1)
print("\n".join(L[:30]))
