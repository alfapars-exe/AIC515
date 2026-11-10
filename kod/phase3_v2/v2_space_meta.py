"""v2_space_meta.py — Space 'Sonuçlar' sekmesi için kısa AIForge-Doc v2 özeti → app/meta/v2_tables.md ve hf_space/meta/v2_tables.md"""
import os
from pathlib import Path
import pandas as pd
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
VP = Path(str(_ROOT)); R = VP / "phase3_v2" / "results"
T = pd.read_csv(R / "T_v2.csv"); M = pd.read_csv(R / "T_v2_marker_split.csv")
M["d"] = M.delta_ci.str.extract(r"([+-][0-9.]+)").astype(float)
for d in (T, M): d["base"] = d.model.str.replace(r"[-_]s4[234]$", "", regex=True)
NAME = {"unet_bbox": "U-Net box (3 tohum)", "unet_pixel_eps8_morph": "U-Net piksel (3)", "unet_glyph_v2": "U-Net glif (3)",
        "dlv3p_bbox": "DeepLabV3+ box", "unet_bbox_mr": "U-Net box, işaretçisiz (3 tohum)", "unet_pixel_eps8_morph_nm_mr": "U-Net piksel, işaretçisiz",
        "unet_glyph_v2_nm_mr": "U-Net glif, işaretçisiz", "unet_bbox_geminionly": "U-Net box, yalnız Gemini",
        "aiforge-tiles-yolo11s-img1024-conserv": "YOLO11s kare (3)", "aiforge-full-yolo11s-img1280-conserv": "YOLO11s tam 1280",
        "aiforge-tiles_geminionly-yolo11s-img1024-conserv": "YOLO11s yalnız Gemini", "aiforge-tiles_authneg-yolo11s-img1024-conserv": "YOLO11s +authentic kare"}
g = T.groupby("base")[["auprc_box", "auprc_pix", "auprc_glyph", "img_auroc"]].mean()
m = M.pivot_table(index="base", columns="v1_subset", values="d")
rows = [(NAME[b], *g.loc[b].round(3).tolist(), round(m.loc[b, "v1 kalıntılı"], 3), round(m.loc[b, "v1 kalıntısız"], 3)) for b in NAME if b in g.index]
_mr = ""  # işaretçisiz box U-Net, v1 kalıntısız tarifler: tohum ortalamalı ΔAP ve grup-bootstrap aralığı (v2_marker_split.py çıktısı)
_agg = R / "T_v2_marker_split_agg.csv"
if _agg.exists():
    _a = pd.read_csv(_agg); _a = _a[(_a.base == "unet_bbox_mr") & (_a.v1_subset == "v1 kalıntısız")]
    if len(_a): _mr = f" ({int(_a.n_seeds.iloc[0])} tohum: {str(_a.delta_ci.iloc[0]).replace('-', '−')})"
tbl = pd.DataFrame(rows, columns=["model (v1'de eğitildi)", "box AUPRC", "piksel AUPRC", "glif AUPRC", "görüntü AUROC", "ΔAP v1 kalıntılı", "ΔAP v1 kalıntısız"])
L = ["## AIForge-Doc v2 (GPT-Image-2) — harici test, yeniden eğitim yok", "",
     "567 çiftlik birincil küme: v1 karşılığı test bölümünde, kaynak belgesi v1 eğitiminde hiç görülmemiş. v2'de yalnız düzenlenen kutu geri yapıştırılır; "
     "yeşil işaretçi kalıntısı yoktur (3066 çiftin hiçbirinde kutu dışında farklı piksel yok; renk kuralı yalnız yeşil tonlu tek bir fişte, kutunun içinde tetikleniyor). Eşikler v1 doğrulamadan, değişmeden. ΔAP: aynı spesifikasyonun v1 test görüntüsünden v2 görüntüsüne "
     "görüntü-başı AP değişimi; v1 görüntüsünde işaretçi kalıntısı olan (397) ve olmayan (170) çiftler ayrı.", "", tbl.to_markdown(index=False), "",
     "**Okuma:** box U-Net v1'de 0.95 iken v2'de 0.52'ye düşüyor; kaybın çoğu v1'de işaretçi taşıyan spesifikasyonlarda. İşaretçisiz eğitilen box U-Net "
     "kalıntısız spesifikasyonlarda ölçülebilir kayıp göstermiyor" + _mr + "; kareli YOLO11s 0.89'da kalıyor."]
if (R / "image_level_v2_all.csv").exists():
    IA = pd.read_csv(R / "image_level_v2_all.csv"); IA = IA[IA.subset.isin(["all", "primary", "seen"])]
    L += ["", "**Tüm v2 (3066 çift) görüntü AUROC** — 'seen': kaynak belge v1 eğitiminde vardı; README zero-shot taban çizgileri: TruFor 0.599, DocTamper 0.585, insan 0.501.", "",
          IA[["model", "subset", "n_pairs", "auroc_top1pct"]].round(3).to_markdown(index=False)]
txt = "\n".join(L)
for d in (_KOD / "app" / "meta",):  # hf_space türetilir (tools/build_space.py)
    d.mkdir(parents=True, exist_ok=True); (d / "v2_tables.md").write_text(txt, encoding="utf-8")
print(txt)
