"""v2_space_meta.py — Space 'Sonuçlar' sekmesi için kısa AIForge-Doc v2 özeti → app/meta/v2_tables.md ve hf_space/meta/v2_tables.md"""
import os
from pathlib import Path
import pandas as pd
VP = Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")); R = VP / "phase3_v2" / "results"
T = pd.read_csv(R / "T_v2.csv"); M = pd.read_csv(R / "T_v2_marker_split.csv")
M["d"] = M.delta_ci.str.extract(r"([+-][0-9.]+)").astype(float)
for d in (T, M): d["base"] = d.model.str.replace(r"[-_]s4[234]$", "", regex=True)
NAME = {"unet_bbox": "U-Net box (3 tohum)", "unet_pixel_eps8_morph": "U-Net piksel (3)", "unet_glyph_v2": "U-Net glif (3)",
        "dlv3p_bbox": "DeepLabV3+ box", "unet_bbox_mr": "U-Net box, işaretçisiz eğitim", "unet_pixel_eps8_morph_nm_mr": "U-Net piksel, işaretçisiz",
        "unet_glyph_v2_nm_mr": "U-Net glif, işaretçisiz", "unet_bbox_geminionly": "U-Net box, yalnız Gemini",
        "aiforge-tiles-yolo11s-img1024-conserv": "YOLO11s kare (3)", "aiforge-full-yolo11s-img1280-conserv": "YOLO11s tam 1280",
        "aiforge-tiles_geminionly-yolo11s-img1024-conserv": "YOLO11s yalnız Gemini", "aiforge-tiles_authneg-yolo11s-img1024-conserv": "YOLO11s +authentic kare"}
g = T.groupby("base")[["auprc_box", "auprc_pix", "auprc_glyph", "img_auroc"]].mean()
m = M.pivot_table(index="base", columns="v1_subset", values="d")
rows = [(NAME[b], *g.loc[b].round(3).tolist(), round(m.loc[b, "v1 kalıntılı"], 3), round(m.loc[b, "v1 kalıntısız"], 3)) for b in NAME if b in g.index]
tbl = pd.DataFrame(rows, columns=["model (v1'de eğitildi)", "box AUPRC", "piksel AUPRC", "glif AUPRC", "görüntü AUROC", "ΔAP v1 kalıntılı", "ΔAP v1 kalıntısız"])
L = ["## AIForge-Doc v2 (GPT-Image-2) — harici test, yeniden eğitim yok", "",
     "567 çiftlik birincil küme: v1 karşılığı test bölümünde, kaynak belgesi v1 eğitiminde hiç görülmemiş. v2'de yalnız düzenlenen kutu geri yapıştırılır; "
     "yeşil işaretçi kalıntısı yoktur (3066 görüntünün 0'ı). Eşikler v1 doğrulamadan, değişmeden. ΔAP: aynı spesifikasyonun v1 test görüntüsünden v2 görüntüsüne "
     "görüntü-başı AP değişimi; v1 görüntüsünde işaretçi kalıntısı olan (397) ve olmayan (170) çiftler ayrı.", "", tbl.to_markdown(index=False), "",
     "**Okuma:** box U-Net v1'de 0.95 iken v2'de 0.52'ye düşüyor; kaybın çoğu v1'de işaretçi taşıyan spesifikasyonlarda. İşaretçisiz eğitilen box U-Net "
     "kalıntısız spesifikasyonlarda hiç kaybetmiyor (tek tohum); kareli YOLO11s 0.89'da kalıyor."]
if (R / "image_level_v2_all.csv").exists():
    IA = pd.read_csv(R / "image_level_v2_all.csv"); IA = IA[IA.subset.isin(["all", "primary", "seen"])]
    L += ["", "**Tüm v2 (3066 çift) görüntü AUROC** — 'seen': kaynak belge v1 eğitiminde vardı; README zero-shot taban çizgileri: TruFor 0.599, DocTamper 0.585, insan 0.501.", "",
          IA[["model", "subset", "n_pairs", "auroc_top1pct"]].round(3).to_markdown(index=False)]
txt = "\n".join(L)
for d in (VP / "app" / "meta", VP / "hf_space" / "meta"):
    d.mkdir(parents=True, exist_ok=True); (d / "v2_tables.md").write_text(txt, encoding="utf-8")
print(txt)
