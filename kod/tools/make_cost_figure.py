"""make_cost_figure.py — Maliyet–doğruluk şekli (rapor için okunaklı sürüm).

Kaynaklar (yalnız okunur): phase3/results/speed_bench.csv (ms/görüntü, parametre) ve
phase3_yolo/results/detection_test.csv (resmî test, group == "all", kutu AP50).
Çıktı: reports/figures/fig_maliyet_ap.png
(phase3/figures/fig_cost_ap.png aynı veriyi log eksenli saçılım olarak verir; etiketleri üst üste bindiği için
raporda bu çubuk biçimi kullanılır. Değerler birebir aynıdır.)
"""
import sys
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

NAMES = {
    "unet_bbox_s42": "U-Net (kutu referansı)",
    "dlv3p_bbox_s42": "DeepLabV3+ (kutu referansı)",
    "unetjoint_pixel_eps8_morph_s42": "U-Net birleşik kayıp (piksel + kutu)",
    "aiforge-tiles-yolo11n-img1024-conserv-s42": "YOLO11n · kare 1024",
    "aiforge-tiles-yolo11s-img1024-conserv-s42": "YOLO11s · kare 1024",
    "aiforge-tiles-yolo11m-img1024-conserv-s42": "YOLO11m · kare 1024",
    "aiforge-full-yolo11s-img1280-conserv-s42": "YOLO11s · tam görüntü 1280",
}
C_SEG, C_YOLO = "#4C72B0", "#DD8452"

sp = pd.read_csv(ROOT / "phase3" / "results" / "speed_bench.csv")
det = pd.read_csv(ROOT / "phase3_yolo" / "results" / "detection_test.csv")
det = det[det.group == "all"].set_index("model")
df = sp[sp.model.isin(NAMES)].copy()
df["ap50"] = df.model.map(det.AP50)
assert df.ap50.notna().all() and len(df) == len(NAMES), "kaynak tablolarda eksik model"
df["ad"] = df.model.map(NAMES)
df = df.sort_values(["family", "ms_per_img"], ascending=[False, True]).reset_index(drop=True)
col = [C_YOLO if f == "yolo" else C_SEG for f in df.family]

fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), dpi=300, sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 0.8], wspace=0.08))
y = range(len(df))
for ax, colname, title, fmt, xmax in ((axes[0], "ap50", "Kutu AP50 (resmî test)", "{:.3f}", 1.12),
                                      (axes[1], "ms_per_img", "Çıkarım süresi (ms / görüntü)", "{:.0f}", df.ms_per_img.max() * 1.22),
                                      (axes[2], "params_M", "Parametre (milyon)", "{:.1f}", df.params_M.max() * 1.25)):
    ax.barh(list(y), df[colname], color=col, height=0.62)
    for i, v in enumerate(df[colname]):
        ax.text(v + xmax * 0.015, i, fmt.format(v), va="center", fontsize=8)
    ax.set_xlim(0, xmax); ax.set_title(title, fontsize=9.5); ax.tick_params(labelsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(list(y)); axes[0].set_yticklabels(df.ad, fontsize=8.5); axes[0].invert_yaxis()
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_YOLO), plt.Rectangle((0, 0), 1, 1, color=C_SEG)],
           labels=["kutu dedektörü (YOLO11)", "bölütleme modeli (U-Net, DeepLabV3+)"], fontsize=8.5, loc="lower center",
           ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.08))
out = ROOT / "reports" / "figures" / "fig_maliyet_ap.png"
fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(out)
print(df[["ad", "ap50", "ms_per_img", "params_M"]].round(3).to_string(index=False))
