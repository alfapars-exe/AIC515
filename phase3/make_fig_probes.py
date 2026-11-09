"""make_fig_probes.py — Bildiri Şekil: sonda koşullarının görsel tanımı (TR). Bir val çifti (işaretçili, Gemini, 1–5 % alan)
için bbox çevresinde kırpılmış 6 panel: forged | authentic | forged-repaired | authentic+marker | render-only | content-only.
Çıktı: phase3/figures/fig_probe_conditions.png (300 dpi, IEEE tek sütun genişliği 3.5 in)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os; os.environ.setdefault("P3_NO_CUDA", "1")
import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import p3ext as X
D = X.D
tab = D.split_table("val")
cand = tab[(tab.tool == "gemini") & (tab.area_bin == "1-5%")]
FIG = X.HERE / "figures"; FIG.mkdir(exist_ok=True)
for nid in cand.index:
    f = D.read_rgb(D.img_path(nid, "val")); a = D.read_rgb(D.auth_path(tab.loc[nid, "pair_name"], "val"))
    if f.shape != a.shape or max(f.shape) > 1600: continue
    mb = D.read_mask(D.bbox_mask_path(nid, "val")) > 0
    pg = D.read_gt(nid, "pixel_eps8_morph", "val", shape=mb.shape); pgn = D.read_gt(nid, "pixel_eps8_morph_nm", "val", shape=mb.shape)
    if pg is None or pgn is None: continue
    ring = ndi.binary_dilation(mb, iterations=6) & ~ndi.binary_erosion(mb, iterations=6)
    res = X.gmask(f) & ~X.gmask(a) & ring
    if res.sum() < 150: continue
    resd = ndi.binary_dilation(res, iterations=1); pg, pgn = pg > 0, pgn > 0
    rep = f.copy(); rep[resd] = a[resd]
    am = a.copy(); am[resd] = f[resd]
    sr = ndi.binary_dilation(pg | resd, iterations=2); ro = f.copy(); ro[sr] = a[sr]
    sc = ndi.binary_dilation(pgn, iterations=2) & ~resd; co = a.copy(); co[sc] = f[sc]
    x0, y0, x1, y1 = D.mask_bbox(mb.astype(np.uint8)); m = 24
    ys, xs = slice(max(0, y0 - m), y1 + m), slice(max(0, x0 - m), x1 + m)
    panels = [("forged", f), ("authentic", a), ("forged-repaired", rep), ("authentic+marker", am), ("render-only", ro), ("content-only", co)]
    fig, ax = plt.subplots(3, 2, figsize=(3.5, 2.6), dpi=300)
    for k, (t, im) in enumerate(panels):
        A = ax[k // 2, k % 2]; A.imshow(im[ys, xs]); A.set_title(t, fontsize=7, pad=2); A.axis("off")
        rect = plt.Rectangle((x0 - xs.start - 0.5, y0 - ys.start - 0.5), x1 - x0, y1 - y0, fill=False, ec="#d62728", lw=0.5, ls="--")
        A.add_patch(rect)
    fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.2)
    fig.savefig(FIG / "fig_probe_conditions.png", dpi=300); print("example", nid, "residue px", int(res.sum())); break
