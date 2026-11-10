"""XAI görüntüleyici yardımcıları (SHAP ve Kernel sekmeleri ortak):
  * en-boy oranı korunarak ekran çözünürlüğüne indirme (orijinal veriler bellekte tam çözünürlükte kalır),
  * katkı dağılımı şekli (histogram · birikimli kütle · toplamlar · GT içi/dışı),
  * yüksek çözünürlüklü dışa aktarım: natif çözünürlükte PNG'ler + 300 dpi PNG ve PDF yayın şekli (renk çubuğu, piksel
    koordinat eksenleri),
  * fare tekerleğiyle yakınlaştırma / sürükleyerek kaydırma / çift tıkla sıfırlama (ZOOM_JS) ve büyük görüntü CSS'i.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401

DISPLAY_MAX = 2400


def display(img, maxside=DISPLAY_MAX):
    if img is None:
        return None
    h, w = img.shape[:2]
    s = maxside / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def display_map(m, maxside=DISPLAY_MAX):
    """Piksel başına değer haritası → ekran boyutu (alan ortalaması; renk ölçeği göreli olduğundan görüntü aynı kalır)."""
    h, w = m.shape[:2]
    s = maxside / max(h, w)
    if s >= 1:
        return m.astype(np.float32)
    return cv2.resize(m.astype(np.float32), (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def display_mask(mask, shape_hw):
    if mask is None:
        return None
    if mask.shape[:2] == tuple(shape_hw):
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def fig_to_rgb(fig, dpi=110):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def distribution_figure(m, gt=None, fx=None, base=None, title="Açıklanan hedefe katkı dağılımı"):
    """m: piksel başına SHAP (h, w) — tam çözünürlük. Pozitif = SAHTE yönü."""
    v = m.astype(np.float64).ravel()
    pos, neg = v[v > 0], v[v < 0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    nz = v[v != 0]
    if nz.size:
        lim = np.percentile(np.abs(nz), 99.9)
        bins = np.linspace(-lim, lim, 121)
        ax.hist(np.clip(nz, -lim, lim), bins=bins, color="#888888")
        ax.hist(np.clip(pos, 0, lim), bins=bins, color="#c0392b", alpha=0.85, label=f"pozitif ({pos.size} px)")
        ax.hist(np.clip(neg, -lim, 0), bins=bins, color="#2e6fb5", alpha=0.85, label=f"negatif ({neg.size} px)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter())
        ax.legend(fontsize=8)
    ax.set_title("Piksel başına SHAP değeri histogramı", fontsize=9); ax.set_xlabel("SHAP / piksel", fontsize=8)
    ax = axs[1]
    for arr, col, lab in ((np.sort(pos)[::-1], "#c0392b", "pozitif kütle"), (np.sort(-neg)[::-1], "#2e6fb5", "negatif kütle")):
        if arr.size and arr.sum() > 0:
            cum = np.cumsum(arr) / arr.sum()
            frac = np.arange(1, arr.size + 1) / v.size
            ax.plot(frac * 100, cum * 100, color=col, label=lab)
            k = np.searchsorted(cum, 0.5)
            ax.axvline(frac[min(k, len(frac) - 1)] * 100, color=col, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("görüntü alanının yüzdesi (en güçlü piksellerden)", fontsize=8)
    ax.set_ylabel("kütlenin yüzdesi", fontsize=8); ax.set_title("Birikimli katkı kütlesi (yoğunlaşma)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[2]
    labels = ["Σ pozitif", "Σ negatif", "ΣSHAP (net)"]
    vals = [float(pos.sum()), float(neg.sum()), float(v.sum())]
    cols = ["#c0392b", "#2e6fb5", "#555555"]
    if gt is not None and gt.any():
        g = gt.ravel()
        labels += ["GT içi net", "GT dışı net"]
        vals += [float(v[g].sum()), float(v[~g].sum())]
        cols += ["#e67e22", "#7f8c8d"]
    ax.barh(labels, vals, color=cols)
    for i, x in enumerate(vals):
        ax.text(x, i, f" {x:+.4f}", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ttl = "Toplamlar"
    if fx is not None and base is not None and np.isfinite(fx) and np.isfinite(base):
        ttl += f" · f(x) − E[f] = {fx - base:+.4f}"
    ax.set_title(ttl, fontsize=9); ax.invert_yaxis()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig_to_rgb(fig)


def _sym(m, q=99.5):
    a = np.abs(m[np.isfinite(m)])
    v = float(np.percentile(a, q)) if a.size else 0.0
    return v if v > 0 else (float(a.max()) if a.size and a.max() > 0 else 1e-9)


def publication_figure(img, m, title, outdir, stem, gt=None, region=None, cmap="RdBu_r", cbar_label="SHAP (SAHTE yönü +)",
                       symmetric=True, dpi=300, maxside=3000):
    """Orijinal | analiz haritası (renk çubuklu) | bindirme — piksel koordinat eksenleriyle; 300 dpi PNG + PDF."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    H, W = img.shape[:2]
    imd, md = display(img, maxside), display_map(m, maxside)
    ext = (0, W, H, 0)
    v = _sym(m) if symmetric else None
    ar = H / W
    fig, axs = plt.subplots(1, 3, figsize=(16, max(3.2, 16 / 3 * ar) + 0.9))
    axs[0].imshow(imd, extent=ext); axs[0].set_title("Orijinal (tam görüntü)", fontsize=9)
    if symmetric:
        im = axs[1].imshow(md, cmap=cmap, vmin=-v, vmax=v, extent=ext)
    else:
        im = axs[1].imshow(md, cmap=cmap, extent=ext)
    axs[1].set_title("Analiz sonucu", fontsize=9)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03, label=cbar_label)
    axs[2].imshow(imd, extent=ext)
    if symmetric:
        w_ = np.clip(np.abs(md) / v, 0, 1)
        rgba = matplotlib.colormaps[cmap](0.5 + 0.5 * np.clip(md / v, -1, 1)); rgba[..., 3] = 0.85 * w_
    else:
        lo, hi = np.percentile(md, [1, 99.5])
        z = np.clip((md - lo) / (hi - lo + 1e-12), 0, 1)
        rgba = matplotlib.colormaps[cmap](z); rgba[..., 3] = 0.65 * z
    axs[2].imshow(rgba, extent=ext); axs[2].set_title("Orijinal üzerine bindirme", fontsize=9)
    for a in axs:
        a.set_xlabel("x (px)", fontsize=8); a.set_ylabel("y (px)", fontsize=8); a.tick_params(labelsize=7)
        for msk, col in ((gt, "#00a0ff"), (region, "#28dc28")):
            if msk is not None and np.any(msk):
                a.contour(display_mask(msk, md.shape[:2]).astype(float), levels=[0.5], colors=[col], linewidths=0.8, extent=ext,
                          origin="upper")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    png, pdf = out / f"{stem}_yayin_300dpi.png", out / f"{stem}_yayin.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(png), str(pdf)]


def export_native(outdir, stem, panels: dict, arrays: dict = None):
    """panels: {ad: RGB uint8 (natif çözünürlük)} → PNG; arrays: {ad: float ndarray} → NPY (float32)."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    files = []
    for k, im in panels.items():
        if im is None:
            continue
        p = out / f"{stem}_{k}.png"
        cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.ndim == 3 else im)
        files.append(str(p))
    for k, a in (arrays or {}).items():
        p = out / f"{stem}_{k}.npy"
        np.save(p, np.asarray(a, np.float32)); files.append(str(p))
    return files


def new_export_dir(tag):
    return Path(tempfile.mkdtemp(prefix=f"ai515_{tag}_"))


# Büyük, en-boy korumalı görüntü kutuları; yakınlaştırılabilir görüntülerde imleç ipucu
ZOOM_CSS = """
.xai-zoom .image-frame, .xai-zoom [data-testid="image"] {overflow:hidden !important;}
.xai-zoom img {object-fit:contain !important; cursor:zoom-in; user-select:none; -webkit-user-drag:none;}
.xai-zoom-note {font-size:.8em; opacity:.7}
"""

# Fare tekerleği: imleç noktası çevresinde yakınlaştırma (1–32×) · sürükleme: kaydırma · çift tık: sıfırlama.
# Gradio görüntüyü yenilediğinde (yeni src) dönüşüm sıfırlanır. Tam ekran için bileşenin kendi düğmesi kullanılır.
ZOOM_JS = """
() => {
  if (window.__xaiZoom) return; window.__xaiZoom = true;
  const init = (img) => {
    if (img.dataset.xz) return; img.dataset.xz = '1';
    let s = 1, tx = 0, ty = 0, drag = null;
    const box = img.parentElement; if (!box) return;
    box.style.overflow = 'hidden';
    const apply = () => { img.style.transformOrigin = '0 0'; img.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
                          img.style.cursor = s > 1 ? (drag ? 'grabbing' : 'grab') : 'zoom-in'; };
    const reset = () => { s = 1; tx = 0; ty = 0; apply(); };
    box.addEventListener('wheel', (e) => {
      if (!e.target || e.target !== img) return;
      e.preventDefault();
      const r = img.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const ns = Math.min(32, Math.max(1, s * (e.deltaY < 0 ? 1.25 : 0.8)));
      const k = ns / s; tx -= px * (k - 1); ty -= py * (k - 1); s = ns;
      if (s === 1) { tx = 0; ty = 0; }
      apply();
    }, {passive: false});
    img.addEventListener('mousedown', (e) => { if (s <= 1) return; e.preventDefault(); drag = {x: e.clientX, y: e.clientY}; apply(); });
    window.addEventListener('mousemove', (e) => { if (!drag) return; tx += e.clientX - drag.x; ty += e.clientY - drag.y;
                                                  drag = {x: e.clientX, y: e.clientY}; apply(); });
    window.addEventListener('mouseup', () => { if (drag) { drag = null; apply(); } });
    img.addEventListener('dblclick', reset);
    img.addEventListener('load', reset);
  };
  const scan = () => document.querySelectorAll('.xai-zoom img').forEach(init);
  new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
  scan();
}
"""
