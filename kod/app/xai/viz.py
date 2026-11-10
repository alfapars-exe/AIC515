"""Görselleştirme: SHAP ısı haritası (simetrik ölçekli ıraksak RdBu), bindirme, pozitif/negatif katkı,
SAHTE/GERÇEK sınıf panelleri, bölge kütlesi grafiği, konum görüntüsü. Tüm çıktılar RGB uint8."""
from __future__ import annotations
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_COLOR = (0, 160, 255)       # mavi — uygulamadaki GT rengi
MARKER_COLOR = (255, 0, 255)   # magenta — uygulamadaki işaretçi rengi
MODEL_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
                (240, 50, 230), (210, 245, 60), (128, 128, 0), (0, 128, 128)]


def sym_vmax(m, q=99.5):
    a = np.abs(m[np.isfinite(m)])
    v = float(np.percentile(a, q)) if a.size else 0.0
    return v if v > 0 else (float(a.max()) if a.size and a.max() > 0 else 1e-9)


def _cmap(name, x):
    return (matplotlib.colormaps[name](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def diverging(m, vmax=None):
    v = vmax or sym_vmax(m)
    return _cmap("RdBu_r", 0.5 + 0.5 * m / v)


def positive(m, vmax=None):
    v = vmax or sym_vmax(m)
    return _cmap("Reds", np.clip(m, 0, None) / v)


def negative(m, vmax=None):
    v = vmax or sym_vmax(m)
    return _cmap("Blues", np.clip(-m, 0, None) / v)


def blend(img, heat, alpha=0.55, weight=None):
    img = img.astype(np.float32); heat = heat.astype(np.float32)
    if weight is None:
        return (img * (1 - alpha) + heat * alpha).astype(np.uint8)
    w = (alpha * weight)[..., None]
    return (img * (1 - w) + heat * w).astype(np.uint8)


def contour(img, mask, color, th=2):
    if mask is None or not np.any(mask):
        return img
    out = img.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, color, th)
    return out


def annotate(img, gt=None, marker=None, region=None):
    out = contour(img, gt, GT_COLOR)
    if marker is not None and marker.any():
        out = out.copy(); out[cv2.dilate(marker.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = MARKER_COLOR
    if region is not None:
        out = contour(out, region, (40, 220, 40), 1)
    return out


def side_by_side(a, b, gap=6):
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0)), constant_values=255)
    return np.concatenate([pad(a), np.full((h, gap, 3), 255, np.uint8), pad(b)], 1)


def panels(result, win_img, gt_win=None, marker_win=None, region=None):
    """Sabit panel seti. win_img: açıklanan pencere (uint8 RGB); region: hedef bölge (yeşil kontur)."""
    m = result.shap_map()
    v = sym_vmax(m)
    heat = diverging(m, v)
    weight = np.clip(np.abs(m) / v, 0, 1)
    fake = annotate(diverging(m, v), gt_win, None, region)
    real = annotate(diverging(-m, v), gt_win, None, region)
    return {
        "Orijinal (açıklanan pencere)": annotate(win_img, gt_win, marker_win, region),
        "SHAP ısı haritası (kırmızı: SAHTE yönü · mavi: GERÇEK yönü)": annotate(heat, gt_win, None, region),
        "Bindirme": annotate(blend(win_img, heat, 0.75, weight), gt_win, None, region),
        "Pozitif katkı (SAHTE'yi artıran)": annotate(positive(m, v), gt_win, None, region),
        "Negatif katkı (GERÇEK'e iten)": annotate(negative(m, v), gt_win, None, region),
        "Sınıf panelleri: SAHTE | GERÇEK (= −SAHTE)": side_by_side(fake, real),
    }


def region_masses(m, gt_win=None, marker_win=None, ring_px=6):
    """Pozitif SHAP kütlesinin dağılımı: GT kutusu içi / kutu kenar halkası (±6 px, işaretçinin bulunduğu yer) /
    yeşil işaretçi pikselleri / geri kalan. Oranlar pozitif toplam kütleye göredir."""
    pos = np.clip(m, 0, None)
    tot = float(pos.sum())
    out = {}
    if tot <= 0:
        return {"pozitif kütle yok": 1.0}
    if gt_win is not None and gt_win.any():
        g = gt_win.astype(np.uint8)
        k = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
        ring = (cv2.dilate(g, k) > 0) & ~(cv2.erode(g, k) > 0)
        inside = (g > 0) & ~ring
        out["GT kutusu içi"] = float(pos[inside].sum() / tot)
        out["kutu kenar halkası (±6 px)"] = float(pos[ring].sum() / tot)
        out["belgenin geri kalanı"] = float(pos[~(inside | ring)].sum() / tot)
    else:
        out["tüm pencere (GT yok)"] = 1.0
    if marker_win is not None and marker_win.any():
        mk = cv2.dilate(marker_win.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out["↳ yeşil işaretçi pikselleri (±1 px)"] = float(pos[mk].sum() / tot)
    return out


def fig_to_rgb(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def mass_chart(masses_by_model: dict, title="Pozitif SHAP kütlesinin bölgelere dağılımı"):
    """masses_by_model: {model: {bölge: oran}} → yatay gruplu çubuk grafik."""
    if not masses_by_model:
        return None
    cats = []
    for d in masses_by_model.values():
        for c in d:
            if c not in cats:
                cats.append(c)
    names = list(masses_by_model)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(cats) * max(1, len(names)) ** 0.6 + 1.4), dpi=110)
    hgt = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        vals = [100 * masses_by_model[n].get(c, 0.0) for c in cats]
        col = np.array(MODEL_COLORS[i % len(MODEL_COLORS)]) / 255
        ys = np.arange(len(cats)) + i * hgt
        ax.barh(ys, vals, height=hgt * 0.92, color=col, label=n)
        for y, v in zip(ys, vals):
            ax.text(v + 0.6, y, f"%{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(np.arange(len(cats)) + 0.4 - hgt / 2); ax.set_yticklabels(cats, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("pozitif kütle payı (%)"); ax.set_xlim(0, 105)
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_rgb(fig)


def location_image(rgb, results, maxside=1100):
    """Tam görüntü üzerinde açıklanan pencerelerin konumu (orijinal koordinatlar)."""
    out = rgb.copy()
    th = max(2, out.shape[1] // 400)
    for i, r in enumerate(results):
        y, x, h, w = r.target["window"]
        s = r.target.get("scale", 1.0) or 1.0
        y0, x0, y1, x1 = int(y / s), int(x / s), int((y + h) / s), int((x + w) / s)
        col = MODEL_COLORS[i % len(MODEL_COLORS)]
        cv2.rectangle(out, (x0, y0), (min(x1, out.shape[1]) - 1, min(y1, out.shape[0]) - 1), col, th)
        cv2.putText(out, str(i + 1), (x0 + 4, y0 + 18 + 4 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.6 + 0.2 * th, col, th)
    h, w = out.shape[:2]
    s = maxside / max(h, w)
    if s < 1:
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out


def summary_figure(result, win_img, gt_win=None, marker_win=None):
    """Dışa aktarım için renk çubuklu özet şekil (orijinal · SHAP · bindirme)."""
    m = result.shap_map(); v = sym_vmax(m)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), dpi=110)
    axs[0].imshow(annotate(win_img, gt_win, marker_win)); axs[0].set_title("Pencere (mavi: GT · magenta: işaretçi)", fontsize=8)
    im = axs[1].imshow(m, cmap="RdBu_r", vmin=-v, vmax=v); axs[1].set_title("SHAP (SAHTE yönü +)", fontsize=8)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
    axs[2].imshow(blend(win_img, diverging(m, v), 0.75, np.clip(np.abs(m) / v, 0, 1))); axs[2].set_title("Bindirme", fontsize=8)
    for a in axs:
        a.axis("off")
    fig.suptitle(f"{result.label} · {result.method} · ref: {result.reference} · f(x)={result.fx:.3f} · E[f]={result.base:.3f} · "
                 f"ΣSHAP={result.sum_shap:.3f}", fontsize=8)
    fig.tight_layout()
    return fig_to_rgb(fig)
