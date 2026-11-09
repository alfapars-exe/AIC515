"""Modeller arası SHAP karşılaştırması.

Kurallar
  * Tüm hedefler olasılık ölçeğinde (sigmoid) → ölçek uyumlu. Ancak HEDEF TÜRÜ farklıysa (ResNet maks patch vs
    U-Net pencere skoru) ham SHAP büyüklükleri doğrudan karşılaştırılamaz → yalnız normalize / sıralama temelli
    ölçütler (top-k% maskesi, Spearman) kullanılır ve uyarı verilir.
  * Piksel karşılaştırması aynı fiziksel bölgede yapılır: pencereler orijinal koordinata çevrilir, kesişimleri alınır.
    Ölçeklenmiş (uzun kenar > 4096) U-Net ile ölçeklenmemiş ResNet aynı ızgarada olmadığından karşılaştırılmaz.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .viz import MODEL_COLORS, fig_to_rgb


def check_scale(results):
    kinds = sorted({r.target.get("kind") for r in results})
    scales = sorted({round(float(r.target.get("scale", 1.0) or 1.0), 6) for r in results})
    same_kind = len(kinds) == 1
    msgs = ["Tüm modeller tek sigmoid çıkışlı → hedefler olasılık ölçeğinde [0, 1]."]
    if same_kind:
        msgs.append(f"Hedef türü ortak (`{kinds[0]}`) → ham SHAP değerleri doğrudan karşılaştırılabilir.")
    else:
        msgs.append(f"⚠️ Hedef türleri farklı ({', '.join(kinds)}) → ham SHAP büyüklükleri karşılaştırılmaz; "
                    "yalnız normalize (max|SHAP| = 1) haritalar ve sıralama temelli ölçütler (top-%5 IoU/Dice, Spearman).")
    if len(scales) > 1:
        msgs.append("⚠️ Pencereler farklı ölçekte (bazı U-Net girdileri küçültülmüş) → yalnız aynı ölçektekiler karşılaştırılır.")
    return dict(same_kind=same_kind, kinds=kinds, scales=scales, note=" ".join(msgs))


def to_common(results):
    """Her sonucun haritasını orijinal koordinatlardaki ortak kesişime kırp. Dönüş: (adlar, haritalar, dikdörtgen, dışlananlar)."""
    ok = [r for r in results if r.ok]
    if not ok:
        return [], [], None, []
    s0 = round(float(ok[0].target.get("scale", 1.0) or 1.0), 6)
    keep, dropped = [], []
    for r in ok:
        (keep if round(float(r.target.get("scale", 1.0) or 1.0), 6) == s0 else dropped).append(r)
    rects = []
    for r in keep:
        y, x, h, w = r.target["window"]
        rects.append((y, x, y + h, x + w))
    y0 = max(a[0] for a in rects); x0 = max(a[1] for a in rects)
    y1 = min(a[2] for a in rects); x1 = min(a[3] for a in rects)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return [], [], None, [r.label for r in ok]
    names, maps = [], []
    for r, (ry, rx, _, _) in zip(keep, rects):
        m = r.shap_map()
        maps.append(m[y0 - ry:y1 - ry, x0 - rx:x1 - rx])
        names.append(r.label)
    return names, maps, (y0, x0, y1, x1), [r.label for r in dropped]


def normalize(m):
    v = float(np.abs(m).max())
    return m / v if v > 0 else m


def topk_mask(m, frac=0.05):
    """En yüksek %frac pozitif SHAP pikselleri."""
    k = max(1, int(frac * m.size))
    thr = np.partition(m.ravel(), -k)[-k]
    return (m >= thr) & (m > 0)


def iou_dice(masks):
    n = len(masks)
    I = np.eye(n); D = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = masks[i], masks[j]
            inter = float((a & b).sum()); uni = float((a | b).sum()); s = float(a.sum() + b.sum())
            I[i, j] = I[j, i] = inter / uni if uni else np.nan
            D[i, j] = D[j, i] = 2 * inter / s if s else np.nan
    return I, D


def block_sums(m, grid=16):
    H, W = m.shape
    ys = np.linspace(0, H, grid + 1).astype(int); xs = np.linspace(0, W, grid + 1).astype(int)
    return np.array([[m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum() for j in range(grid)] for i in range(grid)]).ravel()


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float); r[order] = np.arange(len(a))
    # bağlar: ortalama sıra
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def spearman_matrix(maps, grid=16):
    B = [_rank(block_sums(m, grid)) for m in maps]
    n = len(B); S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = B[i] - B[i].mean(), B[j] - B[j].mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            S[i, j] = S[j, i] = float((a * b).sum() / den) if den > 0 else np.nan
    return S


def common_specific(masks, base_img=None):
    """Ortak bölge (tüm modellerin top-k'sı) = sarı; ≥2 model = gri; tek modele özgü = model rengi."""
    n = len(masks)
    cnt = np.sum(masks, axis=0)
    h, w = masks[0].shape
    out = np.full((h, w, 3), 255, np.uint8) if base_img is None else (base_img.astype(np.float32) * 0.45 + 140).astype(np.uint8)
    for i, m in enumerate(masks):
        out[m & (cnt == 1)] = MODEL_COLORS[i % len(MODEL_COLORS)]
    out[(cnt >= 2) & (cnt < n)] = (90, 90, 90)
    out[cnt == n] = (255, 215, 0)
    return out


def matrix_figure(M, names, title, vmin=0.0, vmax=1.0, cmap="viridis"):
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * n, 0.8 + 0.8 * n), dpi=110)
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    short = [f"{i + 1}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(short); ax.set_yticklabels(short)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "—" if not np.isfinite(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (np.isfinite(M[i, j]) and M[i, j] < (vmin + vmax) / 2) else "black")
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig_to_rgb(fig)


def compare_results(results, base_img_full=None, frac=0.05, grid=16):
    """Karşılaştırma çıktıları: IoU/Dice (top-%5 pozitif), 16×16 blok Spearman, ortak/özgü harita, normalize haritalar."""
    chk = check_scale([r for r in results if r.ok])
    names, maps, rect, dropped = to_common(results)
    out = dict(check=chk, note=chk["note"], names=names, rect=rect, dropped=dropped)
    if len(maps) < 2:
        out["note"] += " Karşılaştırma için ortak bölgede en az iki başarılı açıklama gerekir."
        return out
    nmaps = [normalize(m) for m in maps]
    masks = [topk_mask(m, frac) for m in nmaps]
    I, D = iou_dice(masks)
    S = spearman_matrix(maps, grid)
    base = None
    if base_img_full is not None and rect is not None:
        y0, x0, y1, x1 = rect
        base = base_img_full[y0:y1, x0:x1]
        if base.shape[:2] != masks[0].shape:
            base = None
    out.update(iou=I, dice=D, spearman=S, masks=masks, maps=nmaps, common_rgb=common_specific(masks, base),
               common_frac=float(np.logical_and.reduce(masks).sum() / max(1, np.logical_or.reduce(masks).sum())))
    if dropped:
        out["note"] += f" Ölçek farkı nedeniyle dışlananlar: {', '.join(dropped)}."
    return out


def comparison_figure(cmp):
    """Tek PNG: IoU · Spearman matrisleri + ortak/özgü harita + lejant."""
    if "iou" not in cmp:
        return None
    names = cmp["names"]
    a = matrix_figure(cmp["iou"], names, "Top-%5 pozitif bölge IoU")
    b = matrix_figure(cmp["spearman"], names, "16×16 blok Spearman ρ", vmin=-1, vmax=1, cmap="RdBu_r")
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=110)
    ax.imshow(cmp["common_rgb"]); ax.axis("off")
    ax.set_title(f"Ortak (sarı) / ≥2 model (gri) / modele özgü (renk) · ortak pay %{100 * cmp['common_frac']:.1f}", fontsize=7)
    for i, n in enumerate(names):
        ax.plot([], [], "s", color=np.array(MODEL_COLORS[i % len(MODEL_COLORS)]) / 255, label=f"{i + 1}: {n}")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False)
    fig.tight_layout()
    c = fig_to_rgb(fig)
    h = max(x.shape[0] for x in (a, b, c))
    pad = lambda x: np.pad(x, ((0, h - x.shape[0]), (0, 8), (0, 0)), constant_values=255)
    return np.concatenate([pad(a), pad(b), pad(c)], 1)
