"""appcore.scenarios — girdi senaryoları, bozulmalar (JPEG, ölçekleme, bulanıklık, CLAHE, gri), kırpma ve montaj."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .imaging import (  # noqa: F401
    K3, KX, clip_u8, derive_pixel_gt, green_marker_mask
)

# ================================================================== GİRDİ SENARYOLARI
INPUT_TYPES = ["Sahte belge (forged)", "Authentic eş (negatif kontrol)", "Sahte, yeşil işaretçi onarılmış",
               "Authentic + yalnız yeşil işaretçi (kısayol testi)",
               "Yalnız yeniden-render (karakterler geri alınmış · Faz 3 sondası)",
               "Yalnız içerik (render ve işaretçi yok · Faz 3 sondası)"]


DEGRADATIONS = ["Yok (ham PNG)", "JPEG Q75", "JPEG Q50", "0.75× yeniden ölçekleme", "0.5× yeniden ölçekleme",
                "Gauss bulanıklık σ=1", "Gauss gürültüsü σ=5", "CLAHE (L kanalı, Faz 3)", "Gri seviye (Faz 3)"]


def degrade(img, kind):
    if kind.startswith("JPEG"):
        q = int(kind.split("Q")[1])
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if "yeniden ölçekleme" in kind:
        f = float(kind.split("×")[0])
        h, w = img.shape[:2]
        s = cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if "bulanıklık" in kind:
        return cv2.GaussianBlur(img, (0, 0), 1.0)
    if "gürültü" in kind:
        return clip_u8(img.astype(np.float32) + np.random.default_rng(0).normal(0, 5, img.shape))
    if kind.startswith("CLAHE"):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if kind.startswith("Gri"):
        return np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    return img


def marker_ops(forged, auth):
    gm = green_marker_mask(forged) & ~green_marker_mask(auth)
    gm = cv2.dilate(gm.astype(np.uint8), K3, iterations=1) > 0
    repaired = forged.copy(); repaired[gm] = auth[gm]
    transplanted = auth.copy(); transplanted[gm] = forged[gm]
    return gm, repaired, transplanted


def build_input(forged, auth, input_type):
    """Dönüş: (girdi, gerçek etiket 1=sahte/0=gerçek, not). Senaryolar PHASE 2 Ek-C sondasıyla aynıdır."""
    if input_type == INPUT_TYPES[0]:
        return forged, 1, "sahte belge olduğu gibi"
    if auth is None:
        raise gr.Error("Bu senaryo aynı boyutta bir authentic eş gerektirir; 'eşli' örnek seçin (filtre: 'yalnız authentic eşi olanlar').")
    if input_type == INPUT_TYPES[1]:
        return auth, 0, "orijinal (değiştirilmemiş) belge"
    gm, rep, tra = marker_ops(forged, auth)
    if input_type in INPUT_TYPES[4:6]:  # Faz 3 bölge-görünümü sondası (probes_p3.py tanımı; tek fark: işaretçi kalıntısı bbox halkasıyla sınırlanmaz)
        r = derive_pixel_gt(forged, auth)
        if r is None:
            raise gr.Error("Bu senaryo aynı boyutta bir authentic eş gerektirir.")
        pg = r[0]
        if input_type == INPUT_TYPES[4]:
            sel = cv2.dilate((pg | gm).astype(np.uint8), KX, iterations=2) > 0  # KX: 4-komşu haç = ndi.binary_dilation varsayılanı
            out = forged.copy(); out[sel] = auth[sel]
            return out, 0, (f"sahte belge; değişen karakterler ve işaretçi ({int(sel.sum())} px) authentic'e döndürüldü — "
                            "yalnız üreticinin alanı yeniden çizme izi kalır, içerik değişmedi")
        sel = (cv2.dilate((pg & ~gm).astype(np.uint8), KX, iterations=2) > 0) & ~gm
        out = auth.copy(); out[sel] = forged[sel]
        return out, 1, (f"orijinal belgeye yalnız değişen karakterler ({int(sel.sum())} px) yapıştırıldı — "
                        "yeniden-render izi ve işaretçi yok, içerik sahte")
    if input_type == INPUT_TYPES[2]:
        return rep, 1, f"sahte belge, {int(gm.sum())} px işaretçi authentic'ten geri kopyalandı (içerik sahte kalır)"
    return tra, 0, f"orijinal belge + yalnız {int(gm.sum())} px yeşil işaretçi (hiçbir karakter değişmedi)"


def bbox_of(mask):
    if mask is None:
        return None
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def zoom_box(bb, shape, pad_frac=0.8, min_side=180):
    H, W = shape[:2]
    if bb is None:
        cx, cy, bw, bh = W // 2, H // 2, min(W, 600), min(H, 400)
    else:
        cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    hw = max(min_side, int(bw * (1 + pad_frac))) // 2; hh = max(min_side // 2, int(bh * (1 + 2 * pad_frac))) // 2
    hw = max(hw, hh * 3 // 2)
    x0, x1 = max(0, cx - hw), min(W, cx + hw); y0, y1 = max(0, cy - hh), min(H, cy + hh)
    return x0, y0, x1, y1


def montage(panels, zb, title):
    """Yakın plan paneli: (başlık, görüntü) listesi → tek görüntü (matplotlib; Türkçe karakter destekli)."""
    x0, y0, x1, y1 = zb
    n = len(panels); cols = min(4, n); rows = int(math.ceil(n / cols))
    aspect = (x1 - x0) / max(1, (y1 - y0))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows / max(aspect, 0.6) + 0.6), dpi=110, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (t, im) in zip(axes.ravel(), panels):
        crop = im[y0:y1, x0:x1]
        if crop.size == 0:  # farklı boyutlu panel (güvenlik): tüm görüntüyü göster
            crop = im
        ax.imshow(crop, cmap="gray" if crop.ndim == 2 else None, interpolation="nearest")
        ax.set_title(t, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def show_df(df, nd=3):
    """Arayüz için: sayıları yuvarla, boşları '—' yap (CSV çıktıları ham kalır)."""
    if df is None or not len(df):
        return df
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].round(nd)
    return d.astype(object).where(pd.notna(d), "—")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
