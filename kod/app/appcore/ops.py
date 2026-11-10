"""appcore.ops — ön işleme operasyon kaydı (OPS, 13 kategori) ve hazır hatlar (PRESETS)."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    IMAGENET_MEAN, IMAGENET_STD
)
from .imaging import (  # noqa: F401
    K3, block_dct_image, clip_u8, fft_spectrum_image, freq_filter, green_marker_mask, odd, project_freq_channels,
    to_gray, to_rgb, to_u8
)

# ================================================================== ÖN İŞLEME OPERASYON KAYDI
# Her op: (fonksiyon, [(param_adı, etiket, min, max, adım, varsayılan)] (≤3), mod_seçenekleri | None, açıklama)
# fonksiyon imzası: f(img_u8, p1, p2, p3, mode) -> (out_u8, bilgi_str)
OPS: dict[str, dict] = {}


def op(cat, name, params=(), modes=None, doc=""):
    def deco(fn):
        OPS.setdefault(cat, {})[name] = dict(fn=fn, params=list(params), modes=modes, doc=doc)
        return fn
    return deco


# ---- 1) Renk işleme (Hafta 1)
@op("1 · Renk işleme", "Gri tonlama (luminance)", doc="Y = 0.299R + 0.587G + 0.114B")
def _gray(img, p1, p2, p3, mode):
    return to_gray(img), "RGB → gri (luminance)"


@op("1 · Renk işleme", "Kanal seç", modes=["R", "G", "B", "H (HSV)", "S (HSV)", "V (HSV)", "L (Lab)", "a (Lab)", "b (Lab)"])
def _channel(img, p1, p2, p3, mode):
    rgb = to_rgb(img)
    if mode in ("R", "G", "B"):
        return rgb[..., "RGB".index(mode)], f"{mode} kanalı"
    if "HSV" in mode:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return hsv[..., "HSV".index(mode[0])], f"HSV {mode[0]} kanalı"
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[..., "Lab".index(mode[0])], f"Lab {mode[0]} kanalı"


@op("1 · Renk işleme", "Renk kuantalama (K-Means renk)", [("k", "K (renk sayısı)", 2, 16, 1, 4)])
def _quant(img, p1, p2, p3, mode):
    rgb = to_rgb(img)
    Z = rgb.reshape(-1, 3).astype(np.float32)
    k = int(p1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(Z, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    out = cen[lab.ravel()].reshape(rgb.shape)
    return clip_u8(out), f"{k} renge kuantalandı"


@op("1 · Renk işleme", "Yeşil işaretçi maskesi (proje artefaktı)", doc="G − max(R,B) > 50 ve G > 110 (phase2 probu)")
def _green(img, p1, p2, p3, mode):
    m = green_marker_mask(to_rgb(img))
    return (m * 255).astype(np.uint8), f"yeşil işaretçi pikseli: {int(m.sum())}"


# ---- 2) Yoğunluk dönüşümleri (Hafta 2)
@op("2 · Yoğunluk dönüşümü", "Negatif")
def _neg(img, p1, p2, p3, mode):
    return 255 - img, "s = 255 − r"


@op("2 · Yoğunluk dönüşümü", "Log dönüşümü", [("c", "c (ölçek)", 0.1, 3.0, 0.05, 1.0)])
def _log(img, p1, p2, p3, mode):
    x = img.astype(np.float32)
    out = p1 * 255.0 / np.log1p(255.0) * np.log1p(x)
    return clip_u8(out), f"s = c·log(1+r), c={p1:.2f}"


@op("2 · Yoğunluk dönüşümü", "Gamma (güç yasası)", [("gamma", "γ", 0.1, 5.0, 0.05, 1.0)])
def _gamma(img, p1, p2, p3, mode):
    x = img.astype(np.float32) / 255.0
    return clip_u8(255.0 * np.power(x, p1)), f"s = r^γ, γ={p1:.2f}"


@op("2 · Yoğunluk dönüşümü", "Kontrast germe (yüzdelik)", [("lo", "alt yüzdelik", 0, 20, 0.5, 1), ("hi", "üst yüzdelik", 80, 100, 0.5, 99)])
def _stretch(img, p1, p2, p3, mode):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, p1), np.percentile(x, p2)
    out = (x - lo) / max(hi - lo, 1e-6) * 255.0
    return clip_u8(out), f"[{lo:.0f},{hi:.0f}] → [0,255]"


@op("2 · Yoğunluk dönüşümü", "Yoğunluk dilimleme", [("a", "alt", 0, 255, 1, 100), ("b", "üst", 0, 255, 1, 180)],
    modes=["dilim beyaz, diğerleri siyah", "dilim beyaz, diğerleri korunur"])
def _slice(img, p1, p2, p3, mode):
    g = to_gray(img)
    sel = (g >= p1) & (g <= p2)
    if mode.startswith("dilim beyaz, diğerleri siyah"):
        return (sel * 255).astype(np.uint8), f"dilim [{int(p1)},{int(p2)}]"
    out = g.copy(); out[sel] = 255
    return out, f"dilim [{int(p1)},{int(p2)}] beyazlatıldı"


@op("2 · Yoğunluk dönüşümü", "Bit düzlemi", [("bit", "bit (0=LSB)", 0, 7, 1, 7)])
def _bitplane(img, p1, p2, p3, mode):
    g = to_gray(img)
    return (((g >> int(p1)) & 1) * 255).astype(np.uint8), f"bit düzlemi {int(p1)}"


# ---- 3) Histogram işleme
@op("3 · Histogram", "Histogram eşitleme (global)")
def _he(img, p1, p2, p3, mode):
    if img.ndim == 2:
        return cv2.equalizeHist(img), "global eşitleme (gri)"
    ycc = cv2.cvtColor(to_rgb(img), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), "Y kanalında global eşitleme"


@op("3 · Histogram", "CLAHE", [("clip", "clip limit", 0.5, 10, 0.5, 2.0), ("tile", "tile boyutu", 2, 32, 1, 8)])
def _clahe(img, p1, p2, p3, mode):
    cl = cv2.createCLAHE(clipLimit=float(p1), tileGridSize=(int(p2), int(p2)))
    if img.ndim == 2:
        return cl.apply(img), f"CLAHE clip={p1}, tile={int(p2)}"
    ycc = cv2.cvtColor(to_rgb(img), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cl.apply(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), f"CLAHE (Y) clip={p1}, tile={int(p2)}"


# ---- 4) Eşikleme (Hafta 6)
@op("4 · Eşikleme", "Global eşik", [("T", "T", 0, 255, 1, 128)], modes=["binary", "binary_inv"])
def _thr(img, p1, p2, p3, mode):
    g = to_gray(img)
    t = cv2.THRESH_BINARY if mode == "binary" else cv2.THRESH_BINARY_INV
    _, out = cv2.threshold(g, int(p1), 255, t)
    return out, f"T={int(p1)} ({mode})"


@op("4 · Eşikleme", "Otsu", modes=["binary", "binary_inv"])
def _otsu(img, p1, p2, p3, mode):
    g = to_gray(img)
    t = cv2.THRESH_BINARY if mode == "binary" else cv2.THRESH_BINARY_INV
    T, out = cv2.threshold(g, 0, 255, t + cv2.THRESH_OTSU)
    return out, f"Otsu T*={T:.0f}"


@op("4 · Eşikleme", "Adaptif eşik", [("block", "blok boyutu (tek)", 3, 101, 2, 31), ("C", "C", -30, 30, 1, 10)],
    modes=["gaussian", "mean"])
def _adapt(img, p1, p2, p3, mode):
    g = to_gray(img)
    m = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if mode == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(g, 255, m, cv2.THRESH_BINARY, odd(p1), float(p2)), f"adaptif {mode}, blok={odd(p1)}, C={p2}"


@op("4 · Eşikleme", "Sauvola (belge ikilileştirme)", [("w", "pencere", 5, 75, 2, 25), ("k", "k", 0.05, 0.5, 0.01, 0.2)],
    doc="GT türetiminde kullanılan belge ikilileştirmesi (make_gt.py: w=25, k=0.2)")
def _sauvola(img, p1, p2, p3, mode):
    g = to_gray(img)
    th = threshold_sauvola(g, window_size=odd(p1), k=float(p2))
    return ((g > th) * 255).astype(np.uint8), f"Sauvola w={odd(p1)}, k={p2:.2f} (metin siyah)"


# ---- 5) Gürültü ve gürültü giderme (Hafta 3, 5)
@op("5 · Gürültü / giderme", "Gürültü ekle", [("sigma", "Gauss σ / tuz-biber oranı(%)", 0, 50, 1, 10)],
    modes=["gauss", "tuz-biber", "speckle"])
def _noise(img, p1, p2, p3, mode):
    rng = np.random.default_rng(0)
    x = img.astype(np.float32)
    if mode == "gauss":
        out = x + rng.normal(0, p1, x.shape)
    elif mode == "speckle":
        out = x + x * rng.normal(0, p1 / 100.0, x.shape)
    else:
        out = x.copy()
        p = p1 / 100.0
        m = rng.random(x.shape[:2])
        out[m < p / 2] = 0; out[m > 1 - p / 2] = 255
    return clip_u8(out), f"{mode} gürültü, param={p1}"


@op("5 · Gürültü / giderme", "Ortalama (box) filtre", [("k", "çekirdek", 3, 31, 2, 5)])
def _box(img, p1, p2, p3, mode):
    return cv2.blur(img, (odd(p1), odd(p1))), f"box {odd(p1)}×{odd(p1)}"


@op("5 · Gürültü / giderme", "Medyan filtre", [("k", "çekirdek", 3, 31, 2, 5)])
def _median(img, p1, p2, p3, mode):
    return cv2.medianBlur(img, odd(p1)), f"medyan {odd(p1)}×{odd(p1)}"


@op("5 · Gürültü / giderme", "Gauss filtre", [("k", "çekirdek", 3, 31, 2, 5), ("sigma", "σ", 0.1, 10, 0.1, 1.0)])
def _gauss(img, p1, p2, p3, mode):
    return cv2.GaussianBlur(img, (odd(p1), odd(p1)), float(p2)), f"Gauss k={odd(p1)}, σ={p2:.1f}"


@op("5 · Gürültü / giderme", "Bilateral filtre", [("d", "d", 3, 25, 2, 9), ("sc", "σ renk", 5, 200, 5, 75), ("ss", "σ uzay", 5, 200, 5, 75)])
def _bilat(img, p1, p2, p3, mode):
    return cv2.bilateralFilter(img, int(p1), float(p2), float(p3)), f"bilateral d={int(p1)}, σc={p2}, σs={p3}"


@op("5 · Gürültü / giderme", "Non-local means", [("h", "h", 1, 40, 1, 10)])
def _nlm(img, p1, p2, p3, mode):
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, float(p1), 7, 21), f"NLM h={p1}"
    return cv2.fastNlMeansDenoisingColored(to_rgb(img), None, float(p1), float(p1), 7, 21), f"NLM renkli h={p1}"


# ---- 6) Keskinleştirme (Hafta 3)
@op("6 · Keskinleştirme", "Laplacian keskinleştirme", [("c", "c (ağırlık)", 0.1, 3.0, 0.1, 1.0)], doc="g = f − c·∇²f")
def _lapsharp(img, p1, p2, p3, mode):
    x = img.astype(np.float32)
    lap = cv2.Laplacian(x, cv2.CV_32F, ksize=1)
    return clip_u8(x - p1 * lap), f"f − {p1:.1f}·∇²f"


@op("6 · Keskinleştirme", "Unsharp mask / high-boost", [("sigma", "σ (bulanıklık)", 0.5, 10, 0.5, 2.0), ("k", "k (kuvvet; >1 high-boost)", 0.1, 5.0, 0.1, 1.0)])
def _unsharp(img, p1, p2, p3, mode):
    x = img.astype(np.float32)
    bl = cv2.GaussianBlur(x, (0, 0), float(p1))
    return clip_u8(x + p2 * (x - bl)), f"f + {p2:.1f}·(f − G_σ={p1}*f)"


# ---- 7) Kenar tespiti
@op("7 · Kenar tespiti", "Sobel", [("k", "çekirdek", 1, 7, 2, 3)], modes=["büyüklük", "x", "y"])
def _sobel(img, p1, p2, p3, mode):
    g = to_gray(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=odd(p1)); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=odd(p1))
    out = {"x": np.abs(gx), "y": np.abs(gy)}.get(mode, np.hypot(gx, gy))
    return to_u8(out), f"Sobel k={odd(p1)} ({mode})"


@op("7 · Kenar tespiti", "Prewitt", modes=["büyüklük", "x", "y"])
def _prewitt(img, p1, p2, p3, mode):
    g = to_gray(img).astype(np.float32)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32); ky = kx.T
    gx = cv2.filter2D(g, cv2.CV_32F, kx); gy = cv2.filter2D(g, cv2.CV_32F, ky)
    out = {"x": np.abs(gx), "y": np.abs(gy)}.get(mode, np.hypot(gx, gy))
    return to_u8(out), f"Prewitt ({mode})"


@op("7 · Kenar tespiti", "Scharr")
def _scharr(img, p1, p2, p3, mode):
    g = to_gray(img).astype(np.float32)
    return to_u8(np.hypot(cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1))), "Scharr büyüklük"


@op("7 · Kenar tespiti", "Laplacian", [("k", "çekirdek", 1, 7, 2, 3)])
def _lap(img, p1, p2, p3, mode):
    g = to_gray(img).astype(np.float32)
    return to_u8(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=odd(p1)))), f"|∇²f| k={odd(p1)}"


@op("7 · Kenar tespiti", "LoG (Laplacian of Gaussian)", [("sigma", "σ", 0.5, 8, 0.5, 2.0)])
def _log_edge(img, p1, p2, p3, mode):
    g = cv2.GaussianBlur(to_gray(img).astype(np.float32), (0, 0), float(p1))
    return to_u8(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1))), f"LoG σ={p1}"


@op("7 · Kenar tespiti", "Canny", [("t1", "alt eşik", 0, 255, 1, 50), ("t2", "üst eşik", 0, 255, 1, 150)])
def _canny(img, p1, p2, p3, mode):
    return cv2.Canny(to_gray(img), int(p1), int(p2)), f"Canny ({int(p1)},{int(p2)})"


# ---- 8) Morfoloji (Hafta 6)
_MORPH = {"Erozyon": cv2.MORPH_ERODE, "Genişleme": cv2.MORPH_DILATE, "Açma": cv2.MORPH_OPEN, "Kapama": cv2.MORPH_CLOSE,
          "Gradyan": cv2.MORPH_GRADIENT, "Top-hat": cv2.MORPH_TOPHAT, "Black-hat": cv2.MORPH_BLACKHAT}


_SHAPE = {"kare": cv2.MORPH_RECT, "elips": cv2.MORPH_ELLIPSE, "çarpı": cv2.MORPH_CROSS}


for _name in _MORPH:
    def _mk(nm):
        def _f(img, p1, p2, p3, mode):
            k = cv2.getStructuringElement(_SHAPE[mode], (odd(p1), odd(p1)))
            return cv2.morphologyEx(img, _MORPH[nm], k, iterations=int(p2)), f"{nm} {odd(p1)}×{odd(p1)} {mode}, iter={int(p2)}"
        return _f
    op("8 · Morfoloji", _name, [("k", "yapı elemanı boyutu", 3, 31, 2, 3), ("it", "iterasyon", 1, 10, 1, 1)],
       modes=list(_SHAPE))(_mk(_name))


@op("8 · Morfoloji", "Bağlı bileşen filtresi (min alan)", [("minA", "min alan (px)", 1, 2000, 1, 8)],
    doc="make_gt.py'deki <8 px bileşen filtresi")
def _ccfilter(img, p1, p2, p3, mode):
    g = to_gray(img)
    b = (g > 127).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= int(p1); keep[0] = False
    return (keep[lab] * 255).astype(np.uint8), f"{int(keep.sum())}/{n - 1} bileşen kaldı (min {int(p1)} px)"


# ---- 9) Geometrik dönüşümler (Hafta 2)
_INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}


@op("9 · Geometrik", "Döndürme", [("ang", "açı (°)", -180, 180, 1, 15), ("s", "ölçek", 0.1, 3.0, 0.05, 1.0)], modes=list(_INTERP))
def _rot(img, p1, p2, p3, mode):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(p1), float(p2))
    return cv2.warpAffine(img, M, (w, h), flags=_INTERP[mode], borderMode=cv2.BORDER_REPLICATE), f"döndürme {p1}°, ölçek {p2}"


@op("9 · Geometrik", "Yeniden boyutlandırma", [("f", "ölçek çarpanı", 0.1, 4.0, 0.05, 0.75)], modes=list(_INTERP),
    doc="Not: dayanıklılık probu 0.75× küçült → 1× büyüt (bkz. 'Dayanıklılık' kategorisi)")
def _resize(img, p1, p2, p3, mode):
    h, w = img.shape[:2]
    out = cv2.resize(img, (max(1, int(round(w * p1))), max(1, int(round(h * p1)))), interpolation=_INTERP[mode])
    return out, f"{w}×{h} → {out.shape[1]}×{out.shape[0]} ({mode})"


@op("9 · Geometrik", "Öteleme", [("dx", "dx (px)", -500, 500, 1, 30), ("dy", "dy (px)", -500, 500, 1, 20)])
def _trans(img, p1, p2, p3, mode):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, p1], [0, 1, p2]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT), f"öteleme ({int(p1)},{int(p2)})"


@op("9 · Geometrik", "Aynalama", modes=["yatay", "dikey", "her iki"])
def _flip(img, p1, p2, p3, mode):
    code = {"yatay": 1, "dikey": 0, "her iki": -1}[mode]
    return cv2.flip(img, code), f"aynalama ({mode}) — metin yönü bozulur; eğitimde KULLANILMADI"


@op("9 · Geometrik", "Afin (kaydırma/shear)", [("shx", "shear x", -1.0, 1.0, 0.05, 0.2), ("shy", "shear y", -1.0, 1.0, 0.05, 0.0)])
def _shear(img, p1, p2, p3, mode):
    h, w = img.shape[:2]
    M = np.float32([[1, p1, 0], [p2, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE), f"shear ({p1:.2f},{p2:.2f})"


@op("9 · Geometrik", "Kırpma (merkez)", [("S", "kenar (px)", 64, 2048, 32, 512)])
def _crop(img, p1, p2, p3, mode):
    h, w = img.shape[:2]; S = int(min(p1, h, w))
    y0, x0 = (h - S) // 2, (w - S) // 2
    return img[y0:y0 + S, x0:x0 + S], f"merkez kırpma {S}×{S} (natif çözünürlük, yeniden örnekleme yok)"


# ---- 10) Normalizasyon
@op("10 · Normalizasyon", "Min-max [0,255]")
def _minmax(img, p1, p2, p3, mode):
    return to_u8(img.astype(np.float32)), "min-max ölçekleme"


@op("10 · Normalizasyon", "Z-score (görüntüleme için yeniden ölçeklendi)")
def _z(img, p1, p2, p3, mode):
    x = img.astype(np.float32); z = (x - x.mean()) / (x.std() + 1e-6)
    return clip_u8(z * 64 + 128), f"μ={x.mean():.1f}, σ={x.std():.1f} → z·64+128"


@op("10 · Normalizasyon", "ImageNet normalizasyonu (model girdisi önizleme)")
def _imnet(img, p1, p2, p3, mode):
    x = to_rgb(img).astype(np.float32) / 255.0
    z = (x - IMAGENET_MEAN) / IMAGENET_STD
    return clip_u8(z * 64 + 128), "(x/255 − μ)/σ ImageNet; görüntüleme için ×64+128"


@op("10 · Normalizasyon", "Boyut normalizasyonu (uzun kenar)", [("L", "uzun kenar (px)", 128, 4096, 32, 1024)])
def _sizenorm(img, p1, p2, p3, mode):
    h, w = img.shape[:2]; s = p1 / max(h, w)
    out = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return out, f"uzun kenar {int(p1)} px (proje: >4096 ise küçültülür, aksi hâlde natif)"


# ---- 11) Frekans domeni (Hafta 4)
@op("11 · Frekans domeni", "FFT genlik spektrumu (log)")
def _fft(img, p1, p2, p3, mode):
    return fft_spectrum_image(to_gray(img)), "log|F(u,v)| (merkezlenmiş)"


for _kind, _label in (("ideal", "İdeal"), ("gauss", "Gaussian"), ("butterworth", "Butterworth")):
    def _mkf(kind):
        def _f(img, p1, p2, p3, mode):
            hp = mode == "yüksek-geçiren"
            out = freq_filter(to_gray(img), kind, float(p1), float(p2), highpass=hp)
            return (to_u8(out) if hp else clip_u8(out)), f"{kind} {mode} D0={p1:.0f}" + (f", n={int(p2)}" if kind == "butterworth" else "")
        return _f
    op("11 · Frekans domeni", f"{_label} filtre", [("D0", "D0 (kesim frekansı)", 1, 300, 1, 20), ("n", "n (Butterworth derecesi)", 1, 10, 1, 2)],
       modes=["alçak-geçiren", "yüksek-geçiren"])(_mkf(_kind))


@op("11 · Frekans domeni", "Projenin frekans kanalları (U-Net D2)", modes=["FFT yüksek-geçiren D0=20", "Laplacian artığı"],
    doc="common/data.py freq_channels: örnek başına standartlaştırılmış")
def _projfreq(img, p1, p2, p3, mode):
    ch = project_freq_channels(to_rgb(img))
    i = 0 if mode.startswith("FFT") else 1
    return clip_u8(ch[i] * 40 + 128), f"{mode} (z·40+128)"


@op("11 · Frekans domeni", "8×8 blok DCT görselleştirme")
def _dct(img, p1, p2, p3, mode):
    return block_dct_image(to_gray(img)), "log|DCT| 8×8 blok (JPEG/DTD ipucu; veri PNG olduğundan iz yok)"


# ---- 12) Klasik segmentasyon (Hafta 7)
@op("12 · Klasik segmentasyon", "K-Means (yoğunluk/renk)", [("k", "K", 2, 10, 1, 3)], modes=["gri", "RGB"])
def _kmeans(img, p1, p2, p3, mode):
    src = to_gray(img)[..., None] if mode == "gri" else to_rgb(img)
    Z = src.reshape(-1, src.shape[-1]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(Z, int(p1), None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lab = lab.reshape(src.shape[:2])
    out = to_u8(lab.astype(np.float32))
    return out, f"K-Means K={int(p1)} ({mode}); etiketler gri seviyeye eşlendi"


@op("12 · Klasik segmentasyon", "Watershed (uzaklık dönüşümü işaretçili)", [("md", "min işaretçi uzaklığı", 3, 100, 1, 15)])
def _ws(img, p1, p2, p3, mode):
    g = to_gray(img)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, K3, iterations=2)
    dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=int(p1), labels=b)
    markers = np.zeros(dist.shape, np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    lab = watershed(-dist, markers, mask=b > 0)
    rng = np.random.default_rng(0)
    pal = np.vstack([[0, 0, 0], rng.integers(40, 255, (max(lab.max(), 1), 3))]).astype(np.uint8)
    return pal[lab], f"watershed: {lab.max()} bölge (Otsu-inv + açma + uzaklık dönüşümü)"


@op("12 · Klasik segmentasyon", "Bağlı bileşen etiketleme (renkli)", [("minA", "min alan", 1, 5000, 1, 20)])
def _cc(img, p1, p2, p3, mode):
    g = to_gray(img)
    if len(np.unique(g)) > 2:
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > 127).astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= int(p1); keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    rng = np.random.default_rng(1)
    pal = np.vstack([[0, 0, 0], rng.integers(40, 255, (max(n - 1, 1), 3))]).astype(np.uint8)
    return pal[lab], f"{int(keep.sum())} bileşen (min alan {int(p1)})"


# ---- 13) Dayanıklılık bozulmaları (projede test-zamanı probları)
@op("13 · Dayanıklılık (bozulma)", "JPEG yeniden kodlama", [("q", "kalite", 10, 100, 1, 75)])
def _jpeg(img, p1, p2, p3, mode):
    rgb = to_rgb(img)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(p1)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), f"JPEG Q{int(p1)} ({len(buf) / 1024:.0f} KB)"


@op("13 · Dayanıklılık (bozulma)", "Yeniden ölçekleme (küçült→büyüt)", [("f", "çarpan", 0.25, 0.95, 0.05, 0.75)])
def _rescale(img, p1, p2, p3, mode):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(round(w * p1))), max(1, int(round(h * p1)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), f"{p1:.2f}× → 1× (yeniden örnekleme izi)"


@op("13 · Dayanıklılık (bozulma)", "Gauss bulanıklık σ", [("sigma", "σ", 0.2, 5.0, 0.1, 1.0)])
def _blur(img, p1, p2, p3, mode):
    return cv2.GaussianBlur(img, (0, 0), float(p1)), f"Gauss blur σ={p1:.1f}"


@op("13 · Dayanıklılık (bozulma)", "Yeşil işaretçiyi onar (authentic'ten kopyala)", doc="Authentic eş gerekir; yalnız işaretçi pikselleri kopyalanır (Ek-C probu)")
def _repair(img, p1, p2, p3, mode):
    return img, "authentic eş yüklü değilse etkisiz (Girdi sekmesinden authentic yükleyin)"


CATEGORIES = list(OPS.keys())

PRESETS = {
    "— seçiniz —": [],
    "Belge ikilileştirme (gri → Gauss → Sauvola → açma)": [
        ("1 · Renk işleme", "Gri tonlama (luminance)", {}),
        ("5 · Gürültü / giderme", "Gauss filtre", {"p1": 3, "p2": 0.8}),
        ("4 · Eşikleme", "Sauvola (belge ikilileştirme)", {"p1": 25, "p2": 0.2}),
        ("8 · Morfoloji", "Açma", {"p1": 3, "p2": 1, "mode": "kare"}),
    ],
    "Frekans analizi (gri → FFT yüksek-geçiren D0=20)": [
        ("1 · Renk işleme", "Gri tonlama (luminance)", {}),
        ("11 · Frekans domeni", "Gaussian filtre", {"p1": 20, "p2": 2, "mode": "yüksek-geçiren"}),
    ],
    "Kenar haritası (gri → Gauss → Canny → genişleme)": [
        ("1 · Renk işleme", "Gri tonlama (luminance)", {}),
        ("5 · Gürültü / giderme", "Gauss filtre", {"p1": 5, "p2": 1.0}),
        ("7 · Kenar tespiti", "Canny", {"p1": 50, "p2": 150}),
        ("8 · Morfoloji", "Genişleme", {"p1": 3, "p2": 1, "mode": "kare"}),
    ],
    "Dayanıklılık probu: JPEG Q75": [("13 · Dayanıklılık (bozulma)", "JPEG yeniden kodlama", {"p1": 75})],
    "Dayanıklılık probu: 0.75× yeniden ölçekleme": [("13 · Dayanıklılık (bozulma)", "Yeniden ölçekleme (küçült→büyüt)", {"p1": 0.75})],
    "Dayanıklılık probu: Gauss blur σ=1": [("13 · Dayanıklılık (bozulma)", "Gauss bulanıklık σ", {"p1": 1.0})],
    "Kontrast iyileştirme (CLAHE → unsharp)": [
        ("3 · Histogram", "CLAHE", {"p1": 2.0, "p2": 8}),
        ("6 · Keskinleştirme", "Unsharp mask / high-boost", {"p1": 2.0, "p2": 1.0}),
    ],
}


def run_op(cur, cat, name, p1, p2, p3, mode, auth):
    if cur is None:
        raise gr.Error("Önce bir görüntü yükleyin ya da Simülasyon sekmesinden bir örneği aktarın.")
    spec = OPS[cat][name]
    if name.startswith("Yeşil işaretçiyi onar"):
        if auth is None or auth.shape[:2] != cur.shape[:2]:
            raise gr.Error("Bu adım aynı boyutta bir authentic eş gerektirir.")
        rgb = to_rgb(cur).copy(); gm = green_marker_mask(rgb) & ~green_marker_mask(auth)
        gm = cv2.dilate(gm.astype(np.uint8), K3, iterations=1) > 0
        rgb[gm] = auth[gm]
        return rgb, f"işaretçi onarıldı: {int(gm.sum())} px authentic'ten kopyalandı"
    return spec["fn"](cur, p1, p2, p3, mode)


def replay(start, steps, auth):
    cur = start
    for s in steps:
        cur, _ = run_op(cur, s["cat"], s["name"], *s["p"], s["mode"], auth)
    return cur
