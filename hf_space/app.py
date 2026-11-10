"""
app.py — AIC515 Görüntü İşleme · AIForge-Doc Sahtecilik Lokalizasyonu — Simülasyon + Aşamalı İşlem Hattı (TR)

Sekmeler
  0 · Simülasyon   : veri setinden bir örnek seç (filtre / rastgele / kimlik) → girdi senaryosu (sahte, authentic,
                     işaretçi aktarımı/onarımı) → bozulma → ResNet-18 + U-Net'ler → adım adım çıktı, metrikler
                     (bbox ve piksel-değişim GT'si), görüntü-düzeyi karar, yakın plan paneli, indirilebilir rapor.
  1 · Toplu test   : N örnek üzerinde aynı hat; model başına toplu metrikler, tespit doğruluğu, boru hattının
                     kayıtlı sonuçlarıyla yeniden-üretim karşılaştırması, CSV/JSON.
  2–7              : girdi yükleme, 61 ön işleme işlemi (aşamalı pipeline), U-Net, ResNet-18, yeşil işaretçi, sonuçlar.
  8 · XAI Analysis : iki bağımsız alt sekme —
      SHAP Analysis : tahmin → SHAP (Partition / Gradient; authentic / blur / inpaint referansı) → model karşılaştırması;
                     xai/ paketi (docs/XAI.md).
      Filter Visualization / Katman Filtreleri : SHAP'tan bağımsız — ilk conv katmanlarının checkpoint'teki gerçek kernel'leri, görüntüye
                     uygulanmış aktivasyon haritaları (conv / BN / aktivasyon), orijinalle karşılaştırma; xai_filters.py
                     (docs/XAI_FILTERS.md).

Çalıştırma
  Yerel (proje kökü):   python app/app.py                        → http://127.0.0.1:7860
  Hugging Face Space :  ZeroGPU (@spaces.GPU); veri/modeller özel bucket'tan AI515_DATA_ROOT (/data) altına bağlanır.
"""
from __future__ import annotations
import os, sys, io, re, json, time, math, glob, random, tempfile, datetime
from pathlib import Path

os.environ.setdefault("USERNAME", "user")  # torch inductor getpass sorunu (ENV_SETUP.md)
# ZeroGPU: `spaces` torch/CUDA'dan ÖNCE içe aktarılmalı. Paket yoksa (yerel) dekoratör etkisiz kalır.
try:
    import spaces  # type: ignore
    _HAS_SPACES = True
except Exception:
    spaces = None
    _HAS_SPACES = False

import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import segmentation_models_pytorch as smp
    TORCH_OK = True
    _TORCH_ERR = ""
except Exception as e:  # torch yoksa yalnız klasik işlemler çalışır
    TORCH_OK = False
    _TORCH_ERR = str(e)

from skimage.filters import threshold_sauvola
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ------------------------------------------------------------------ yollar
# Yerelde app/ proje kökünün altındadır (ROOT = üst klasör). Space'te app.py repo kökündedir; veri ve modeller
# bucket'tan AI515_DATA_ROOT altına bağlanır ve proje dizin yapısını izler (data/, models/, dataset_audit/).
APP_DIR = Path(__file__).resolve().parent
IN_SPACE = bool(os.environ.get("SPACE_ID"))
ROOT = Path(os.environ.get("VIZE_ROOT") or (APP_DIR if (APP_DIR / "baselines").exists() else APP_DIR.parent))
DATA_ROOT = Path(os.environ.get("AI515_DATA_ROOT") or ROOT)
DATA = DATA_ROOT / "data" / "AIForge-Doc-v1"
MODELS = DATA_ROOT / "models"
AUDIT = DATA_ROOT / "dataset_audit"
if not any(MODELS.glob("*.pt")) and any((ROOT / "models").glob("*.pt")):
    MODELS = ROOT / "models"
if not (AUDIT / "samples_forged.csv").exists() and (ROOT / "dataset_audit" / "samples_forged.csv").exists():
    AUDIT = ROOT / "dataset_audit"
META = APP_DIR / "meta"  # uygulamayla gelen küçük tablolar (eşikler, referans sonuçlar, işaretçi listesi)
RESULTS_MD = ROOT / "baselines" / "results" / "results_tables.md"
P2_SUMMARY = ROOT / "phase2" / "experiments" / "results" / "summary_test_agg.csv"
TAU_JSON = ROOT / "phase2" / "experiments" / "results" / "tau.json"
if not TAU_JSON.exists() and (META / "tau.json").exists():
    TAU_JSON = META / "tau.json"

sys.path.insert(0, str(ROOT / "baselines"))
try:
    import features_classical as FC
    FC_OK = True
except Exception:
    FC_OK = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _zero_gpu() -> bool:
    if not (_HAS_SPACES and IN_SPACE):
        return False
    try:
        from spaces.config import Config  # type: ignore
        return bool(Config.zero_gpu)
    except Exception:
        return bool(os.environ.get("SPACES_ZERO_GPU"))


ZERO_GPU = _zero_gpu()
if TORCH_OK:
    DEVICE = torch.device("cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu")
else:
    DEVICE = None


def gpu(duration=60):
    """@spaces.GPU sarmalayıcısı: ZeroGPU'da GPU ister, diğer ortamlarda etkisizdir. duration sayı ya da çağrılabilir."""
    def deco(fn):
        if _HAS_SPACES:
            try:
                return spaces.GPU(duration=duration)(fn)
            except Exception:
                return fn
        return fn
    return deco



# ------------------------------------------------------------------ görüntü yardımcıları
def to_u8(x: np.ndarray) -> np.ndarray:
    """float dizi → uint8 (min-max ile [0,255] ölçekleme; zaten uint8 ise dokunma)."""
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-8:
        return np.zeros(x.shape, np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=2)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def clip_u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def odd(k):
    k = int(round(k))
    return k if k % 2 == 1 else k + 1


def hist_figure(before: np.ndarray, after: np.ndarray):
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.6), dpi=100)
    for a, im, t in ((ax[0], before, "Önce"), (ax[1], after, "Sonra")):
        g = to_gray(im) if im.ndim == 3 else im
        a.hist(g.ravel(), bins=256, range=(0, 255), color="#4C72B0")
        a.set_title(f"{t} — histogram (gri)", fontsize=9)
        a.set_xlim(0, 255); a.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ frekans yardımcıları
def _dist_grid(H, W):
    fy = np.fft.fftfreq(H)[:, None] * H
    fx = np.fft.fftfreq(W)[None, :] * W
    return np.sqrt(fy ** 2 + fx ** 2)  # fftshift'siz (merkez (0,0) köşede)


def freq_filter(gray_u8: np.ndarray, kind: str, d0: float, order: float = 2.0, highpass: bool = False):
    g = gray_u8.astype(np.float32)
    H, W = g.shape
    D = _dist_grid(H, W)
    if kind == "ideal":
        Hf = (D <= d0).astype(np.float32)
    elif kind == "gauss":
        Hf = np.exp(-(D ** 2) / (2.0 * d0 ** 2))
    else:  # butterworth
        Hf = 1.0 / (1.0 + (D / max(d0, 1e-6)) ** (2 * order))
    if highpass:
        Hf = 1.0 - Hf
    Fg = np.fft.fft2(g)
    out = np.real(np.fft.ifft2(Fg * Hf))
    return out


def project_freq_channels(rgb_u8: np.ndarray, d0: float = 20.0):
    """Projenin U-Net D2 varyantındaki 2 ek kanal: FFT Gaussian yüksek-geçiren (D0=20) + 3×3 Laplacian artığı,
    her biri örnek başına standartlaştırılır (common/data.py: freq_channels ile aynı tanım)."""
    g = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32) / 255.0
    hp = freq_filter((g * 255).astype(np.uint8), "gauss", d0, highpass=True) / 255.0
    lap = cv2.Laplacian(cv2.copyMakeBorder(g, 1, 1, 1, 1, cv2.BORDER_REPLICATE), cv2.CV_32F, ksize=1)[1:-1, 1:-1]
    out = []
    for c in (hp, lap):
        out.append((c - c.mean()) / (c.std() + 1e-6))
    return np.stack(out, axis=0).astype(np.float32)  # (2,H,W)


def fft_spectrum_image(gray_u8: np.ndarray):
    F_ = np.fft.fftshift(np.fft.fft2(gray_u8.astype(np.float32)))
    return to_u8(np.log1p(np.abs(F_)))


def block_dct_image(gray_u8: np.ndarray):
    g = gray_u8.astype(np.float32) - 128.0
    H, W = g.shape
    H8, W8 = H // 8 * 8, W // 8 * 8
    g = g[:H8, :W8]
    Dm = cv2.dct(np.eye(8, dtype=np.float32))
    blocks = g.reshape(H8 // 8, 8, W8 // 8, 8).transpose(0, 2, 1, 3)
    coef = Dm @ blocks @ Dm.T
    out = coef.transpose(0, 2, 1, 3).reshape(H8, W8)
    return to_u8(np.log1p(np.abs(out)))


# ------------------------------------------------------------------ yeşil işaretçi dedektörü (phase2/experiments/probe_green_marker.py)
def green_marker_mask(rgb_u8: np.ndarray) -> np.ndarray:
    r, g, b = [rgb_u8[..., i].astype(np.int16) for i in range(3)]
    return (g - np.maximum(r, b) > 50) & (g > 110)


# ------------------------------------------------------------------ piksel-değişim GT türetimi (phase2/gt/make_gt.py: pixel_eps8_morph)
K3 = np.ones((3, 3), np.uint8)
KX = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Faz 3 sondaları (scipy.ndimage varsayılan yapı elemanı)


def derive_pixel_gt(forged_rgb: np.ndarray, auth_rgb: np.ndarray, eps: int = 8):
    if forged_rgb.shape != auth_rgb.shape:
        return None
    D = np.abs(forged_rgb.astype(np.int16) - auth_rgb.astype(np.int16)).max(axis=2).astype(np.uint8)
    e = (D > eps).astype(np.uint8)
    o = cv2.morphologyEx(e, cv2.MORPH_OPEN, K3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= 8
    keep[0] = False
    o = keep[lab].astype(np.uint8)
    return cv2.morphologyEx(o, cv2.MORPH_CLOSE, K3).astype(bool), D


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




# ================================================================== VERİ SETİ DİZİNİ
TOOL_SHORT = {"gemini-nano": "Gemini", "qwen-inpaint": "Ideogram"}
SPLIT_DIR = {"training": "TrainingSet", "testing": "TestingSet"}
_INDEX = None
_EXISTS: dict = {}
_SAMPLE_CACHE: dict = {}


def _read_meta_csv(name):
    for p in (META / name, AUDIT / name, ROOT / "phase2" / "gt" / name):
        if p.exists():
            return pd.read_csv(p, dtype={"new_id": str}, low_memory=False)
    return None


def dataset_index() -> pd.DataFrame:
    """samples_forged.csv (+ yeşil işaretçi bayrağı) → örnek dizini. Anahtar: '<split>:<9 haneli id>'."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    df = _read_meta_csv("samples_forged.csv")
    if df is None:
        rows = []
        for split, d in SPLIT_DIR.items():
            for p in sorted(glob.glob(str(DATA / d / "images" / "*.png"))):
                rows.append(dict(new_id=Path(p).stem, split=split))
        df = pd.DataFrame(rows, columns=["new_id", "split"])
    keep = [c for c in ["new_id", "split", "pair_name", "pair_accepted", "pair_same_size", "assigned_tool", "source_dataset",
                        "area_bin", "field_name", "original_value", "forged_value", "W", "H", "megapixels", "language", "doc_type"] if c in df.columns]
    df = df[keep].copy()
    df["new_id"] = df["new_id"].astype(str).str.zfill(9)
    for c, d in (("assigned_tool", ""), ("source_dataset", ""), ("area_bin", ""), ("pair_name", ""), ("field_name", ""),
                 ("original_value", ""), ("forged_value", ""), ("language", ""), ("doc_type", "")):
        if c not in df.columns:
            df[c] = d
        df[c] = df[c].fillna(d).astype(str)
    for c in ("W", "H"):
        if c not in df.columns:
            df[c] = 0
    if "megapixels" not in df.columns:
        df["megapixels"] = df["W"] * df["H"] / 1e6
    df["has_pair"] = df.get("pair_accepted", pd.Series(False, index=df.index)).astype(str).eq("True")
    g = _read_meta_csv("green_marker_all_forged.csv")
    if g is not None and "has_marker" in g.columns:
        g["new_id"] = g["new_id"].astype(str).str.zfill(9)
        df = df.merge(g[["new_id", "split", "has_marker"]], on=["new_id", "split"], how="left")
        df["has_marker"] = df["has_marker"].astype(str).eq("True")
    else:
        df["has_marker"] = False
    df["tool"] = df["assigned_tool"].map(TOOL_SHORT).fillna(df["assigned_tool"])
    df["key"] = df["split"] + ":" + df["new_id"]
    df["label"] = df.apply(lambda r: f"{r.split}:{r.new_id} · {r.source_dataset} · {r.tool} · alan {r.area_bin} · "
                                     f"{r.megapixels:.1f} MP" + (" · eşli" if r.has_pair else "") + (" · işaretçi" if r.has_marker else ""), axis=1)
    _INDEX = df.set_index("key", drop=False)
    return _INDEX


def sample_files(key: str):
    split, nid = key.split(":")
    d = DATA / SPLIT_DIR.get(split, "TrainingSet")
    r = dataset_index().loc[key] if key in dataset_index().index else None
    auth = d / "authentic" / r.pair_name if (r is not None and r.has_pair and r.pair_name) else None
    return d / "images" / f"{nid}.png", d / "masks" / f"{nid}.png", auth


def sample_exists(key: str) -> bool:
    if key not in _EXISTS:
        _EXISTS[key] = sample_files(key)[0].exists()
    return _EXISTS[key]


def filter_index(split="hepsi", source="hepsi", tool="hepsi", area="hepsi", pair_only=False, marker="hepsi", max_mp=40.0):
    df = dataset_index()
    m = pd.Series(True, index=df.index)
    if split != "hepsi":
        m &= df["split"].eq({"eğitim (training)": "training", "test (testing)": "testing"}.get(split, split))
    if source != "hepsi":
        m &= df["source_dataset"].eq(source)
    if tool != "hepsi":
        m &= df["tool"].eq(tool)
    if area != "hepsi":
        m &= df["area_bin"].eq(area)
    if pair_only:
        m &= df["has_pair"]
    if marker == "var":
        m &= df["has_marker"]
    elif marker == "yok":
        m &= ~df["has_marker"]
    if max_mp:
        m &= df["megapixels"].astype(float) <= float(max_mp)
    return df[m]


def sample_choices(df=None, limit=6000):
    df = dataset_index() if df is None else df
    return [(r.label, r.key) for r in df.head(limit).itertuples()]


def read_rgb(p: Path):
    a = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(str(p))
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def load_dataset_sample(key: str):
    """(forged RGB, mask 0/255, authentic RGB | None, meta dict). Son 6 örnek önbellekte tutulur."""
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]
    ip, mp, ap = sample_files(key)
    if not ip.exists():
        raise FileNotFoundError(f"{ip} bulunamadı (veri seti bucket'a yükleniyor olabilir).")
    img = read_rgb(ip)
    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
    auth = None
    if ap is not None and ap.exists():
        a = read_rgb(ap)
        auth = a if a.shape == img.shape else None
    r = dataset_index().loc[key] if key in dataset_index().index else None
    meta = dict(key=key, new_id=key.split(":")[1], split=key.split(":")[0])
    if r is not None:
        meta.update(source=r.source_dataset, tool=r.tool, area_bin=r.area_bin, field=r.field_name, original=r.original_value,
                    forged=r.forged_value, language=r.language, doc_type=r.doc_type, has_marker_index=bool(r.has_marker),
                    pair_name=r.pair_name if r.has_pair else "")
    out = (img, mask, auth, meta)
    if len(_SAMPLE_CACHE) > 6:
        _SAMPLE_CACHE.pop(next(iter(_SAMPLE_CACHE)))
    _SAMPLE_CACHE[key] = out
    return out


def meta_md(meta, img, mask, auth):
    h, w = img.shape[:2]
    s = f"**{meta.get('key')}** · {meta.get('source', '?')} · {meta.get('tool', '?')} · {meta.get('doc_type', '')} · dil `{meta.get('language', '')}` · {w}×{h} px ({w * h / 1e6:.2f} MP)"
    if meta.get("field"):
        s += f"  \nDeğiştirilen alan `{meta['field']}`: `{meta.get('original', '')}` → `{meta.get('forged', '')}` · alan dilimi {meta.get('area_bin', '')}"
    if mask is not None:
        m = mask > 127
        s += f"  \nGT (bbox) maskesi: {int(m.sum())} px (%{100 * m.mean():.2f})"
    s += "  \nAuthentic eş: " + (f"`{meta.get('pair_name')}` ✓" if auth is not None else "yok (resmî test authentic'leri çoğunlukla ayna değildir)")
    return s


# ================================================================== MODELLER
_REG: dict = {}          # ad → (model, in_ch, tür)
_LOAD_ERR: dict = {}


# [FAZ3] model kaydı: PHASE2/ (Faz 2 + Faz 3 segmentasyon), PHASE1/, YOLO/ (Faz 3 kutu dedektörleri)
_HIDE = re.compile(r"(_e20$|geminionly_s4[34]$)")  # 20 epoch ve ek held-out tohumları arayüzde gösterilmez


def list_unet_ckpts():
    ph2 = []
    for pat in ("unet_*.pt", "dlv3p_*.pt", "unetjoint_*.pt"):
        ph2 += sorted(glob.glob(str(MODELS / "phase2" / pat)))
    out = [f"PHASE2/{Path(p).stem}" for p in ph2 if not _HIDE.search(Path(p).stem)]
    out += [f"PHASE1/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "unet_*.pt")))]
    out += [f"YOLO/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "phase3_yolo" / "aiforge-*.pt")))
            if not re.search(r"(_stem$|-s4[34]$)", Path(p).stem)]  # tohum tekrarları (s43/s44) ve katman dökümleri gizli
    return out


def list_seg_ckpts():
    """Piksel segmentasyon modelleri (XAI adaptörleri için; YOLO hariç)."""
    return [n for n in list_unet_ckpts() if not n.startswith("YOLO/")]


def list_resnet_ckpts():
    out = [f"PHASE1/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "resnet18_*.pt")))]
    out += [f"PHASE2/{Path(p).stem}" for p in sorted(glob.glob(str(MODELS / "phase2" / "h4_resnet18*.pt")))]
    return out


def ckpt_path(name: str) -> Path:
    phase, stem = name.split("/", 1)
    if phase == "YOLO":
        return MODELS / "phase3_yolo" / f"{stem}.pt"
    return (MODELS / f"{stem}.pt") if phase == "PHASE1" else (MODELS / "phase2" / f"{stem}.pt")


def friendly(name: str) -> str:
    stem = name.split("/", 1)[-1]
    seed = stem.rsplit("_s", 1)[-1] if "_s4" in stem else ""
    gem = " · yalnız Gemini" if "geminionly" in stem else ""
    if name.startswith("YOLO/"):
        m = re.match(r"aiforge-(?P<d>[a-z_]+)-yolo11(?P<a>[nsm])-img(?P<i>\d+)-(?P<g>[a-z]+)-s(?P<s>\d+)", stem)
        if not m:
            return f"YOLO11 · {stem}"
        d = m.group("d")
        how = ("tam görüntü 1280" if d == "full" else "kare 1024") + (" · yalnız Gemini" if "geminionly" in d else "") + \
              (" · +authentic kare" if "authneg" in d else "") + (" · varsayılan aug" if m.group("g") == "default" else "")
        return f"YOLO11{m.group('a')} · kutu tespiti, {how} (Faz 3, s{m.group('s')})"
    if stem.startswith("dlv3p_"):
        g = "bbox-GT" if "bbox" in stem else ("piksel-GT" if "pixel" in stem else "glif-GT")
        return f"DeepLabV3+ · {g} (Faz 3, s{seed})"
    if stem.startswith("unetjoint_"):
        g = "piksel-GT" if "pixel" in stem else "glif-GT"
        return f"U-Net · {g} + kutu yardımcı kaybı (Faz 3, s{seed})"
    if "_mr_" in stem:
        g = "bbox-GT" if "bbox" in stem else ("piksel-GT" if "pixel" in stem else "glif-GT")
        return f"U-Net · {g} · işaretçisiz eğitim (Faz 3, s{seed})"
    if stem.startswith("unet_bbox"):
        return f"U-Net · bbox-GT{gem} (s{seed})"
    if stem.startswith("unet_pixel"):
        return f"U-Net · piksel-GT{gem} (s{seed})"
    if stem.startswith("unet_glyph"):
        return f"U-Net · glif-GT{gem} (s{seed})"
    if stem == "unet_rgb_best":
        return "U-Net · RGB (PHASE1, 400 örnek)"
    if stem == "unet_rgbfreq_best":
        return "U-Net · RGB+FFT+Laplacian (PHASE1)"
    if stem.startswith("resnet18_"):
        return {"resnet18_ft_best": "ResNet-18 · tam ince ayar", "resnet18_probe_best": "ResNet-18 · linear probe",
                "resnet18_scratch_best": "ResNet-18 · sıfırdan"}.get(stem, stem)
    if stem.startswith("h4_resnet18"):
        return "ResNet-18 · H4 global-iz sondası" + (" (poz. kontrol)" if "poscontrol" in stem else "")
    return stem


def _build_unet(name):
    sd = torch.load(ckpt_path(name), map_location="cpu")
    if any(k.startswith("net.") for k in sd):  # PHASE1 UNetWrapper
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
    in_ch = int(sd["encoder.conv1.weight"].shape[1])
    stem = name.split("/", 1)[-1]
    if stem.startswith("dlv3p_"):  # Faz 3: ikinci mimari
        m = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)
    elif stem.startswith("unetjoint_"):  # Faz 3: kanal 0 = piksel/glif maske, kanal 1 = kutu yardımcı (çıkarımda kanal 0)
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=2)
    else:
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_ch, classes=1)
    m.load_state_dict(sd)
    if stem.startswith("unetjoint_"):  # kutu yardımcı başını at: yalnız kanal 0 (birebir aynı çıktı, tek kanal → XAI/çıkarım uyumlu)
        c = m.segmentation_head[0]
        h = nn.Conv2d(c.in_channels, 1, c.kernel_size, padding=c.padding)
        with torch.no_grad():
            h.weight.copy_(c.weight[:1]); h.bias.copy_(c.bias[:1])
        m.segmentation_head[0] = h
    return m.to(DEVICE).eval(), in_ch


def _build_resnet(name):
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(512, 1)
    m.load_state_dict(torch.load(ckpt_path(name), map_location="cpu"))
    return m.to(DEVICE).eval()


def _build_yolo(name):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics"))
    from ultralytics import YOLO  # yalnız YOLO seçildiğinde gerekir
    m = YOLO(str(ckpt_path(name)))
    if DEVICE is not None:
        m.to(DEVICE)
    return m


def get_model(name):
    if name not in _REG:
        if name.startswith("YOLO/"):
            _REG[name] = (_build_yolo(name), 3, "yolo")
        elif "resnet18" in name:
            _REG[name] = (_build_resnet(name), 3, "resnet")
        else:
            m, c = _build_unet(name)
            _REG[name] = (m, c, "unet")
    return _REG[name]


def preload_models():
    """ZeroGPU kuralı: modeller modül düzeyinde cuda'ya yerleştirilir (emülasyon); çıkarım @spaces.GPU içinde."""
    if not TORCH_OK:
        return
    t0 = time.time()
    for n in list_unet_ckpts() + list_resnet_ckpts():
        try:
            get_model(n)
        except Exception as e:  # bozuk/eksik ağırlık uygulamayı durdurmasın
            _LOAD_ERR[n] = str(e)
    print(f"[ai515] {len(_REG)} model yüklendi ({DEVICE}, ZeroGPU={ZERO_GPU}) {time.time() - t0:.1f}s; hata: {list(_LOAD_ERR)}", flush=True)


def _autocast():
    if DEVICE is not None and DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def norm_chw(rgb_u8: np.ndarray) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _windows(H, W, win, stride):
    ys = list(range(0, max(H - win, 0) + 1, stride)); xs = list(range(0, max(W - win, 0) + 1, stride))
    if ys[-1] + win < H: ys.append(H - win)
    if xs[-1] + win < W: xs.append(W - win)
    return [(y, x) for y in ys for x in xs]


def _unet_prepare(rgb_u8, win=512, max_side=4096):
    """U-Net girdi hazırlığı: uzun kenar > max_side ise INTER_AREA ile küçült; kısa kenar < win ise sağ/alttan replicate
    dolgu. Dönüş: (görüntü, (h, w) dolgusuz boyut, ölçek). _predict_unet ve XAI adaptörü aynı fonksiyonu kullanır."""
    H0, W0 = rgb_u8.shape[:2]
    s = 1.0
    img = rgb_u8
    if max(H0, W0) > max_side:
        s = max_side / max(H0, W0)
        img = cv2.resize(img, (max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    ph, pw = max(0, win - h), max(0, win - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return img, (h, w), s


def _unet_window_forward(model, in_ch, xb, amp=True):
    """Pencere toplu ileri geçişi → (N, h, w) olasılık. 5 kanallı modelde frekans kanalları pencere başına
    standartlaştırılır (projedeki tanım). amp=False: fp32 (XAI gradyan açıklaması). Tek doğruluk kaynağı."""
    if in_ch == 5:  # projede frekans kanalları pencere başına standartlaştırılır
        fc = xb[:, 3:]
        mu = fc.mean(dim=(2, 3), keepdim=True); sd = fc.std(dim=(2, 3), keepdim=True) + 1e-6
        xb = torch.cat([xb[:, :3], (fc - mu) / sd], 1)
    if amp:
        with _autocast():
            return torch.sigmoid(model(xb).float())[:, 0]
    return torch.sigmoid(model(xb.float()).float())[:, 0]


def _predict_yolo(name, rgb_u8, tile=1024, stride=768, conf=0.001, iou=0.6, batch=8):
    """phase3_yolo/infer_to_maps.py protokolü: natif çözünürlükte 1024 kare / 768 adım (ya da '-full-' modelde tam görüntü 1280),
    kareler arası NMS; her kutu güveniyle haritaya boyanır (örtüşmede maksimum) → U-Net ile aynı metrikler."""
    from torchvision.ops import nms
    model, _, _ = get_model(name)
    stem = name.split("/", 1)[-1]
    H0, W0 = rgb_u8.shape[:2]
    t0 = time.time()
    dev = 0 if (DEVICE is not None and DEVICE.type == "cuda") else "cpu"
    kw = dict(conf=conf, iou=iou, verbose=False, device=dev, half=(dev != "cpu"))
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if "-full-" in stem:
        r = model.predict(bgr, imgsz=1280, **kw)[0]
        b = r.boxes.xyxy.float().cpu().numpy(); c = r.boxes.conf.float().cpu().numpy(); nw = 1
    else:
        ph, pw = max(0, tile - H0), max(0, tile - W0)
        img = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else bgr
        H, W = img.shape[:2]
        wins = _windows(H, W, tile, stride); nw = len(wins)
        bl, cl = [], []
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            outs = model.predict([np.ascontiguousarray(img[y:y + tile, x:x + tile]) for (y, x) in chunk], imgsz=tile, **kw)
            for (y, x), r in zip(chunk, outs):
                bb = r.boxes.xyxy.float().cpu().numpy()
                if len(bb):
                    bl.append(bb + np.array([x, y, x, y], np.float32)); cl.append(r.boxes.conf.float().cpu().numpy())
        if bl:
            b = np.concatenate(bl); c = np.concatenate(cl)
            keep = nms(torch.from_numpy(b), torch.from_numpy(c), iou).numpy(); b, c = b[keep], c[keep]
        else:
            b, c = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    if len(b):
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W0); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H0)
    p = np.zeros((H0, W0), np.float32)
    for (x0, y0, x1, y1), s_ in sorted(zip(b.tolist(), c.tolist()), key=lambda t: t[1]):
        ys_, xs_ = slice(int(y0), int(np.ceil(y1))), slice(int(x0), int(np.ceil(x1)))
        p[ys_, xs_] = np.maximum(p[ys_, xs_], s_)
    top = sorted(zip(c.tolist(), b.tolist()), reverse=True)[:10]
    return p, dict(n_windows=nw, resized=0, sec=time.time() - t0, n_boxes=int(len(b)),
                   boxes=[(round(s_, 3), [int(v) for v in bx]) for s_, bx in top])


def _predict_unet(name, rgb_u8, win=512, stride=448, max_side=4096, batch=8):
    """infer_fullimage.py protokolü: 512 pencere, 448 adım, örtüşme ortalaması; uzun kenar > 4096 ise küçült → harita geri büyüt."""
    if name.startswith("YOLO/"):
        return _predict_yolo(name, rgb_u8)
    model, in_ch, _ = get_model(name)
    H0, W0 = rgb_u8.shape[:2]
    img, (h, w), _s = _unet_prepare(rgb_u8, win, max_side)
    resized = _s != 1.0
    H, W = img.shape[:2]
    x = torch.from_numpy(norm_chw(img))
    if in_ch == 5:
        x = torch.cat([x, torch.from_numpy(project_freq_channels(img))], 0)
    x = x.to(DEVICE)
    wins = _windows(H, W, win, stride)
    acc = torch.zeros((H, W), dtype=torch.float32, device=DEVICE); cnt = torch.zeros_like(acc)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            xb = torch.stack([x[:, y:y + win, xx:xx + win] for (y, xx) in chunk])
            out = _unet_window_forward(model, in_ch, xb)
            for j, (y, xx) in enumerate(chunk):
                acc[y:y + win, xx:xx + win] += out[j]; cnt[y:y + win, xx:xx + win] += 1
    p = (acc / cnt).cpu().numpy()[:h, :w]
    if resized:
        p = cv2.resize(p, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return p.astype(np.float32), dict(n_windows=len(wins), resized=int(resized), sec=time.time() - t0)


def _predict_resnet(name, rgb_u8, patch=224, stride=112, batch=32):
    """224 natif patch kayan pencere → patch skorları (örtüşme ortalaması haritası) + patch listesi."""
    model, _, _ = get_model(name)
    h, w = rgb_u8.shape[:2]
    img = rgb_u8
    ph, pw = max(0, patch - h), max(0, patch - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    H, W = img.shape[:2]
    x = torch.from_numpy(norm_chw(img)).to(DEVICE)
    wins = _windows(H, W, patch, stride)
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32); scores = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            xb = torch.stack([x[:, y:y + patch, xx:xx + patch] for (y, xx) in chunk])
            with _autocast():
                s = torch.sigmoid(model(xb).float()).squeeze(1).cpu().numpy()
            for j, (y, xx) in enumerate(chunk):
                acc[y:y + patch, xx:xx + patch] += s[j]; cnt[y:y + patch, xx:xx + patch] += 1
                scores.append((float(s[j]), int(y), int(xx)))
    return (acc / np.maximum(cnt, 1))[:h, :w], scores, time.time() - t0


def _mp(img):
    return min(img.shape[0] * img.shape[1], 4096 * 4096) / 1e6


def _dur_infer(unet_names, resnet_names, imgs, stride=112):
    mp = sum(_mp(i) for i in imgs)
    return int(min(240, max(20, 8 + mp * (1.2 * len(unet_names) + 0.8 * len(resnet_names)))))


@gpu(duration=_dur_infer)
def _gpu_infer(unet_names, resnet_names, imgs, stride=112):
    """Tek GPU çağrısında birden çok görüntü × model. Olasılıklar 8-bit döner (metrikler zaten 8-bit nicemlemeyle
    hesaplanır; ZeroGPU süreçler arası aktarımı 4× küçülür)."""
    out = []
    for img in imgs:
        res = {}
        for n in unet_names:
            p, info = _predict_unet(n, img)
            res[n] = (q_u8(p), info)
        for n in resnet_names:
            hm, sc, sec = _predict_resnet(n, img, stride=int(stride))
            res[n] = (q_u8(hm), dict(scores=sc, sec=sec))
        out.append(res)
    return out


def gpu_infer(unet_names, resnet_names, imgs, stride=112):
    raw = _gpu_infer(list(unet_names), list(resnet_names), [np.ascontiguousarray(i) for i in imgs], int(stride))
    return [{k: (u.astype(np.float32) / 255.0, info) for k, (u, info) in r.items()} for r in raw]


# ================================================================== METRİKLER
def q_u8(prob):
    return np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)


def hist_metrics(prob: np.ndarray, gt: np.ndarray, thr: float):
    """Boru hattıyla aynı: olasılıklar 8-bit'e nicemlenir (PNG tahmin), 256 kutulu histogramla TÜM pikseller üzerinde
    kesin AP; F1/IoU eşik τ'da (u8/255 ≥ τ); Oracle-F1 = görüntü başına en iyi eşik."""
    u = q_u8(prob); g = gt.astype(bool)
    hp = np.bincount(u[g], minlength=256)[:256].astype(np.float64)
    hn = np.bincount(u[~g], minlength=256)[:256].astype(np.float64)
    P = hp.sum()
    tp = np.cumsum(hp[::-1])[::-1]; fp = np.cumsum(hn[::-1])[::-1]  # tp[t] = #pos(u ≥ t)
    TPd, FPd = tp[::-1], fp[::-1]                                   # eşik azalan sırada
    prec = TPd / np.maximum(TPd + FPd, 1); rec = TPd / max(P, 1)
    ap = float(np.sum((rec - np.r_[0, rec[:-1]]) * prec)) if P > 0 else float("nan")
    k = int(min(255, max(0, math.ceil(thr * 255 - 1e-9))))
    TP, FP = tp[k], fp[k]; FN = P - TP
    f1 = 2 * TP / max(2 * TP + FP + FN, 1) if P > 0 else float("nan")
    iou = TP / max(TP + FP + FN, 1) if P > 0 else float("nan")
    f1all = 2 * tp / np.maximum(2 * tp + fp + (P - tp), 1)
    return dict(F1=float(f1), IoU=float(iou), AP=ap, OracleF1=float(f1all.max()) if P > 0 else float("nan"),
                TP=int(TP), FP=int(FP), FN=int(FN), P=int(P))


def image_score(prob):
    u = q_u8(prob).ravel()
    k = max(1, int(0.01 * u.size))
    return float(np.partition(u, -k)[-k:].mean() / 255.0)


def rectangularity(binm: np.ndarray):
    ys, xs = np.where(binm)
    if len(xs) == 0:
        return float("nan"), None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return float(binm.sum() / ((x1 - x0) * (y1 - y0))), (int(x0), int(y0), int(x1), int(y1))


def postprocess(prob: np.ndarray, thr: float, morph_k: int, use_otsu: bool):
    if use_otsu:
        T, b = cv2.threshold(q_u8(prob), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binm = b > 0; thr_used = T / 255.0
    else:
        binm = q_u8(prob) >= int(min(255, max(0, math.ceil(thr * 255 - 1e-9)))); thr_used = thr
    if morph_k and morph_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(morph_k), odd(morph_k)))
        b = cv2.morphologyEx(binm.astype(np.uint8), cv2.MORPH_OPEN, k)
        binm = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k) > 0
    return binm, thr_used


def heatmap_rgb(prob: np.ndarray):
    return cv2.cvtColor(cv2.applyColorMap(q_u8(prob), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(rgb_u8: np.ndarray, binm, gt=None, bbox=None, alpha=0.45, gt_color=(0, 160, 255)):
    out = to_rgb(rgb_u8).copy()
    if gt is not None:
        g = gt.astype(bool)
        out[g] = (out[g] * 0.65 + np.array(gt_color) * 0.35).astype(np.uint8)
        ys, xs = np.where(g)
        if len(xs):
            cv2.rectangle(out, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), gt_color, max(1, out.shape[1] // 400))
    if binm is not None:
        b = binm.astype(bool)
        out[b] = (out[b] * (1 - alpha) + np.array([255, 40, 40]) * alpha).astype(np.uint8)
    if bbox is not None:
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2] - 1, bbox[3] - 1), (40, 220, 40), max(1, out.shape[1] // 400))
    return out


def disp(img, maxside=1280):
    """Galeri için küçültme (yalnız görüntüleme)."""
    if img is None:
        return None
    h, w = img.shape[:2]
    s = maxside / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


TAU = _load_json(TAU_JSON)
IMG_TAU = _load_json(META / "image_tau.json")
_REF = None


def reference_table():
    global _REF
    if _REF is None:
        p = META / "reference_per_image_s42.csv"
        _REF = pd.read_csv(p, dtype={"new_id": str}) if p.exists() else pd.DataFrame()
        if len(_REF):
            _REF["new_id"] = _REF["new_id"].astype(str).str.zfill(9)
            _REF = _REF.set_index(["split", "new_id", "model", "gt"]).sort_index()
    return _REF


def ref_lookup(split, nid, stem, gt):
    r = reference_table()
    try:
        row = r.loc[(split, nid, stem, gt)]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
    except Exception:
        return None


def default_tau(model_name: str, gt_kind: str):
    stem = model_name.split("/", 1)[-1]
    try:
        return float(TAU[stem][gt_kind])
    except Exception:
        return 0.5


def image_threshold(model_name: str):
    stem = model_name.split("/", 1)[-1]
    if stem in IMG_TAU and isinstance(IMG_TAU[stem], dict):
        return float(IMG_TAU[stem]["thr"]), True
    return 0.5, False


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


# ================================================================== SİMÜLASYON (tek örnek)
SIM_COLS = ["Model", "Görüntü skoru", "Görüntü eşiği", "Karar", "Doğru", "τ piksel", "F1 bbox-GT", "IoU bbox-GT", "AP bbox-GT",
            "Oracle-F1", "F1 piksel-GT", "AP piksel-GT", "Kutu kapsama", "Pozitif piksel %", "Dikdörtgensellik",
            "Ref F1@0.5 (boru hattı)", "Uyg. F1@0.5", "Süre s"]


def evaluate_unet(name, prob, label, gt_bbox, gt_pixel, meta, clean_forged):
    stem = name.split("/", 1)[-1]
    tb = default_tau(name, "bbox"); tp = default_tau(name, "pixel_eps8_morph")
    sc = image_score(prob); it, calibrated = image_threshold(name)
    dec = int(sc >= it)
    binm, _ = postprocess(prob, tb, 0, False)
    rect, pbb = rectangularity(binm)
    row = {"Model": friendly(name), "Görüntü skoru": sc, "Görüntü eşiği": f"{it:.3f}" + ("" if calibrated else " (kalibre değil)"),
           "Karar": "SAHTE" if dec else "GERÇEK", "Doğru": "✓" if dec == label else "✗", "τ piksel": tb,
           "Pozitif piksel %": 100 * binm.mean(), "Dikdörtgensellik": rect}
    if gt_bbox is not None:
        gb = gt_bbox > 127
        row["Kutu kapsama"] = float((binm & gb).sum() / max(gb.sum(), 1))
        if label == 1:
            m = hist_metrics(prob, gb, tb)
            row.update({"F1 bbox-GT": m["F1"], "IoU bbox-GT": m["IoU"], "AP bbox-GT": m["AP"], "Oracle-F1": m["OracleF1"]})
            row["Uyg. F1@0.5"] = hist_metrics(prob, gb, 0.5)["F1"]
            ref = ref_lookup(meta["split"], meta["new_id"], stem, "bbox") if clean_forged else None
            if ref is not None:
                row["Ref F1@0.5 (boru hattı)"] = float(ref["f1_0p5"])
    if gt_pixel is not None and label == 1:
        m = hist_metrics(prob, gt_pixel, tp)
        row.update({"F1 piksel-GT": m["F1"], "AP piksel-GT": m["AP"]})
    return row, binm, pbb


def simulate(key, input_type, degradation, use_pipeline, steps, unet_sel, resnet_sel, stride):
    """Adım adım simülasyon (generator). Her adımda arayüz güncellenir."""
    log, gallery, t_start = [], [], time.time()
    empty = pd.DataFrame(columns=SIM_COLS)
    sent = {"n": -1}

    def emit(df=empty, verdict="", zoom=None, files=None, state=None, final=False):
        # Galeri yalnız yeni görüntü eklendiyse gönderilir (ağ yükünü azaltır); diğer adımlarda gr.skip().
        g = gr.skip() if (len(gallery) == sent["n"] and not final) else list(gallery)
        sent["n"] = len(gallery)
        return ("\n".join(log), g, show_df(df), verdict, zoom, files, state)

    def step(msg):
        log.append(f"- ✓ **{msg}** · {time.time() - t_start:.1f} s")

    key = (key or "").strip()
    if not key:
        raise gr.Error("Önce bir örnek seçin (filtre + 'Rastgele seç' ya da listeden).")
    if key not in dataset_index().index:
        raise gr.Error(f"'{key}' veri setinde yok. Biçim: training:000000123 ya da testing:000000045.")
    if not unet_sel and not resnet_sel:
        raise gr.Error("En az bir model seçin.")
    log.append(f"### Simülasyon · `{key}`")
    log.append("- ⏳ veri yükleniyor…"); yield emit()
    forged, mask, auth, meta = load_dataset_sample(key)
    log.pop(); step(f"1/7 Veri yüklendi — {forged.shape[1]}×{forged.shape[0]} px, GT {'var' if mask is not None else 'yok'}, authentic {'var' if auth is not None else 'yok'}")
    bb = bbox_of(mask)
    gallery.append((disp(forged), "1 · Sahte görüntü (forged)"))
    if mask is not None:
        gallery.append((disp(overlay(forged, None, mask > 127)), "1 · Resmî GT: alan kutusu (bbox)"))
    yield emit()

    # 2 · GT analizi
    gt_pixel, D, gt_note = None, None, ""
    if auth is not None:
        gallery.append((disp(auth), "2 · Authentic eş (orijinal)"))
        gt_pixel, D = derive_pixel_gt(forged, auth)
        if mask is not None:
            gb = mask > 127
            inside = float((D[gb] > 8).mean()) if gb.any() else float("nan")
            outside = int(((D > 8) & ~gb).sum())
            gt_note = f"bbox piksellerinin %{100 * inside:.0f}'i gerçekten değişmiş (ε=8); bbox dışında {outside} değişen piksel"
        gallery.append((disp(cv2.cvtColor(cv2.applyColorMap(np.clip(D.astype(np.int32) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)), "2 · |sahte − authentic| × 4"))
        gallery.append((disp(overlay(forged, None, gt_pixel, gt_color=(255, 200, 0))), "2 · Türetilmiş piksel-değişim GT (eps8_morph)"))
    step("2/7 Ground truth analizi — " + (gt_note or "authentic eş yok → yalnız bbox GT"))
    yield emit()

    # 3 · yeşil işaretçi
    gm = green_marker_mask(forged)
    if auth is not None:
        gm &= ~green_marker_mask(auth)
    n_ring = 0
    if mask is not None:
        m = (mask > 127).astype(np.uint8)
        ring = (cv2.dilate(m, np.ones((13, 13), np.uint8)) > 0) & ~(cv2.erode(m, np.ones((13, 13), np.uint8)) > 0)
        n_ring = int((gm & ring).sum())
    gv = forged.copy(); gv[cv2.dilate(gm.astype(np.uint8), K3) > 0] = [255, 0, 255]
    gallery.append((disp(gv), f"3 · Yeşil işaretçi pikselleri (magenta): {int(gm.sum())} px"))
    step(f"3/7 Yeşil işaretçi taraması — kutu kenarında {n_ring} px → " + ("**kalıntı VAR**" if n_ring >= 20 else "kalıntı yok"))
    yield emit()

    # 4 · girdi
    inp, label, note = build_input(forged, auth, input_type)
    if degradation and not degradation.startswith("Yok"):
        inp = degrade(inp, degradation); note += f" + {degradation}"
    if use_pipeline and steps:
        inp = to_rgb(replay(inp, steps, auth)); note += f" + ön işleme pipeline'ı ({len(steps)} adım)"
    same_geom = inp.shape[:2] == forged.shape[:2]
    gallery.append((disp(inp), f"4 · Model girdisi: {note}"))
    clean_forged = input_type == INPUT_TYPES[0] and degradation.startswith("Yok") and not (use_pipeline and steps)
    step(f"4/7 Model girdisi hazır — {note}; gerçek etiket **{'SAHTE' if label else 'GERÇEK'}**")
    yield emit()

    # 5–6 · çıkarım (tek GPU çağrısı)
    log.append(f"- ⏳ {len(unet_sel)} U-Net + {len(resnet_sel)} ResNet çalışıyor ({'ZeroGPU' if ZERO_GPU else DEVICE})…"); yield emit()
    t_inf = time.time()
    res = gpu_infer(list(unet_sel), list(resnet_sel), [inp], int(stride))[0]
    log.pop(); step(f"5/7 Çıkarım tamamlandı — {time.time() - t_inf:.1f} s ({'ZeroGPU' if ZERO_GPU else DEVICE})")
    rows, panels_zoom, verdicts = [], [], []
    gtb = mask if (mask is not None and same_geom) else None
    gtp = gt_pixel if (gt_pixel is not None and same_geom) else None
    for n in resnet_sel:
        hm, info = res[n]
        smax = max(s for s, _, _ in info["scores"])
        dec = int(smax >= 0.5)
        rows.append({"Model": friendly(n), "Görüntü skoru": smax, "Görüntü eşiği": "0.500", "Karar": "SAHTE" if dec else "GERÇEK",
                     "Doğru": "✓" if dec == label else "✗", "Süre s": info["sec"]})
        ov = cv2.addWeighted(inp, 0.55, heatmap_rgb(hm), 0.45, 0)
        gallery.append((disp(overlay(ov, None, gtb > 127 if gtb is not None else None)), f"5 · {friendly(n)} — patch skor haritası (maks {smax:.2f})"))
        panels_zoom.append((f"{friendly(n)}\npatch skoru", ov))
        verdicts.append((friendly(n), dec, smax, 0.5))
    for n in unet_sel:
        prob, info = res[n]
        row, binm, pbb = evaluate_unet(n, prob, label, gtb, gtp, meta, clean_forged)
        row["Süre s"] = info["sec"]
        rows.append(row)
        gallery.append((disp(heatmap_rgb(prob)), f"6 · {friendly(n)} — olasılık haritası"))
        gallery.append((disp(overlay(inp, binm, gtb > 127 if gtb is not None else None, pbb)), f"6 · {friendly(n)} — τ={row['τ piksel']:.2f} (kırmızı tahmin · mavi GT · yeşil tahmin kutusu)"))
        panels_zoom.append((f"{friendly(n)}\nτ={row['τ piksel']:.2f}", overlay(inp, binm, None, None)))
        verdicts.append((friendly(n), int(row["Karar"] == "SAHTE"), row["Görüntü skoru"], float(row["Görüntü eşiği"].split()[0])))
    df = pd.DataFrame(rows, columns=SIM_COLS)
    step("6/7 Değerlendirme — metrikler hesaplandı")
    yield emit(df=df)

    # 7 · yakın plan + karar + rapor
    zb = zoom_box(bb if same_geom else None, inp.shape)
    pz = [("Model girdisi", inp)]
    if auth is not None and input_type == INPUT_TYPES[0] and same_geom:
        pz.append(("Authentic", auth))
        pz.append(("|sahte − authentic| × 4", cv2.cvtColor(cv2.applyColorMap(np.clip(D.astype(np.int32) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)))
    if gtb is not None:
        pz.append(("GT: bbox", overlay(inp, None, gtb > 127)))
    if gtp is not None:
        pz.append(("GT: piksel-değişim", overlay(inp, None, gtp, gt_color=(255, 200, 0))))
    pz += panels_zoom
    zoom = montage(pz, zb, f"{key} · yakın plan (alan kutusu çevresi) · girdi: {input_type}")
    truth = "SAHTE" if label else "GERÇEK"
    vlines = [f"### Gerçek etiket: **{truth}** — {note}"]
    for name, dec, sc, th in verdicts:
        ok = dec == label
        vlines.append(f"- {'✅' if ok else '❌'} **{name}** → **{'SAHTE' if dec else 'GERÇEK'}** (skor {sc:.3f} {'≥' if dec else '<'} eşik {th:.3f})")
    vlines.append("")
    vlines += insights(df, label, input_type, n_ring, gt_note, degradation, auth is not None, clean_forged)
    verdict = "\n".join(vlines)
    files = write_sim_report(key, meta, input_type, degradation, steps if use_pipeline else [], df, zoom, gallery, verdict)
    step(f"7/7 Rapor hazır — toplam {time.time() - t_start:.1f} s")
    print(f"[sim] {key} · {input_type} · {degradation} · {len(unet_sel)}U+{len(resnet_sel)}R · çıkarım {time.time() - t_inf:.1f}s · toplam {time.time() - t_start:.1f}s", flush=True)
    state = dict(key=key, inp=inp, forged=forged, mask=mask, auth=auth)
    yield emit(df=df, verdict=verdict, zoom=zoom, files=files, state=state, final=True)


def insights(df, label, input_type, n_ring, gt_note, degradation, has_auth, clean_forged):
    out = ["#### Yorum"]
    get = lambda name_part, col: next((r[col] for _, r in df.iterrows() if name_part in r["Model"] and pd.notna(r.get(col))), None)
    rb = get("bbox-GT (s", "Dikdörtgensellik")
    if rb is not None and not math.isnan(rb):
        out.append(f"- **Kutu davranışı:** bbox-GT modelinin tahmini dikdörtgensellik {rb:.2f} — " +
                   ("kutunun tamamını işaretliyor (kutu maskesiyle eğitimin beklenen sonucu)." if rb > 0.7 else "tahmin glif düzeyinde parçalı."))
    f1b, f1p = get("bbox-GT (s", "F1 bbox-GT"), get("bbox-GT (s", "F1 piksel-GT")
    if f1b is not None and f1p is not None:
        out.append(f"- **H1 (GT seçimi):** aynı tahmin bbox-GT ile F1 {f1b:.3f}, piksel-değişim GT ile F1 {f1p:.3f} → fark {f1b - f1p:+.3f}. "
                   "Kutu maskesi, değişmeyen kutu piksellerini de doğru sayar; raporlanan skor GT tanımına bağlıdır.")
    if gt_note:
        out.append(f"- **GT analizi:** {gt_note}.")
    if n_ring >= 20:
        out.append(f"- **Yeşil işaretçi:** kutu kenarında {n_ring} px kalıntı var. 'Sahte, yeşil işaretçi onarılmış' senaryosuyla tekrar çalıştırıp skorun ne kadar düştüğüne bakın (PHASE 2 Ek-C: AP 0,988 → 0,763).")
    if input_type == INPUT_TYPES[3]:
        cov = get("bbox-GT (s", "Kutu kapsama")
        if cov is not None:
            out.append(f"- **Kısayol testi:** içerik değişmediği hâlde bbox-GT modeli kutunun %{100 * cov:.0f}'ini işaretledi" +
                       (" → model içeriği değil işaretçiyi kullanıyor." if cov > 0.3 else " → bu örnekte işaretçi tek başına yeterli olmadı."))
    if input_type == INPUT_TYPES[1]:
        out.append("- **Negatif kontrol:** girdi orijinal belgedir; 'GERÇEK' kararı ve düşük pozitif piksel oranı beklenir.")
    if degradation and not degradation.startswith("Yok"):
        out.append(f"- **Bozulma ({degradation}):** PHASE 2'de ΔAUPRC (bbox modeli) JPEG Q75 −0,154, 0,75× −0,112, blur σ=1 −0,190; 'Yok' ile karşılaştırın.")
    if clean_forged:
        rf, uf = get("bbox-GT (s42)", "Ref F1@0.5 (boru hattı)"), get("bbox-GT (s42)", "Uyg. F1@0.5")
        if rf is not None and uf is not None:
            out.append(f"- **Yeniden üretim:** boru hattının kayıtlı F1 (τ = 0,5) = {rf:.3f}, uygulamanın hesapladığı = {uf:.3f} (fark {uf - rf:+.3f}).")
    if not has_auth:
        out.append("- Authentic eş olmadığı için piksel-değişim GT'si ve işaretçi senaryoları bu örnekte kullanılamaz.")
    return out


def _outdir(prefix):
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d


def write_sim_report(key, meta, input_type, degradation, steps, df, zoom, gallery, verdict):
    d = _outdir("ai515_sim_")
    stem = key.replace(":", "_")
    df.to_csv(d / f"{stem}_metrikler.csv", index=False)
    cv2.imwrite(str(d / f"{stem}_yakin_plan.png"), cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
    rep = dict(zaman=datetime.datetime.now().isoformat(timespec="seconds"), ornek=meta, girdi=input_type, bozulma=degradation,
               on_isleme=[f"{s['name']} — {s['info']}" for s in (steps or [])], cihaz=str(DEVICE), zero_gpu=ZERO_GPU,
               metrikler=json.loads(df.to_json(orient="records", force_ascii=False)), yorum=verdict)
    (d / f"{stem}_rapor.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = [f"# Simülasyon raporu — {key}", "", f"- Zaman: {rep['zaman']}", f"- Girdi senaryosu: {input_type}", f"- Bozulma: {degradation}",
          f"- Ön işleme: {', '.join(rep['on_isleme']) or 'yok'}", "", verdict, "", df.to_markdown(index=False, floatfmt=".3f"), "",
          f"![yakın plan]({stem}_yakin_plan.png)"]
    (d / f"{stem}_rapor.md").write_text("\n".join(md), encoding="utf-8")
    return [str(d / f"{stem}_rapor.md"), str(d / f"{stem}_rapor.json"), str(d / f"{stem}_metrikler.csv"), str(d / f"{stem}_yakin_plan.png")]


# ================================================================== TOPLU TEST
BATCH_MODES = ["Yalnız sahte görüntüler (lokalizasyon)", "Sahte + authentic eşi (dengeli tespit)"]


def batch_test(split, source, tool, area, marker, n, seed, mode, degradation, unet_sel, resnet_sel, max_mp):
    t0 = time.time()
    if not unet_sel and not resnet_sel:
        raise gr.Error("En az bir model seçin.")
    pair_only = mode == BATCH_MODES[1]
    df = filter_index(split, source, tool, area, pair_only, marker, max_mp)
    keys = list(df.index)
    random.Random(int(seed)).shuffle(keys)
    keys = [k for k in keys if sample_exists(k)][: int(n)] if len(keys) else []
    if not keys:
        raise gr.Error("Filtrelere uyan (ve bucket'ta mevcut) örnek yok. Filtreleri gevşetin.")
    rows, msg = [], []
    cols_ps = ["Örnek", "Kaynak", "Üretici", "Alan", "İşaretçi", "Girdi", "Etiket", "Model", "Görüntü skoru", "Karar", "Doğru",
               "F1 bbox-GT", "IoU bbox-GT", "AP bbox-GT", "F1 piksel-GT", "AP piksel-GT", "Uyg. F1@0.5", "Ref F1@0.5", "Uyg. AP", "Ref AP", "Pozitif piksel %"]
    for i, k in enumerate(keys, 1):
        try:
            forged, mask, auth, meta = load_dataset_sample(k)
        except Exception as e:
            msg.append(f"{k}: {e}"); continue
        inputs = [("sahte", degrade(forged, degradation), 1)]
        if pair_only and auth is not None:
            inputs.append(("authentic", degrade(auth, degradation), 0))
        gtp = derive_pixel_gt(forged, auth)[0] if auth is not None else None
        res = gpu_infer(list(unet_sel), list(resnet_sel), [x[1] for x in inputs], 112)
        clean = degradation.startswith("Yok")
        for (kind, img, label), rr in zip(inputs, res):
            base = {"Örnek": k, "Kaynak": meta.get("source"), "Üretici": meta.get("tool"), "Alan": meta.get("area_bin"),
                    "İşaretçi": "var" if meta.get("has_marker_index") else "yok", "Girdi": kind, "Etiket": label}
            for nme in resnet_sel:
                hm, info = rr[nme]
                smax = max(s for s, _, _ in info["scores"])
                rows.append({**base, "Model": friendly(nme), "Görüntü skoru": smax, "Karar": int(smax >= 0.5), "Doğru": int((smax >= 0.5) == label)})
            for nme in unet_sel:
                prob, _ = rr[nme]
                stem = nme.split("/", 1)[-1]
                sc = image_score(prob); it, _ = image_threshold(nme); dec = int(sc >= it)
                tb, tp = default_tau(nme, "bbox"), default_tau(nme, "pixel_eps8_morph")
                r = {**base, "Model": friendly(nme), "Görüntü skoru": sc, "Karar": dec, "Doğru": int(dec == label)}
                binm, _ = postprocess(prob, tb, 0, False)
                r["Pozitif piksel %"] = 100 * binm.mean()
                if label == 1 and mask is not None:
                    gb = mask > 127
                    m = hist_metrics(prob, gb, tb); m5 = hist_metrics(prob, gb, 0.5)
                    r.update({"F1 bbox-GT": m["F1"], "IoU bbox-GT": m["IoU"], "AP bbox-GT": m["AP"], "Uyg. F1@0.5": m5["F1"], "Uyg. AP": m["AP"]})
                    ref = ref_lookup(meta["split"], meta["new_id"], stem, "bbox") if clean else None
                    if ref is not None:
                        r.update({"Ref F1@0.5": float(ref["f1_0p5"]), "Ref AP": float(ref["ap_img"])})
                    if gtp is not None:
                        mp_ = hist_metrics(prob, gtp, tp)
                        r.update({"F1 piksel-GT": mp_["F1"], "AP piksel-GT": mp_["AP"]})
                rows.append(r)
        per = pd.DataFrame(rows, columns=cols_ps)
        agg = aggregate(per)
        prog = f"**{i}/{len(keys)}** örnek işlendi · {time.time() - t0:.0f} s · {'ZeroGPU' if ZERO_GPU else DEVICE}" + (f"  \n⚠ {len(msg)} örnek atlandı" if msg else "")
        yield prog, show_df(per), show_df(agg), None, None
    per = pd.DataFrame(rows, columns=cols_ps)
    agg = aggregate(per)
    fig = batch_figure(per)
    d = _outdir("ai515_batch_")
    per.to_csv(d / "toplu_test_ornek_bazinda.csv", index=False); agg.to_csv(d / "toplu_test_ozet.csv", index=False)
    cfg = dict(zaman=datetime.datetime.now().isoformat(timespec="seconds"), filtre=dict(split=split, source=source, tool=tool, area=area, marker=marker, max_mp=max_mp),
               n=len(keys), seed=int(seed), mod=mode, bozulma=degradation, unet=list(unet_sel), resnet=list(resnet_sel), cihaz=str(DEVICE), zero_gpu=ZERO_GPU,
               sure_s=round(time.time() - t0, 1), ornekler=keys, ozet=json.loads(agg.to_json(orient="records", force_ascii=False)))
    (d / "toplu_test.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    fig.savefig(d / "toplu_test_grafik.png", dpi=120)
    prog = (f"### Tamamlandı — {len(keys)} örnek, {time.time() - t0:.0f} s ({'ZeroGPU' if ZERO_GPU else DEVICE})\n"
            "Referans (rapor, resmî test 812 görüntü, 3 tohum): bbox-GT modeli F1@τ* 0,846 · AUPRC 0,950; aynı tahminler piksel-GT ile F1 ≈ 0,45. "
            "Küçük örneklemde değerler örnekleme hatası içerir; 'Ref' sütunları aynı görüntülerde boru hattının kayıtlı sonuçlarıdır.")
    yield prog, show_df(per), show_df(agg), fig, [str(d / f) for f in ("toplu_test_ozet.csv", "toplu_test_ornek_bazinda.csv", "toplu_test.json", "toplu_test_grafik.png")]


def aggregate(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    from sklearn.metrics import roc_auc_score
    for m, d in per.groupby("Model", sort=False):
        f = d[d["Etiket"] == 1]; a = d[d["Etiket"] == 0]
        r = {"Model": m, "n sahte": len(f), "n authentic": len(a)}
        for c in ("F1 bbox-GT", "IoU bbox-GT", "AP bbox-GT", "F1 piksel-GT", "AP piksel-GT"):
            v = pd.to_numeric(f[c], errors="coerce").dropna()
            r[c] = round(float(v.mean()), 4) if len(v) else None
        r["Tespit TPR"] = round(float(f["Karar"].mean()), 3) if len(f) else None
        r["Yanlış alarm FPR"] = round(float(a["Karar"].mean()), 3) if len(a) else None
        r["Doğruluk"] = round(float(d["Doğru"].mean()), 3)
        try:
            r["Görüntü AUROC"] = round(float(roc_auc_score(d["Etiket"], d["Görüntü skoru"])), 3) if (len(f) and len(a)) else None
        except Exception:
            r["Görüntü AUROC"] = None
        both = f.dropna(subset=["Ref F1@0.5", "Uyg. F1@0.5"]) if "Ref F1@0.5" in f else f.iloc[0:0]
        r["Yeniden üretim n"] = len(both)
        r["|ΔF1@0.5| ort."] = round(float((both["Uyg. F1@0.5"] - both["Ref F1@0.5"]).abs().mean()), 4) if len(both) else None
        r["|ΔAP| ort."] = round(float((both["Uyg. AP"] - both["Ref AP"]).abs().mean()), 4) if len(both) else None
        out.append(r)
    return pd.DataFrame(out)


def batch_figure(per):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=100)
    f = per[per["Etiket"] == 1]
    models = list(dict.fromkeys(per["Model"]))
    x = np.arange(len(models)); w = 0.38
    vb = [pd.to_numeric(f[f["Model"] == m]["F1 bbox-GT"], errors="coerce").mean() for m in models]
    vp = [pd.to_numeric(f[f["Model"] == m]["F1 piksel-GT"], errors="coerce").mean() for m in models]
    ax[0].bar(x - w / 2, np.nan_to_num(vb), w, label="bbox-GT", color="#4C72B0")
    ax[0].bar(x + w / 2, np.nan_to_num(vp), w, label="piksel-değişim GT", color="#DD8452")
    ax[0].set_xticks(x); ax[0].set_xticklabels([m.replace(" · ", "\n", 1) for m in models], fontsize=7)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("ortalama F1@τ"); ax[0].set_title("Lokalizasyon: aynı tahmin, iki GT", fontsize=9); ax[0].legend(fontsize=7)
    for i, m in enumerate(models):
        d = per[per["Model"] == m]
        for lab, col in ((1, "#C44E52"), (0, "#55A868")):
            s = d[d["Etiket"] == lab]["Görüntü skoru"]
            ax[1].scatter(np.full(len(s), i) + (0.12 if lab else -0.12), s, s=12, color=col, alpha=0.7, label=("sahte" if lab else "authentic") if i == 0 else None)
    ax[1].set_xticks(range(len(models))); ax[1].set_xticklabels([m.replace(" · ", "\n", 1) for m in models], fontsize=7)
    ax[1].set_ylabel("görüntü skoru (top-%1)"); ax[1].set_title("Görüntü-düzeyi skorlar", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


# ================================================================== DİĞER SEKMELERİN MANTIĞI
def _info_md(img, mask, auth):
    if img is None:
        return "Görüntü yok."
    h, w = img.shape[:2]
    s = f"**Boyut:** {w}×{h} px ({w * h / 1e6:.2f} MP) · **kanal:** {1 if img.ndim == 2 else img.shape[2]}"
    if mask is not None:
        m = mask > 127
        s += f"  \n**GT maskesi:** {int(m.sum())} px (%{100 * m.mean():.2f} alan)"
        rect, bb = rectangularity(m)
        if bb:
            s += f", bbox {bb[2] - bb[0]}×{bb[3] - bb[1]} (dikdörtgensellik {rect:.2f})"
    if auth is not None:
        s += f"  \n**Authentic eş:** {auth.shape[1]}×{auth.shape[0]} " + ("✓ aynı boyut" if auth.shape[:2] == img.shape[:2] else "⚠ boyut farklı")
    return s


def on_upload(img, mask, auth):
    if img is None:
        raise gr.Error("Önce bir görüntü yükleyin.")
    img = to_rgb(img)
    if mask is not None:
        mask = to_gray(mask)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if auth is not None:
        auth = to_rgb(auth)
        if auth.shape != img.shape:
            gr.Warning("Authentic eş farklı boyutta; piksel-değişim GT'si ve işaretçi işlemleri devre dışı.")
            auth = None
    return img, img, mask, auth, [], _info_md(img, mask, auth), img, img, mask, auth


def on_sample(key):
    key = (key or "").strip()
    if not key or key not in dataset_index().index:
        raise gr.Error("Geçerli bir örnek seçin (ör. testing:000000045).")
    img, mask, auth, meta = load_dataset_sample(key)
    return img, img, mask, auth, [], meta_md(meta, img, mask, auth) + "  \n" + _info_md(img, mask, auth), img, img, mask, auth



# ------------------------------------------------------------------ 2 · Girdi › TASLAKLAR
# Her taslak bir klasör: image.png · mask.png (ops.) · auth.png (ops.) · meta.json (ad, tarih, kaynak, pipeline adımları).
# Konum: AI515_DRAFTS_DIR > yerelde <proje>/drafts > Space'te /data/drafts (yazılabilirse) > geçici klasör.
def _drafts_dir() -> Path:
    cands = [os.environ.get("AI515_DRAFTS_DIR")]
    cands += [str(DATA_ROOT / "drafts"), str(Path(tempfile.gettempdir()) / "ai515_drafts")] if IN_SPACE else [str(ROOT / "drafts")]
    for c in cands:
        if not c:
            continue
        d = Path(c)
        try:
            d.mkdir(parents=True, exist_ok=True)
            t = d / ".write_test"; t.write_text("ok"); t.unlink()
            return d
        except Exception:
            continue
    raise RuntimeError("Taslak klasörü oluşturulamadı.")


DRAFTS = None
_DRAFT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def _drafts() -> Path:
    global DRAFTS
    if DRAFTS is None:
        DRAFTS = _drafts_dir()
    return DRAFTS


def _slug(name: str) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ ", "cgiosuCGIOSU_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", (name or "").translate(tr))[:40].strip("_-")
    return s or "taslak"


def _png_write(path: Path, arr: np.ndarray):
    a = arr if arr.ndim == 2 else cv2.cvtColor(to_rgb(arr), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise gr.Error(f"PNG yazılamadı: {path.name}")
    path.write_bytes(buf.tobytes())


def _png_read(path: Path, gray=False):
    if not path.exists():
        return None
    buf = np.frombuffer(path.read_bytes(), np.uint8)
    a = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return a if gray else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def draft_list():
    """[(etiket, kimlik)] — en yeni önce."""
    out = []
    try:
        root = _drafts()
    except Exception:
        return out
    for d in root.iterdir():
        mj = d / "meta.json"
        if not (d.is_dir() and _DRAFT_ID_RE.match(d.name) and mj.exists()):
            continue
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = [t for t, k in (("maske", "has_mask"), ("eş", "has_auth")) if m.get(k)]
        n = len(m.get("steps") or [])
        if n:
            tags.append(f"{n} adım")
        lbl = f"{m.get('name', d.name)} · {m.get('created', '')[:16].replace('T', ' ')} · {m.get('w')}×{m.get('h')}" + (f" · {', '.join(tags)}" if tags else "")
        out.append((m.get("created", ""), lbl, d.name))
    out.sort(reverse=True)
    return [(lbl, i) for _, lbl, i in out]


def _draft_dd(value=None):
    ch = draft_list()
    ids = [i for _, i in ch]
    return gr.Dropdown(choices=ch, value=value if value in ids else (ids[0] if ids else None))


def draft_save(name, orig, mask, auth, steps, up_img, up_mask, up_auth, sample_key):
    """Girdi sekmesinde yüklü olanı (yoksa yükleme alanlarındakini) taslak olarak kaydeder."""
    src = "yüklü girdi"
    if orig is None:
        if up_img is None:
            raise gr.Error("Kaydedilecek görüntü yok: önce görüntü yükleyin ya da örnek seçin.")
        orig, _, mask, auth, steps, *_ = on_upload(up_img, up_mask, up_auth)
        src = "yükleme alanları"
    did = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{_slug(name)}"
    d, k = _drafts() / did, 1
    while d.exists():
        k += 1; d = _drafts() / f"{did}_{k}"
    did = d.name
    d.mkdir(parents=True)
    _png_write(d / "image.png", orig)
    if mask is not None:
        _png_write(d / "mask.png", to_gray(mask))
    if auth is not None:
        _png_write(d / "auth.png", auth)
    h, w = orig.shape[:2]
    meta = dict(name=(name or "").strip() or did, created=datetime.datetime.now().isoformat(timespec="seconds"),
                w=w, h=h, has_mask=mask is not None, has_auth=auth is not None,
                sample_key=(sample_key or "").strip() or None, steps=list(steps or []), source=src)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    msg = f"💾 Taslak kaydedildi: **{meta['name']}** ({w}×{h}" + (", maske" if meta["has_mask"] else "") + \
          (", authentic eş" if meta["has_auth"] else "") + (f", {len(meta['steps'])} pipeline adımı" if meta["steps"] else "") + \
          f") · kaynak: {src} · `{d}`"
    return _draft_dd(did), msg


def draft_load(did):
    if not did or not _DRAFT_ID_RE.match(str(did)) or not (_drafts() / did / "meta.json").exists():
        raise gr.Error("Geçerli bir taslak seçin.")
    d = _drafts() / did
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    img, mask, auth = _png_read(d / "image.png"), _png_read(d / "mask.png", gray=True), _png_read(d / "auth.png")
    if img is None:
        raise gr.Error("Taslağın görüntü dosyası eksik.")
    steps = list(meta.get("steps") or [])
    cur, note = img, ""
    if steps:
        try:
            cur = replay(img, steps, auth)
        except Exception as e:
            steps, note = [], f"  \n⚠ Pipeline adımları yeniden uygulanamadı ({e}); orijinal görüntü yüklendi."
    info = f"📂 **Taslak:** {meta.get('name', did)} · {meta.get('created', '')[:16].replace('T', ' ')}" + \
           (f" · örnek `{meta['sample_key']}`" if meta.get("sample_key") else "") + "  \n" + _info_md(img, mask, auth) + note
    status = f"📂 Yüklendi: **{meta.get('name', did)}**" + (f" ({len(steps)} pipeline adımı geri yüklendi)" if steps else "")
    return (img, cur, mask, auth, steps, info, img, cur, mask, auth, steps_md(steps), img, mask, auth, status)


def draft_delete(did):
    if not did or not _DRAFT_ID_RE.match(str(did)):
        raise gr.Error("Silinecek taslağı seçin.")
    import shutil
    d = _drafts() / did
    if d.is_dir():
        shutil.rmtree(d)
    return _draft_dd(), f"🗑 Taslak silindi: `{did}`"


def op_choices(cat):
    names = list(OPS[cat].keys())
    return gr.Dropdown(choices=names, value=names[0]), *param_updates(cat, names[0])


def param_updates(cat, name):
    spec = OPS[cat][name]
    ups = []
    for i in range(3):
        if i < len(spec["params"]):
            _, label, mn, mx, st, dv = spec["params"][i]
            ups.append(gr.Slider(label=label, minimum=mn, maximum=mx, step=st, value=dv, visible=True))
        else:
            ups.append(gr.Slider(visible=False))
    if spec["modes"]:
        ups.append(gr.Dropdown(choices=spec["modes"], value=spec["modes"][0], visible=True, label="mod"))
    else:
        ups.append(gr.Dropdown(choices=[], value=None, visible=False))
    ups.append(spec["doc"] or " ")
    return tuple(ups)


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


def preview(cur, cat, name, p1, p2, p3, mode, auth):
    out, info = run_op(cur, cat, name, p1, p2, p3, mode, auth)
    return out, hist_figure(cur, out), f"Önizleme: {name} — {info}"


def apply_step(cur, orig, steps, cat, name, p1, p2, p3, mode, auth):
    out, info = run_op(cur, cat, name, p1, p2, p3, mode, auth)
    steps = list(steps) + [dict(cat=cat, name=name, info=info, p=[p1, p2, p3], mode=mode)]
    return out, out, steps, steps_md(steps), hist_figure(cur, out), f"Uygulandı: {name} — {info}"


def steps_md(steps):
    if not steps:
        return "_Pipeline boş — orijinal görüntü._"
    return "\n".join(f"{i + 1}. **{s['name']}** ({s['cat'].split('·')[1].strip()}) — {s['info']}" for i, s in enumerate(steps))


def replay(start, steps, auth):
    cur = start
    for s in steps:
        cur, _ = run_op(cur, s["cat"], s["name"], *s["p"], s["mode"], auth)
    return cur


def undo(orig, steps, auth):
    steps = list(steps)[:-1]
    cur = replay(orig, steps, auth) if orig is not None else None
    return cur, cur, steps, steps_md(steps), "Son adım geri alındı."


def reset(orig):
    return orig, orig, [], steps_md([]), "Pipeline sıfırlandı."


def apply_preset(orig, preset, auth):
    seq = PRESETS.get(preset, [])
    if orig is None:
        raise gr.Error("Önce bir görüntü yükleyin.")
    if not seq:
        return orig, orig, [], steps_md([]), None, "Preset seçilmedi."
    cur, steps = orig, []
    for cat, name, kw in seq:
        spec = OPS[cat][name]
        p = [sp[5] for sp in spec["params"]] + [0] * (3 - len(spec["params"]))
        for i, key in enumerate(("p1", "p2", "p3")):
            if key in kw:
                p[i] = kw[key]
        mode = kw.get("mode", spec["modes"][0] if spec["modes"] else None)
        cur, info = run_op(cur, cat, name, *p, mode, auth)
        steps.append(dict(cat=cat, name=name, info=info, p=p, mode=mode))
    return cur, cur, steps, steps_md(steps), hist_figure(orig, cur), f"Preset uygulandı: {preset}"


def classical_features(cur, mask):
    if cur is None:
        raise gr.Error("Görüntü yok.")
    if not FC_OK:
        return pd.DataFrame({"hata": ["baselines/features_classical.py bulunamadı"]})
    rgb = to_rgb(cur)
    h, w = rgb.shape[:2]
    S = min(256, h, w)
    bb = bbox_of(mask) if (mask is not None and mask.shape == rgb.shape[:2]) else None
    if bb:
        cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
    else:
        cx, cy = w // 2, h // 2
    x0, y0 = int(np.clip(cx - S // 2, 0, w - S)), int(np.clip(cy - S // 2, 0, h - S))
    f = FC.extract_features(rgb[y0:y0 + S, x0:x0 + S])
    df = pd.DataFrame({"öznitelik": FC.feature_names(), "değer": np.round(f, 5)})
    df.insert(0, "grup", df["öznitelik"].str.split("_").str[0])
    return df


def run_unet(orig, cur, mask, auth, model_name, source, thr, use_otsu, morph_k, gt_kind):
    if not TORCH_OK:
        raise gr.Error(f"torch yüklü değil: {_TORCH_ERR}")
    if orig is None:
        raise gr.Error("Görüntü yok. Girdi sekmesinden yükleyin ya da Simülasyon'dan aktarın.")
    if not model_name:
        raise gr.Error("Model seçin (models/ klasöründe unet_*.pt bulunamadı).")
    img = to_rgb(orig if source == "Orijinal görüntü" else cur)
    prob, info = gpu_infer([model_name], [], [img])[0][model_name]
    binm, thr_used = postprocess(prob, thr, int(morph_k), use_otsu)
    rect, bb = rectangularity(binm)
    sc = image_score(prob); it, cal = image_threshold(model_name)
    gt, gt_note = None, ""
    if gt_kind.startswith("piksel"):
        if auth is not None and auth.shape[:2] == img.shape[:2]:
            gt, _ = derive_pixel_gt(to_rgb(orig), auth)
            gt_note = "GT: piksel-değişim (D>8, morfolojik)"
        else:
            gt_note = "⚠ piksel GT için aynı boyutta authentic gerekir; GT yok."
    elif mask is not None and mask.shape == img.shape[:2]:
        gt = mask > 127; gt_note = "GT: resmî bbox maskesi"
    elif mask is not None:
        gt_note = "⚠ GT boyutu görüntüyle uyuşmuyor (geometrik adım?) — metrik hesaplanmadı."
    rows = [("model", friendly(model_name)), ("girdi", source), ("cihaz", "ZeroGPU" if ZERO_GPU else str(DEVICE)),
            ("pencere sayısı", info["n_windows"]), ("süre (s)", f"{info['sec']:.2f}"),
            ("eşik", f"{thr_used:.2f}" + (" (Otsu)" if use_otsu else "")),
            ("pozitif piksel", f"{int(binm.sum())} (%{100 * binm.mean():.3f})"),
            ("tahmin dikdörtgenselliği", fmt(rect, 2)), ("görüntü skoru (top-%1)", f"{sc:.3f}"),
            ("görüntü kararı", f"{'SAHTE' if sc >= it else 'GERÇEK'} (eşik {it:.3f}{'' if cal else ', kalibre değil'})")]
    if gt is not None:
        m = hist_metrics(prob, gt, thr_used)
        rows += [("F1@eşik", fmt(m["F1"])), ("IoU@eşik", fmt(m["IoU"])), ("AP (görüntü, kesin)", fmt(m["AP"])),
                 ("Oracle-F1", fmt(m["OracleF1"])), ("TP/FP/FN", f"{m['TP']}/{m['FP']}/{m['FN']}")]
    df = pd.DataFrame(rows, columns=["ölçüt", "değer"])
    ov = overlay(img, binm, gt, bb)
    return disp(heatmap_rgb(prob)), disp(ov), disp((binm * 255).astype(np.uint8)), df, "Kırmızı: tahmin · Mavi: GT · Yeşil kutu: tahmin bbox. " + gt_note, None


def compare_unet(orig, cur, mask, auth, model_name, thr, use_otsu, morph_k, gt_kind, steps):
    if not steps:
        raise gr.Error("Pipeline boş; karşılaştırma için önce ön işleme adımı ekleyin.")
    if to_rgb(cur).shape[:2] != to_rgb(orig).shape[:2]:
        raise gr.Error("Pipeline çıktısı farklı boyutta; geometrik adımlarla karşılaştırma desteklenmiyor.")
    a = run_unet(orig, cur, mask, auth, model_name, "Orijinal görüntü", thr, use_otsu, morph_k, gt_kind)
    b = run_unet(orig, cur, mask, auth, model_name, "Pipeline çıktısı", thr, use_otsu, morph_k, gt_kind)
    da, db = a[3].set_index("ölçüt")["değer"], b[3].set_index("ölçüt")["değer"]
    df = pd.DataFrame({"ölçüt": da.index, "orijinal": da.values, "pipeline": db.reindex(da.index).values})
    return a[1], b[1], df, f"Pipeline: {steps_md(steps)}"


def run_resnet(orig, cur, mask, model_name, source, stride):
    if not TORCH_OK:
        raise gr.Error(f"torch yüklü değil: {_TORCH_ERR}")
    if orig is None:
        raise gr.Error("Görüntü yok.")
    if not model_name:
        raise gr.Error("ResNet checkpoint'i bulunamadı.")
    img = to_rgb(orig if source == "Orijinal görüntü" else cur)
    hm, info = gpu_infer([], [model_name], [img], int(stride))[0][model_name]
    scores = info["scores"]
    smax = max(s for s, _, _ in scores)
    rows = [("model", friendly(model_name)), ("girdi", source), ("patch sayısı", len(scores)), ("süre (s)", f"{info['sec']:.2f}"),
            ("görüntü skoru (maks patch)", f"{smax:.3f}"), ("karar (eşik 0.5)", "SAHTE" if smax >= 0.5 else "GERÇEK"),
            ("ortalama patch skoru", f"{np.mean([s for s, _, _ in scores]):.3f}")]
    if mask is not None and mask.shape == img.shape[:2]:
        bb = bbox_of(mask)
        if bb:
            x0b, y0b, x1b, y1b = bb
            hit = lambda y, x: not (x + 224 <= x0b or x >= x1b or y + 224 <= y0b or y >= y1b)
            inter = [s for s, y, x in scores if hit(y, x)]; clean = [s for s, y, x in scores if not hit(y, x)]
            rows += [("GT ile kesişen patch skoru (ort/maks)", f"{np.mean(inter):.3f} / {np.max(inter):.3f}" if inter else "—"),
                     ("temiz patch skoru (ort/maks)", f"{np.mean(clean):.3f} / {np.max(clean):.3f}" if clean else "—")]
    ov = cv2.addWeighted(img, 0.55, heatmap_rgb(hm), 0.45, 0)
    if mask is not None and mask.shape == img.shape[:2]:
        ov = overlay(ov, None, mask > 127)
    return disp(ov), pd.DataFrame(rows, columns=["ölçüt", "değer"])


def run_green(cur, mask, auth):
    if cur is None:
        raise gr.Error("Görüntü yok.")
    rgb = to_rgb(cur)
    gm = green_marker_mask(rgb)
    if auth is not None and auth.shape == rgb.shape:
        gm &= ~green_marker_mask(auth)
    txt = f"Yeşil işaretçi pikseli: **{int(gm.sum())}**"
    if mask is not None and mask.shape == rgb.shape[:2]:
        m = (mask > 127).astype(np.uint8)
        ring = (cv2.dilate(m, np.ones((13, 13), np.uint8)) > 0) & ~(cv2.erode(m, np.ones((13, 13), np.uint8)) > 0)
        nr = int((gm & ring).sum())
        txt += f" · bbox kenar halkasında (±6 px): **{nr}** · bbox içinde: {int((gm & (m > 0)).sum())}"
        txt += "  \n→ " + ("**kalıntı var** (eşik ≥20 px; sahtelerin %75,4'ünde görülen üretim-hattı artefaktı)" if nr >= 20 else "kalıntı eşiğin altında")
    ov = rgb.copy(); ov[cv2.dilate(gm.astype(np.uint8), K3) > 0] = [255, 0, 255]
    return disp(ov), txt


def results_markdown():
    parts = []
    p3 = META / "phase3_tables.md"
    if p3.exists():  # Faz 3: YOLO11, DeepLabV3+, işaretçisiz eğitim, sondalar (bildiri: paper/main.pdf)
        parts.append(p3.read_text(encoding="utf-8") + "\n\n---\n")
    v2 = META / "v2_tables.md"
    if v2.exists():  # AIForge-Doc v2 (GPT-Image-2) harici test: v1 modelleri, yeniden eğitim yok
        parts.append(v2.read_text(encoding="utf-8") + "\n\n---\n")
    if RESULTS_MD.exists():
        parts.append("## Küçük-veri (smoke) baseline sonuçları\n")
        parts.append(RESULTS_MD.read_text(encoding="utf-8"))
    if P2_SUMMARY.exists():
        df = pd.read_csv(P2_SUMMARY)
        df = df[df["subset"] == "all"][["train_gt", "gt", "auprc_pooled_mean", "auprc_pooled_std", "f1_tau_mean", "f1_tau_std", "iou_tau_mean", "n_images"]].round(3)
        parts.append("\n\n## PHASE 2 — resmî test (812), 3 tohum: eğitim GT × değerlendirme GT — `summary_test_agg.csv`\n")
        parts.append(df.to_markdown(index=False))
    if IMG_TAU:
        rows = [(friendly('PHASE2/' + k), v["thr"], v["auroc"], v["tpr"], v["fpr"]) for k, v in IMG_TAU.items() if isinstance(v, dict)]
        parts.append("\n\n## Görüntü-düzeyi karar eşikleri (val 487 sahte + 487 eşleşen authentic, Youden-J)\n")
        parts.append(pd.DataFrame(rows, columns=["model", "eşik", "AUROC", "TPR", "FPR"]).to_markdown(index=False))
    return "\n".join(parts) if parts else "Sonuç dosyaları bulunamadı."


# ---------------- simülasyon sekmesi yardımcıları
def sim_filter(split, source, tool, area, pair_only, marker, max_mp):
    df = filter_index(split, source, tool, area, pair_only, marker, max_mp)
    ch = sample_choices(df)
    return gr.Dropdown(choices=ch, value=None), f"{len(df)} örnek filtreye uyuyor."


def sim_random(split, source, tool, area, pair_only, marker, max_mp):
    df = filter_index(split, source, tool, area, pair_only, marker, max_mp)
    keys = list(df.index)
    random.shuffle(keys)
    for k in keys[:60]:
        if sample_exists(k):
            return gr.Dropdown(choices=sample_choices(df), value=k), f"{len(df)} örnek filtreye uyuyor · rastgele seçildi: `{k}`"
    raise gr.Error("Filtreye uyan, bucket'ta mevcut örnek bulunamadı.")


def sim_preview(key):
    key = (key or "").strip()
    if not key:
        return None, None, "Örnek seçilmedi."
    if key not in dataset_index().index:
        return None, None, f"⚠ '{key}' veri setinde yok (biçim: testing:000000045)."
    if not sample_exists(key):
        return None, None, f"⚠ '{key}' dosyası veri kökünde bulunamadı."
    img, mask, auth, meta = load_dataset_sample(key)
    return disp(img, 900), disp(overlay(img, None, mask > 127) if mask is not None else img, 900), meta_md(meta, img, mask, auth)


def sim_to_tabs(state):
    if not state:
        raise gr.Error("Önce bir simülasyon çalıştırın.")
    img, mask, auth = state["forged"], state["mask"], state["auth"]
    meta = load_dataset_sample(state["key"])[3]
    gr.Info(f"{state['key']} diğer sekmelere aktarıldı.")
    return img, img, mask, auth, [], meta_md(meta, img, mask, auth), img, img, mask, auth, steps_md([])




# ================================================================== XAI · SHAP AÇIKLAMALARI (sekme 8 › SHAP Analysis)
# Ayrı paket: app/xai (registry · targets · adapters · manager · viz · compare · service). Ayrıntı: docs/XAI.md
try:
    sys.path.insert(0, str(APP_DIR))
    import xai as XAI
    XAI_OK, _XAI_ERR = True, ""
except Exception as _e:  # xai yoksa diğer sekmeler etkilenmez
    XAI, XAI_OK, _XAI_ERR = None, False, str(_e)
try:
    import shap as _shap  # noqa: F401
    SHAP_OK, _SHAP_ERR = True, ""
except Exception as _e:
    SHAP_OK, _SHAP_ERR = False, str(_e)

XAI_SOURCES = ["Girdi sekmesi (orijinal)", "Ön işleme pipeline çıktısı", "Son simülasyon girdisi", "Yeni yükleme"]
# Tam görüntü hedefleri (xai/fullimage.py): açıklama orijinal görüntünün TAMAMINI tek hedef + tek referansla kapsar.
# Döşemeli mozaik yerel pencere açıklamalarının birleşimidir (tabanlar farklı) → tam görüntü SHAP olarak sunulmaz.
XT_FULL_DEC = "Tam görüntü · karar hedefi (önerilen)"
XT_FULL_DET = "Tam görüntü · tespitler (ilk k bölge / patch / kutu)"
XT_FULL_GT = "Tam görüntü · GT kutusu"
XT_TILES = "Döşemeli yerel mozaik (512 px pencereler · tam görüntü SHAP DEĞİLDİR)"
XT_WIN_DEC = "Pencere · karar hedefi (512 px pencere / 224 patch)"
XT_WIN_DET = "Pencere · tespitler (ilk k bölge / patch)"
XT_WIN_GT = "Pencere · GT kutusu"
XT_COMMON = "Ortak pencere (karşılaştırma)"
XAI_TARGETS = [XT_FULL_DEC, XT_FULL_DET, XT_FULL_GT, XT_TILES, XT_WIN_DEC, XT_WIN_DET, XT_WIN_GT, XT_COMMON]
XAI_METHODS = ["Partition (önerilen)", "Gradient (ince doku)"]
XAI_REFS = ["Otomatik (authentic eş varsa o, yoksa blur)", "authentic eş", "blur(32,32)", "inpaint_telea"]
_XAI_REF_KEY = {XAI_REFS[0]: "auto", XAI_REFS[1]: "authentic", XAI_REFS[2]: "blur", XAI_REFS[3]: "inpaint"}
XAI_DEPS = None
XAI_SPECS: dict = {}
XAI_MGR = None


def xai_setup():
    """Modeller yüklendikten sonra çağrılır: bağımlılık enjeksiyonu + model kaydı + yönetici."""
    global XAI_DEPS, XAI_SPECS, XAI_MGR
    if not (XAI_OK and TORCH_OK):
        return
    XAI_DEPS = XAI.Deps(get_model=get_model, norm_chw=norm_chw, project_freq_channels=project_freq_channels,
                        windows=_windows, autocast=_autocast, device=DEVICE, unet_window_forward=_unet_window_forward,
                        unet_prepare=_unet_prepare, image_score=image_score, q_u8=q_u8, postprocess=postprocess,
                        default_tau=default_tau, image_threshold=image_threshold, friendly=friendly,
                        list_unet=list_seg_ckpts, list_resnet=list_resnet_ckpts, green_marker_mask=green_marker_mask)
    XAI_SPECS = XAI.build_registry(XAI_DEPS, _REG)
    try:  # YOLO11 kutu dedektörleri: tam görüntü kutu-güveni hedefleri (pencere adaptörleri piksel/patch modelleri içindir)
        XAI_SPECS.update(XAI.fullimage.yolo_specs([n for n in list_unet_ckpts() if n.startswith("YOLO/")], XAI_DEPS))
    except Exception as _e:
        print("[XAI] YOLO kaydı yapılamadı:", _e)
    XAI_MGR = XAI.XAIManager(lambda: XAI_SPECS, gpu_explain)


def _xai_is_full(t):
    return XAI.fullimage.is_full(t.kind) or t.kind == XAI.fullimage.TILE_KIND


def _xai_full_per_eval(t, sp):
    """Tam görüntü işinde değerlendirme başına tahmini süre (s) — ölçüm (RTX 5080): 3 kanal U-Net ≈ 6 ms/pencere,
    5 kanal U-Net ≈ 0.25 s/MP (tam görüntü frekans kanalları, CPU), ResNet ≈ 0.4 ms/patch, YOLO ≈ 25 ms/kare."""
    F = XAI.fullimage
    if t.kind == F.TILE_KIND:
        n = len(F.tile_grid(max(t.h, 512), max(t.w, 512)))
        return n * (0.003 if sp.family == "resnet" else (0.016 if sp.in_ch == 5 else 0.004))
    if sp.family == "resnet":
        return 0.0004 * max(1, len(t.patches or []))
    if sp.family == "yolo":
        n = 1 if "-full-" in t.model else len(_windows(max(t.h, 1024), max(t.w, 1024), 1024, 768))
        return 0.025 * n
    n = len(_windows(max(t.h, 512), max(t.w, 512), 512, 448))
    if sp.in_ch == 5:  # frekans kanalları (FFT-HP + Laplacian) her değerlendirmede tüm görüntüde CPU'da yeniden hesaplanır
        return n * 0.006 + 0.25 * t.h * t.w / 1e6
    return n * 0.006


def _xai_job_cost(t, cfg):
    sp = XAI_SPECS.get(t.model)
    if sp is None:
        return 1.0
    if _xai_is_full(t):
        return 5 + int(getattr(t, "budget", None) or cfg.max_evals) * _xai_full_per_eval(t, sp)
    if cfg.method == "gradient":
        per = 0.03 if sp.family == "unet" else 0.01 * max(1, len(t.patches or [1]))
        return 2 + cfg.grad_nsamples * per
    if sp.family == "resnet":
        per = 0.004 * (len(t.patches) if t.kind == "window_max_patch" else 1)
    else:
        per = 0.012 + (0.05 if sp.in_ch == 5 else 0.0)
    return 2 + cfg.max_evals * (per + 0.002)


def _dur_explain(rgb, auth, jobs, cfg):
    return int(min(120, max(30, 15 + sum(_xai_job_cost(t, cfg) for t in jobs))))


@gpu(duration=_dur_explain)
def _gpu_explain(rgb, auth, jobs, cfg):
    """SHAP açıklayıcıları bu fonksiyonun İÇİNDE kurulur (ZeroGPU); dönüş yalnız numpy + meta veri.
    Tam görüntü / döşemeli işler xai.fullimage'e, pencere işleri mevcut adaptörlere gider (sıra korunur)."""
    out = [None] * len(jobs)
    idx_f = [i for i, t in enumerate(jobs) if _xai_is_full(t)]
    idx_w = [i for i in range(len(jobs)) if i not in idx_f]
    if idx_f:
        for i, r in zip(idx_f, XAI.fullimage.run_full_jobs([jobs[i] for i in idx_f], rgb, auth, XAI_DEPS, XAI_SPECS, cfg)):
            out[i] = r
    if idx_w:
        for i, r in zip(idx_w, XAI.run_jobs([jobs[i] for i in idx_w], rgb, auth, XAI_DEPS, XAI_SPECS, cfg)):
            out[i] = r
    return out


def gpu_explain(rgb, auth, jobs, cfg):
    """İşleri ZeroGPU süre sınırına (≤120 s) sığacak parçalara bölerek çalıştırır."""
    out, chunk, est = [], [], 0.0
    for t in jobs:
        c = _xai_job_cost(t, cfg)
        if chunk and est + c > 100:
            out += _gpu_explain(rgb, auth, chunk, cfg); chunk, est = [], 0.0
        chunk.append(t); est += c
    if chunk:
        out += _gpu_explain(rgb, auth, chunk, cfg)
    return out


def xai_unsupported_md():
    if not XAI_OK:
        return f"⚠️ XAI paketi yüklenemedi: `{_XAI_ERR}`"
    lines = ["**SHAP üretilmeyen modeller:**"]
    for sp in XAI.UNSUPPORTED_MODELS:
        lines.append(f"- {sp.label} — *{sp.status}*: {sp.reason}")
    for n, sp in XAI_SPECS.items():
        if sp.status != "OK":
            lines.append(f"- {sp.label} — *{sp.status}*: {sp.reason}")
    if not SHAP_OK:
        lines.append(f"- ⚠️ `shap` paketi yok: {_SHAP_ERR}")
    return "\n".join(lines)


def _xai_source(src, orig, cur, mask, auth, sim, up_img, up_auth, up_gt):
    note = []
    if src == XAI_SOURCES[0]:
        rgb, a, g, desc = orig, auth, mask, "Girdi sekmesi (orijinal)"
    elif src == XAI_SOURCES[1]:
        rgb, a, g, desc = cur, auth, mask, "Ön işleme pipeline çıktısı"
        if cur is not None and orig is not None and cur is not orig:
            note.append("referans authentic eş işlenmemiş haliyle kullanılır")
    elif src == XAI_SOURCES[2]:
        if not sim:
            raise gr.Error("Henüz simülasyon çalıştırılmadı (sekme 0).")
        rgb, a, g, desc = sim.get("inp"), sim.get("auth"), sim.get("mask"), f"Son simülasyon girdisi ({sim.get('key')})"
    else:
        rgb, a, g, desc = up_img, up_auth, up_gt, "Yeni yükleme"
    if rgb is None:
        raise gr.Error("Görüntü yok: Girdi sekmesinden yükleyin, Simülasyon'dan aktarın ya da 'Yeni yükleme' ile ekleyin.")
    rgb = np.ascontiguousarray(to_rgb(rgb).astype(np.uint8))
    if a is not None:
        a = np.ascontiguousarray(to_rgb(a).astype(np.uint8))
        if a.shape != rgb.shape:
            note.append("authentic eş farklı boyutta → kullanılmadı"); a = None
        elif np.array_equal(a, rgb):
            note.append("girdi authentic eşin kendisi → referans olarak kullanılmadı"); a = None
    if g is not None:
        g = to_gray(g) if g.ndim == 3 else g
        if g.shape != rgb.shape[:2]:
            note.append("GT maskesi farklı boyutta → kullanılmadı"); g = None
    return rgb, a, g, desc + (" · " + "; ".join(note) if note else "")


def xai_predict(src, unets, resnets, orig, cur, mask, auth, sim, up_img, up_auth, up_gt):
    """① Tahmin: mevcut gpu_infer DEĞİŞTİRİLMEDEN çağrılır; skor, eşik, karar ve tespit listesi."""
    if not (XAI_OK and TORCH_OK):
        raise gr.Error(f"XAI kullanılamıyor: {_XAI_ERR or _TORCH_ERR}")
    unets, resnets = list(unets or []), list(resnets or [])
    if not unets and not resnets:
        raise gr.Error("En az bir model seçin.")
    rgb, a, g, desc = _xai_source(src, orig, cur, mask, auth, sim, up_img, up_auth, up_gt)
    t0 = time.time()
    res = gpu_infer(unets, resnets, [rgb], 112)[0]
    preds, rows = {}, []
    for n in resnets:
        hm, info = res[n]
        preds[n] = (hm, info)
        smax = max(s for s, _, _ in info["scores"])
        rows.append({"Model": friendly(n), "Aile": "ResNet-18", "Görüntü skoru": smax, "Eşik": 0.5,
                     "Karar": "SAHTE" if smax >= 0.5 else "GERÇEK", "Patch sayısı": len(info["scores"])})
    for n in unets:
        prob, info = res[n]
        preds[n] = (prob, info)
        sc = image_score(prob); it, cal = image_threshold(n)
        sp = XAI_SPECS.get(n)
        if n.startswith("YOLO/"):
            bx = info.get("boxes") or []
            rows.append({"Model": friendly(n), "Aile": "YOLO11", "Görüntü skoru": sc, "Eşik": it,
                         "Karar": "SAHTE" if sc >= it else "GERÇEK", "Patch sayısı": None,
                         "Tespit bölgesi (τ*)": (f"{info.get('n_boxes', len(bx))} kutu · en yüksek güven {bx[0][0]:.3f}" if bx
                                                 else "kutu yok") + ("" if cal else " · eşik kalibre değil")})
            continue
        tau = default_tau(n, (sp.train_gt if sp else "") or "bbox")
        nreg = len(XAI.targets.unet_regions(prob, tau, XAI_DEPS))
        rows.append({"Model": friendly(n), "Aile": "U-Net", "Görüntü skoru": sc, "Eşik": it,
                     "Karar": "SAHTE" if sc >= it else "GERÇEK", "Patch sayısı": None,
                     "Tespit bölgesi (τ*)": f"{nreg} (τ={tau:.2f})" + ("" if cal else " · eşik kalibre değil")})
    st = dict(rgb=rgb, auth=a, gt=g, preds=preds, desc=desc, models=resnets + unets)
    md = (f"**Kaynak:** {desc} · {rgb.shape[1]}×{rgb.shape[0]} px · GT {'var' if g is not None else 'yok'} · "
          f"authentic eş {'var' if a is not None else 'yok'} · tahmin {time.time() - t0:.1f} s. "
          "Şimdi **② SHAP analizini başlat** ile açıklayın (tahminler tekrar hesaplanmaz).")
    return pd.DataFrame(rows), md, st


def _xai_cfg(method, ref, evals):
    return XAI.ExplainConfig(method="gradient" if str(method).startswith("Gradient") else "partition",
                             reference=_XAI_REF_KEY.get(ref, "auto"), max_evals=int(evals),
                             batch_size=16, grad_nsamples=64)


def _xai_jobs(st, mode, k, cfg=None):
    shape = st["rgb"].shape
    preds = st["preds"]
    F = XAI.fullimage
    fam = lambda n: getattr(XAI_SPECS.get(n), "family", "")
    if mode in (XT_FULL_DEC, XT_FULL_DET, XT_FULL_GT, XT_TILES):
        if mode == XT_FULL_DET:
            jobs = F.full_detection_targets(preds, XAI_SPECS, shape, XAI_DEPS, k=int(k))
        elif mode == XT_FULL_GT:
            if st["gt"] is None:
                raise gr.Error("Bu görüntüde GT maskesi yok; başka bir hedef seçin.")
            jobs = F.full_gt_targets(st["models"], XAI_SPECS, shape, st["gt"], XAI_DEPS, preds)
        elif mode == XT_TILES:
            jobs = F.tile_targets(st["models"], XAI_SPECS, shape, XAI_DEPS)
        else:
            jobs = F.full_decision_targets(preds, XAI_SPECS, shape, XAI_DEPS)
        if ZERO_GPU and cfg is not None:  # ZeroGPU çağrı sınırı (≤120 s): bütçe gerekirse düşürülür (sonuç notunda yazılır)
            keep = []
            for t in jobs:
                c = _xai_job_cost(t, cfg)
                if c > 100:
                    b = int(cfg.max_evals * 95 / c)
                    if b < 60:  # bu çözünürlükte anlamlı bir tam görüntü açıklaması süre sınırına sığmıyor → sahte/kaba sonuç üretme
                        gr.Warning(f"{friendly(t.model)}: tam görüntü SHAP bu çözünürlükte ZeroGPU 120 s sınırına sığmıyor "
                                   f"(tahmini {c:.0f} s) — 'Döşemeli yerel mozaik' ya da 'Pencere' hedeflerini kullanın.")
                        continue
                    t.budget = b
                keep.append(t)
            jobs = keep
    else:
        preds = {n: p for n, p in preds.items() if fam(n) != "yolo"}
        models = [n for n in st["models"] if fam(n) != "yolo"]
        if not models:
            raise gr.Error("Pencere hedefleri piksel/patch modelleri içindir; YOLO11 için 'Tam görüntü' hedeflerini seçin.")
        if mode == XT_WIN_DET:
            jobs = XAI.detection_targets(preds, XAI_SPECS, shape, XAI_DEPS, k=int(k))
        elif mode == XT_WIN_GT:
            if st["gt"] is None:
                raise gr.Error("Bu görüntüde GT maskesi yok; başka bir hedef seçin.")
            jobs = XAI.gt_targets(models, XAI_SPECS, shape, st["gt"], XAI_DEPS)
        elif mode == XT_COMMON:
            jobs = XAI.common_window_targets(models, preds, XAI_SPECS, shape, st["gt"], XAI_DEPS)
        else:
            jobs = XAI.decision_targets(preds, XAI_SPECS, shape, XAI_DEPS)
    if not jobs:
        raise gr.Error("Açıklanacak hedef bulunamadı (ör. eşik üstü tespit bölgesi yok). Başka bir hedef türü deneyin.")
    return jobs


def _xai_window_views(st, t):
    img = XAI.prepare_for(t.prep, st["rgb"], XAI_DEPS)[0][t.y:t.y + t.h, t.x:t.x + t.w]
    gt_w = XAI.window_mask(st["gt"], t, XAI_DEPS) if st["gt"] is not None else None
    gm = green_marker_mask(st["rgb"])
    if st["auth"] is not None:
        gm &= ~green_marker_mask(st["auth"])
    mk_w = XAI.window_mask(gm, t, XAI_DEPS)
    return img, gt_w, mk_w


def _xai_full_views(st, t):
    """Tam görüntü sonuçları için ekran çözünürlüğünde (en-boy korunur, ≤2400 px) orijinal + GT + işaretçi + hedef alanı;
    tam çözünürlüklü maskeler de döner (kütle ve dağılım hesapları orijinal çözünürlükte yapılır)."""
    V = XAI.viewer
    rgb = st["rgb"]
    H0, W0 = rgb.shape[:2]
    img = V.display(rgb)
    hw = img.shape[:2]
    g = st["gt"]
    gt = None if g is None else (np.asarray(g) > (127 if np.asarray(g).dtype == np.uint8 else 0))
    gm = green_marker_mask(rgb)
    if st["auth"] is not None:
        gm &= ~green_marker_mask(st["auth"])
    reg = XAI.fullimage.target_region_orig(t, H0, W0, XAI_DEPS)
    return img, gt, gm, reg, V.display_mask(gt, hw), V.display_mask(gm, hw), V.display_mask(reg, hw)


def _xai_full_badge(st, r, t):
    """Tam görüntü iddiası yalnız doğrulanırsa: harita boyutu = orijinal, tek hedef/tek referans, E[f]+ΣSHAP=f(x)."""
    H0, W0 = st["rgb"].shape[:2]
    m = r.shap_map()
    if t.kind == XAI.fullimage.TILE_KIND:
        tg = r.target
        return (f"**DÖŞEMELİ YEREL MOZAİK — tam görüntü SHAP DEĞİLDİR.** {len(tg.get('tiles', []))} pencere · kapsama "
                f"{'tam' if tg.get('covered') else 'EKSİK'} · pencere tabanları farklı (fark {tg.get('base_spread', 0):.3f}); "
                "her pencere yalnız kendi skorunu açıklar.  \n")
    ok = bool(r.target.get("full")) and m.shape == (H0, W0) and np.isfinite(r.add_err) and r.add_err < 1e-3
    if ok:
        return (f"**TAM GÖRÜNTÜ SHAP (doğrulandı):** harita {W0}×{H0} px = orijinal görüntü, kırpma yok · tek hedef, tek referans · "
                f"E[f] + ΣSHAP = f(x) (hata {r.add_err:.1e}) · ızgara {r.target.get('grid', ['?', '?'])[1]}×"
                f"{r.target.get('grid', ['?', '?'])[0]} hücre ({r.target.get('cell_px', '?')} px).  \n")
    return (f"⚠️ **Tam görüntü doğrulanamadı** (harita {m.shape[1]}×{m.shape[0]}, toplamsallık hatası {r.add_err:.1e}) — "
            "sonuç tam görüntü SHAP açıklaması olarak SUNULMUYOR.  \n")


def _xai_app_score(st, model):
    """Uygulamanın tahmin tablosundaki görüntü skoru (U-Net: tam görüntü top-%1; ResNet: maks patch)."""
    pr = st["preds"].get(model)
    if pr is None:
        return None
    m, info = pr
    if "scores" in info:
        return max(s_ for s_, _, _ in info["scores"])
    return image_score(m)


def _xai_label(i, r):
    return f"{i + 1} · {r.label} · {r.target.get('label', '')}"[:150]


def _xai_render(st, results, jobs):
    """Tüm sonuçlar için paneller, bölge kütleleri ve dışa aktarım dosyaları."""
    panels, masses, gal, sums = {}, {}, [], {}
    for i, (r, t) in enumerate(zip(results, jobs)):
        lab = _xai_label(i, r)
        if not r.ok:
            sums[lab] = f"**{r.label}** — {r.status}: {r.message}"
            continue
        badge = ""
        if _xai_is_full(t):
            import copy as _copy
            m = r.shap_map()
            img, gt, gm, reg, gt_d, gm_d, reg_d = _xai_full_views(st, t)
            rd = _copy.copy(r); rd.values = XAI.viewer.display_map(m)[..., None]
            pn = XAI.viz.panels(rd, img, gt_d, gm_d, reg_d)
            pn["Özet şekil"] = XAI.viz.summary_figure(rd, img, gt_d, gm_d)
            pn["Katkı dağılımı"] = XAI.viewer.distribution_figure(
                m, gt, r.fx, r.base, f"{r.label} · açıklanan hedef: SAHTE yönü (+) · {r.target.get('label', '')}"[:160])
            masses[f"{i + 1} · {r.label}"] = XAI.viz.region_masses(m, gt, gm)
            badge = _xai_full_badge(st, r, t)
        else:
            img, gt_w, mk_w = _xai_window_views(st, t)
            pn = XAI.viz.panels(r, img, gt_w, mk_w, t.region if t.kind in ("region_mean", "gt_bbox_mean") else None)
            pn["Özet şekil"] = XAI.viz.summary_figure(r, img, gt_w, mk_w)
            pn["Katkı dağılımı"] = XAI.viewer.distribution_figure(r.shap_map(), gt_w, r.fx, r.base,
                                                                  f"{r.label} · pencere açıklaması · {r.target.get('label', '')}"[:160])
            masses[f"{i + 1} · {r.label}"] = XAI.viz.region_masses(r.shap_map(), gt_w, mk_w)
        panels[lab] = pn
        for pname, im in pn.items():
            if pname != "Özet şekil":
                gal.append((im, f"{i + 1} · {r.label} — {pname}"))
        g = "".join(f" · {k_} {v:+.4f}" for k_, v in r.groups.items()) if r.groups else ""
        app_sc = _xai_app_score(st, r.model)
        cmp_note = ""
        if app_sc is not None and t.kind == "window_image_score":
            cmp_note = (f"  \nUygulamanın tam görüntü skoru (448 px adımlı pencerelerin örtüşme ortalaması): {app_sc:.4f}"
                        + (" — açıklanan pencere tek başına çok daha yüksek skor veriyor; örtüşme ortalaması sinyali seyreltiyor."
                           if r.fx - app_sc > 0.25 else ""))
        sums[lab] = (badge + f"**{r.label}** · hedef: {r.target.get('label', '')}  \n"
                     f"f(x) = **{r.fx:.4f}** · E[f] (referans: {r.reference}) = {r.base:.4f} · ΣSHAP = {r.sum_shap:+.4f} · "
                     f"toplamsallık hatası {r.add_err:.2e} · {r.n_evals} değerlendirme · {r.seconds:.1f} s"
                     + (f" · tepe {r.peak_mem_mb:.0f} MB" if r.peak_mem_mb else "") + g
                     + cmp_note
                     + ("  \n_" + "; ".join(r.notes) + "_" if r.notes else "")
                     + ("  \n(önbellekten)" if r.cached else ""))
    mass_img = XAI.viz.mass_chart(masses)
    loc = XAI.viz.location_image(st["rgb"], [r for r in results])
    return panels, masses, gal, sums, mass_img, loc


def _xai_export(st, results, panels, mass_img, loc, cfg, cmp=None, cmp_img=None, tag="shap"):
    outdir = Path(tempfile.mkdtemp(prefix=f"ai515_{tag}_"))
    meta = dict(title=st["desc"], image=f"{st['rgb'].shape[1]}×{st['rgb'].shape[0]}", source=st["desc"],
                config=dict(method=cfg.method, reference=cfg.reference, max_evals=cfg.max_evals, grad_nsamples=cfg.grad_nsamples),
                device=str(DEVICE), zero_gpu=ZERO_GPU)
    extra = {"konum.png": loc, "bolge_kutlesi.png": mass_img}
    if cmp_img is not None:
        extra["karsilastirma.png"] = cmp_img
    return XAI.export_results(results, outdir, meta, panels, cmp, extra)


def _xai_hires(st, results, jobs, outdir):
    """Tam görüntü sonuçları: natif çözünürlükte ısı haritası + bindirme PNG, float32 SHAP NPY, 300 dpi PNG + PDF yayın
    şekli (renk çubuğu, piksel eksenleri, GT / hedef konturu)."""
    V, F = XAI.viewer, XAI.fullimage
    files = []
    rgb = st["rgb"]
    H0, W0 = rgb.shape[:2]
    for i, (r, t) in enumerate(zip(results, jobs)):
        if not r.ok or not _xai_is_full(t):
            continue
        try:
            m = r.shap_map()
            _, gt, _, reg, _, _, _ = _xai_full_views(st, t)
            v = XAI.viz.sym_vmax(m)
            heat = XAI.viz.diverging(m, v)
            ov = XAI.viz.blend(rgb, heat, 0.75, np.clip(np.abs(m) / v, 0, 1))
            stem = f"r{i + 1:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", r.model).strip("_")
            files += V.export_native(outdir, stem, {"shap_isi_natif": heat, "bindirme_natif": ov}, {"shap_float32": m})
            kind = ("döşemeli yerel mozaik (tam görüntü SHAP değil)" if t.kind == F.TILE_KIND else "tam görüntü SHAP")
            files += V.publication_figure(rgb, m, f"{r.label} · {kind} · {W0}×{H0} px · f(x)={r.fx:.3f} · E[f]={r.base:.3f} · "
                                                  f"ΣSHAP={r.sum_shap:+.3f}", outdir, stem, gt=gt, region=reg)
        except Exception as e:
            print("[XAI] yüksek çözünürlüklü dışa aktarım:", r.model, e)
    return files


_XAI_PANEL_NAMES = ["Orijinal (açıklanan pencere)", "SHAP ısı haritası (kırmızı: SAHTE yönü · mavi: GERÇEK yönü)", "Bindirme",
                    "Pozitif katkı (SAHTE'yi artıran)", "Negatif katkı (GERÇEK'e iten)", "Sınıf panelleri: SAHTE | GERÇEK (= −SAHTE)"]


def _xai_show_outputs(view, choice):
    """[özet, 6 panel, kaydırıcı (orijinal ↔ bindirme), katkı dağılımı]"""
    if not view or choice not in view.get("panels", {}):
        md = view.get("sums", {}).get(choice, " ") if view else " "
        return [md] + [None] * (len(_XAI_PANEL_NAMES) + 2)
    pn = view["panels"][choice]
    a, b = pn.get(_XAI_PANEL_NAMES[0]), pn.get(_XAI_PANEL_NAMES[2])
    sl = (a, b) if a is not None and b is not None and a.shape == b.shape else None
    return [view["sums"].get(choice, " ")] + [pn.get(n) for n in _XAI_PANEL_NAMES] + [sl, pn.get("Katkı dağılımı")]


def xai_run(st, mode, k, method, ref, evals):
    """② SHAP: tahminden bağımsız; önbellekte olan (model, hedef, yöntem, referans, bütçe) yeniden hesaplanmaz."""
    if not st:
        raise gr.Error("Önce ① Tahmin yap.")
    if not SHAP_OK:
        raise gr.Error(f"shap paketi yüklü değil: {_SHAP_ERR}")
    cfg = _xai_cfg(method, ref, evals)
    jobs = _xai_jobs(st, mode, k, cfg)
    t0 = time.time()
    results = XAI_MGR.explain(st["rgb"], st["auth"], jobs, cfg)
    panels, masses, gal, sums, mass_img, loc = _xai_render(st, results, jobs)
    files = _xai_export(st, results, panels, mass_img, loc, cfg)
    if files:
        files += _xai_hires(st, results, jobs, Path(files[0]).parent)
    df = pd.DataFrame(XAI.service.result_table_rows(results))
    if len(df):
        df.insert(5, "Uygulama skoru", [_xai_app_score(st, r.model) for r in results])
    choices = list(sums)
    view = dict(panels=panels, sums=sums)
    n_ok = sum(r.ok for r in results)
    status = (f"**SHAP tamam:** {n_ok}/{len(results)} açıklama · {time.time() - t0:.1f} s · önbellek isabeti "
              f"{sum(r.cached for r in results)} · hedef: {mode} · yöntem: {cfg.method} · bütçe: "
              f"{cfg.max_evals if cfg.method == 'partition' else cfg.grad_nsamples}")
    first = choices[0] if choices else None
    return [status, df, loc, gr.update(choices=choices, value=first), view, mass_img, gal, files] + _xai_show_outputs(view, first)


def xai_compare(st, method, ref, evals):
    """③ Karşılaştırma: tüm modeller AYNI pencereyi açıklar (önbellek kullanılır); IoU/Dice, Spearman, ortak/özgü bölge."""
    if not st:
        raise gr.Error("Önce ① Tahmin yap.")
    if not SHAP_OK:
        raise gr.Error(f"shap paketi yüklü değil: {_SHAP_ERR}")
    cfg = _xai_cfg(method, ref, evals)
    jobs = _xai_jobs(st, XT_COMMON, 1, cfg)
    results = XAI_MGR.explain(st["rgb"], st["auth"], jobs, cfg)
    cmp = XAI.compare.compare_results(results, st["rgb"])
    fig = XAI.compare.comparison_figure(cmp)
    rows = []
    for i, r in enumerate(results):
        rows.append({"#": i + 1, "Model": r.label, "Hedef türü": r.target.get("kind"), "f(x)": r.fx, "E[f]": r.base,
                     "ΣSHAP": r.sum_shap, "max|SHAP| (normalizasyon)": float(np.abs(r.shap_map()).max()) if r.ok else None,
                     "Durum": r.status + (" (önbellek)" if r.cached else "")})
    md = "**Ölçek / hedef uyumu:** " + cmp["note"]
    if "iou" in cmp:
        n = len(cmp["names"])
        off = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if off:
            md += (f"  \nOrtalama top-%5 IoU {np.nanmean([cmp['iou'][i, j] for i, j in off]):.3f} · ortalama Spearman ρ "
                   f"{np.nanmean([cmp['spearman'][i, j] for i, j in off]):.3f} · tüm modellerde ortak bölge payı "
                   f"%{100 * cmp['common_frac']:.1f}")
    panels, masses, gal, sums, mass_img, loc = _xai_render(st, results, jobs)
    files = _xai_export(st, results, panels, mass_img, loc, cfg, cmp, fig, tag="shap_cmp")
    return md, fig, pd.DataFrame(rows), files

# ================================================================== ARAYÜZ
CSS = """
.small-md p {font-size: 0.9em}
#verdict {border-left: 4px solid #4C72B0; padding-left: 10px}
footer {display:none !important}
#aic-tip {position:fixed; z-index:10000; max-width:340px; padding:8px 11px; border-radius:6px; font-size:13px; line-height:1.45; background:#1f2937; color:#f9fafb; box-shadow:0 4px 14px rgba(0,0,0,.25); pointer-events:none; opacity:0; transition:opacity .12s; left:-9999px; top:0}
#aic-tip.show {opacity:1}
#aic-tip b {color:#fde68a}
#tip-f_split [data-testid="block-info"]::after, #tip-f_source [data-testid="block-info"]::after, #tip-f_tool [data-testid="block-info"]::after, #tip-f_area [data-testid="block-info"]::after, #tip-f_marker [data-testid="block-info"]::after, #tip-f_mp [data-testid="block-info"]::after, #tip-f_pair [data-testid="block-info"]::after, #tip-b_filter [data-testid="block-info"]::after, #tip-b_rand [data-testid="block-info"]::after, #tip-dd_key [data-testid="block-info"]::after, #tip-rd_input [data-testid="block-info"]::after, #tip-dd_deg [data-testid="block-info"]::after, #tip-ck_pipe [data-testid="block-info"]::after, #tip-cg_unet [data-testid="block-info"]::after, #tip-cg_res [data-testid="block-info"]::after, #tip-sl_stride_sim [data-testid="block-info"]::after, #tip-b_sim [data-testid="block-info"]::after, #tip-b_to_tabs [data-testid="block-info"]::after, #tip-t_split [data-testid="block-info"]::after, #tip-t_source [data-testid="block-info"]::after, #tip-t_tool [data-testid="block-info"]::after, #tip-t_area [data-testid="block-info"]::after, #tip-t_marker [data-testid="block-info"]::after, #tip-t_mp [data-testid="block-info"]::after, #tip-t_n [data-testid="block-info"]::after, #tip-t_seed [data-testid="block-info"]::after, #tip-t_mode [data-testid="block-info"]::after, #tip-t_deg [data-testid="block-info"]::after, #tip-t_unet [data-testid="block-info"]::after, #tip-t_res [data-testid="block-info"]::after, #tip-b_batch [data-testid="block-info"]::after, #tip-in_img [data-testid="block-info"]::after, #tip-in_mask [data-testid="block-info"]::after, #tip-in_auth [data-testid="block-info"]::after, #tip-btn_load [data-testid="block-info"]::after, #tip-dd_sample [data-testid="block-info"]::after, #tip-btn_sample [data-testid="block-info"]::after, #tip-tx_draft [data-testid="block-info"]::after, #tip-dd_draft [data-testid="block-info"]::after, #tip-dd_cat [data-testid="block-info"]::after, #tip-dd_op [data-testid="block-info"]::after, #tip-sl1 [data-testid="block-info"]::after, #tip-sl2 [data-testid="block-info"]::after, #tip-sl3 [data-testid="block-info"]::after, #tip-dd_mode [data-testid="block-info"]::after, #tip-btn_prev [data-testid="block-info"]::after, #tip-btn_apply [data-testid="block-info"]::after, #tip-btn_undo [data-testid="block-info"]::after, #tip-btn_reset [data-testid="block-info"]::after, #tip-dd_preset [data-testid="block-info"]::after, #tip-btn_preset [data-testid="block-info"]::after, #tip-dd_unet [data-testid="block-info"]::after, #tip-rd_src [data-testid="block-info"]::after, #tip-rd_gt [data-testid="block-info"]::after, #tip-sl_thr [data-testid="block-info"]::after, #tip-ck_otsu [data-testid="block-info"]::after, #tip-sl_morph [data-testid="block-info"]::after, #tip-btn_unet [data-testid="block-info"]::after, #tip-btn_cmp [data-testid="block-info"]::after, #tip-dd_rn [data-testid="block-info"]::after, #tip-rd_src2 [data-testid="block-info"]::after, #tip-sl_stride [data-testid="block-info"]::after, #tip-btn_rn [data-testid="block-info"]::after, #tip-btn_green [data-testid="block-info"]::after, #tip-x_src [data-testid="block-info"]::after, #tip-x_up_img [data-testid="block-info"]::after, #tip-x_up_auth [data-testid="block-info"]::after, #tip-x_up_gt [data-testid="block-info"]::after, #tip-x_unet [data-testid="block-info"]::after, #tip-x_res [data-testid="block-info"]::after, #tip-b_xpred [data-testid="block-info"]::after, #tip-x_target [data-testid="block-info"]::after, #tip-x_k [data-testid="block-info"]::after, #tip-x_method [data-testid="block-info"]::after, #tip-x_ref [data-testid="block-info"]::after, #tip-x_evals [data-testid="block-info"]::after, #tip-b_xshap [data-testid="block-info"]::after, #tip-b_xcmp [data-testid="block-info"]::after, #tip-x_show [data-testid="block-info"]::after {content:" ⓘ"; opacity:.55; font-size:.9em}
"""


# ---------------------------------------------------------------- sol panel açıklamaları (fareyle üzerine gelince)
TIPS = {
    "tip-f_split": 'Görüntülerin veri setinin hangi bölümünden seçileceği.<br><b>test</b>: modellerin eğitimde hiç görmediği görüntüler (adil değerlendirme).<br><b>eğitim</b>: modellerin öğrendiği görüntüler; sonuçlar iyimser çıkar.<br><b>hepsi</b>: iki bölüm birlikte.',
    "tip-f_source": 'Sahte belgenin türetildiği orijinal belge koleksiyonu (ör. <b>cord</b> fişleri). Belge türüne ve düzenine göre filtrelemeye yarar.',
    "tip-f_tool": 'Belgedeki alanı değiştirmek için kullanılan üretken yapay zekâ (inpainting) modeli, ör. <b>Gemini</b>, <b>Ideogram</b>. Her üretici farklı izler bırakır; modellerin bir üreticiden diğerine genelleşmesini incelemek için kullanılır.',
    "tip-f_area": 'Değiştirilen bölgenin (resmî GT kutusu) görüntü alanına oranı.<br><b>&lt;1%</b>: çok küçük hedef, tespiti en zor rejim.<br><b>1-5%</b>, <b>5-20%</b>, <b>&gt;20%</b>: hedef büyüdükçe lokalizasyon kolaylaşır.',
    "tip-f_marker": "Bazı sahte görüntülerde (çoğunlukla Gemini) düzenlenen alanın çevresinde ~3 px'lik yeşil çerçeve kalıntısı kalmıştır. Bir model sahteciliği değil bu çerçeveyi öğrenebilir (kısayol).<br><b>var</b> / <b>yok</b> ile bu örnekleri ayırabilirsiniz.",
    "tip-f_mp": 'Görüntü boyutu üst sınırı (genişlik × yükseklik / 10⁶). Modeller natif çözünürlükte çalıştığı için büyük görüntülerde çıkarım uzar; hızlı denemeler için düşük tutun.',
    "tip-f_pair": "Yalnızca değiştirilmemiş orijinali (<i>authentic eşi</i>) aynı boyutta bulunan örnekleri listeler. Piksel-değişim GT'si, authentic negatif kontrol ve yeşil işaretçi senaryoları bu eşe ihtiyaç duyar.",
    "tip-b_filter": 'Yukarıdaki filtreleri uygular ve <b>Örnek</b> listesini günceller.',
    "tip-b_rand": 'Filtreye uyan örnekler arasından rastgele birini seçer.',
    "tip-dd_key": 'Çalıştırılacak görüntü. Biçim: <b>bölüm:kimlik</b> · kaynak · üretici · alan oranı · MP (+ <i>eşli</i>, <i>işaretçi</i> etiketleri). Listeden seçebilir ya da kimliği yazarak arayabilirsiniz.',
    "tip-rd_input": 'Modele hangi görüntünün verileceği:<br><b>Sahte belge</b>: görüntü olduğu gibi (beklenen karar SAHTE).<br><b>Authentic eş</b>: değiştirilmemiş orijinal (beklenen GERÇEK); yanlış alarm kontrolü.<br><b>Sahte, yeşil işaretçi onarılmış</b>: işaretçi pikselleri orijinalden geri kopyalanır, içerik sahte kalır (beklenen SAHTE); model işaretçiye mi dayanıyor?<br><b>Authentic + yalnız yeşil işaretçi</b>: orijinale yalnızca işaretçi eklenir, hiçbir karakter değişmez (beklenen GERÇEK); SAHTE derse model kısayol öğrenmiştir.<br>Son üç senaryo authentic eşi olan örnek gerektirir.',
    "tip-dd_deg": 'Modelden önce girdiye uygulanan bozulma; gerçek hayattaki işlemlere dayanıklılığı ölçer (modeller ham PNG ile eğitildi).<br><b>JPEG Q75 / Q50</b>: kayıplı sıkıştırma<br><b>0.75× / 0.5×</b>: küçültüp eski boyuta geri büyütme<br><b>Gauss bulanıklık σ=1</b>, <b>Gauss gürültüsü σ=5</b>',
    "tip-ck_pipe": 'İşaretliyse <b>Ön işleme</b> sekmesinde kurduğunuz işlem zinciri modelden önce girdiye uygulanır; ön işlemenin model başarısına etkisini görmek içindir.',
    "tip-cg_unet": 'Piksel düzeyinde sahte bölge olasılık haritası üreten segmentasyon modelleri. Adlar hangi GT ile eğitildiğini gösterir (bbox, piksel-değişim, glyph). Tam görüntüde 512 px pencere / 448 px adımla çalışır. Birden fazlasını seçip karşılaştırabilirsiniz.',
    "tip-cg_res": 'Görüntüyü 224×224 px parçalara (patch) bölüp her parçayı sahte/gerçek olarak sınıflandıran model. Çıktısı kaba bir ısı haritası ve en yüksek patch skoruna dayanan görüntü kararıdır.',
    "tip-sl_stride_sim": "224 px'lik ResNet penceresinin her adımda kaç piksel kaydırılacağı. Küçük adım daha çok örtüşme ve daha ince ısı haritası verir ama yavaştır (adım yarıya inince patch sayısı ~4 katına çıkar). 224 = örtüşmesiz.",
    "tip-b_sim": 'Seçilen örnek, senaryo ve modellerle hattı adım adım çalıştırır: GT analizi → yeşil işaretçi taraması → girdi senaryosu ve bozulma → modeller → iki GT (bbox ve piksel-değişim) ile metrikler → görüntü-düzeyi karar.',
    "tip-b_to_tabs": 'Bu örneği (sahte görüntü, GT maskesi, authentic eş) Girdi, Ön işleme, U-Net ve ResNet-18 sekmelerine yükler.',
    "tip-t_split": 'Görüntülerin veri setinin hangi bölümünden seçileceği.<br><b>test</b>: modellerin eğitimde hiç görmediği görüntüler (adil değerlendirme).<br><b>eğitim</b>: modellerin öğrendiği görüntüler; sonuçlar iyimser çıkar.<br><b>hepsi</b>: iki bölüm birlikte.',
    "tip-t_source": 'Sahte belgenin türetildiği orijinal belge koleksiyonu (ör. <b>cord</b> fişleri). Belge türüne ve düzenine göre filtrelemeye yarar.',
    "tip-t_tool": 'Belgedeki alanı değiştirmek için kullanılan üretken yapay zekâ (inpainting) modeli, ör. <b>Gemini</b>, <b>Ideogram</b>. Her üretici farklı izler bırakır; modellerin bir üreticiden diğerine genelleşmesini incelemek için kullanılır.',
    "tip-t_area": 'Değiştirilen bölgenin (resmî GT kutusu) görüntü alanına oranı.<br><b>&lt;1%</b>: çok küçük hedef, tespiti en zor rejim.<br><b>1-5%</b>, <b>5-20%</b>, <b>&gt;20%</b>: hedef büyüdükçe lokalizasyon kolaylaşır.',
    "tip-t_marker": "Bazı sahte görüntülerde (çoğunlukla Gemini) düzenlenen alanın çevresinde ~3 px'lik yeşil çerçeve kalıntısı kalmıştır. Bir model sahteciliği değil bu çerçeveyi öğrenebilir (kısayol).<br><b>var</b> / <b>yok</b> ile bu örnekleri ayırabilirsiniz.",
    "tip-t_mp": 'Görüntü boyutu üst sınırı (genişlik × yükseklik / 10⁶). Modeller natif çözünürlükte çalıştığı için büyük görüntülerde çıkarım uzar; hızlı denemeler için düşük tutun.',
    "tip-t_n": 'Filtreye uyan örnekler arasından rastgele kaç tanesinin çalıştırılacağı. Süre N ile doğrusal artar.',
    "tip-t_seed": 'Rastgele örnek seçiminin tohumu. Aynı tohum ve aynı filtreler aynı örnekleri verir (tekrar üretilebilirlik).',
    "tip-t_mode": '<b>Yalnız sahte görüntüler</b>: yalnızca lokalizasyon metrikleri (F1, IoU, AP).<br><b>Sahte + authentic eşi</b>: her sahte görüntünün orijinali de çalıştırılır; böylece görüntü-düzeyi tespit (TPR/FPR/AUROC) hesaplanır. Yalnız authentic eşi olan örnekler seçilir.',
    "tip-t_deg": 'Modelden önce girdiye uygulanan bozulma; gerçek hayattaki işlemlere dayanıklılığı ölçer (modeller ham PNG ile eğitildi).<br><b>JPEG Q75 / Q50</b>: kayıplı sıkıştırma<br><b>0.75× / 0.5×</b>: küçültüp eski boyuta geri büyütme<br><b>Gauss bulanıklık σ=1</b>, <b>Gauss gürültüsü σ=5</b>',
    "tip-t_unet": 'Piksel düzeyinde sahte bölge olasılık haritası üreten segmentasyon modelleri. Adlar hangi GT ile eğitildiğini gösterir (bbox, piksel-değişim, glyph). Tam görüntüde 512 px pencere / 448 px adımla çalışır. Birden fazlasını seçip karşılaştırabilirsiniz.',
    "tip-t_res": 'Görüntüyü 224×224 px parçalara (patch) bölüp her parçayı sahte/gerçek olarak sınıflandıran model. Çıktısı kaba bir ısı haritası ve en yüksek patch skoruna dayanan görüntü kararıdır.',
    "tip-b_batch": 'Filtreye uyan N rastgele örnekte simülasyon hattını koşturur; model başına lokalizasyon (bbox ve piksel-değişim GT), görüntü-düzeyi tespit (TPR/FPR/AUROC) ve boru hattının kayıtlı sonuçlarıyla yeniden-üretim farkı raporlanır.',
    "tip-in_img": 'İncelenecek belge görüntüsü (PNG/JPG). Kayıpsız PNG önerilir; JPEG sıkıştırması sahtecilik izlerini zayıflatır.',
    "tip-in_mask": 'Değiştirilen bölgenin ikili maskesi (beyaz 255 = sahte, siyah 0 = gerçek), sahte görüntüyle aynı boyutta. Verilirse model sonuçları bu GT ile ölçülür; verilmezse yalnızca tahmin gösterilir.',
    "tip-in_auth": "Aynı belgenin değiştirilmemiş hali, aynı boyutta. Verilirse iki görüntünün farkından <b>piksel-değişim GT'si</b> türetilir; yeşil işaretçi analizi de bu eşle yapılır.",
    "tip-btn_load": "Yüklediğiniz görüntüleri (ve varsa maske ile authentic eşi) diğer sekmelerin girdisi yapar. Ön işleme pipeline'ı sıfırlanır.",
    "tip-dd_sample": 'Veri setinden doğrudan örnek seçimi (bölüm:kimlik). Filtrelerle daha hızlı seçim için <b>Simülasyon</b> sekmesini kullanıp örneği oradan aktarabilirsiniz.',
    "tip-btn_sample": "Seçilen örneği (sahte görüntü, GT maskesi ve varsa authentic eşi) yükler. Ön işleme pipeline'ı sıfırlanır.",
    "tip-tx_draft": 'Taslağa vereceğiniz ad (boş bırakılırsa tarih kullanılır).',
    "tip-btn_dsave": "O an yüklü olan girdiyi <b>taslak</b> olarak diske kaydeder: sahte görüntü, GT maskesi, authentic eş ve <b>Ön işleme</b> sekmesindeki pipeline adımları. Henüz <b>Yüklenenleri kullan</b>'a basılmadıysa yukarıdaki yükleme alanları kaydedilir.",
    "tip-dd_draft": 'Daha önce kaydedilen taslaklar (en yeni üstte): ad · tarih · boyut · içerik.',
    "tip-btn_dload": "Seçili taslağı tüm sekmelere girdi olarak yükler; kayıtlı pipeline adımları yeniden uygulanır.",
    "tip-btn_ddel": 'Seçili taslağı kalıcı olarak siler.',
    "tip-dd_cat": 'Ders haftalarına göre gruplanmış ön işleme kategorileri (renk, histogram, eşikleme, gürültü, morfoloji, frekans domeni…).',
    "tip-dd_op": 'Seçili kategorideki işlem. Seçince parametreleri ve kısa açıklaması altta görünür.',
    "tip-sl1": 'Seçili işlemin parametresi; anlamı etikette ve işlem açıklamasında yazar. Değiştirip <b>Önizle</b> ile etkisini görün.',
    "tip-sl2": 'Seçili işlemin parametresi; anlamı etikette ve işlem açıklamasında yazar. Değiştirip <b>Önizle</b> ile etkisini görün.',
    "tip-sl3": 'Seçili işlemin parametresi; anlamı etikette ve işlem açıklamasında yazar. Değiştirip <b>Önizle</b> ile etkisini görün.',
    "tip-dd_mode": 'Seçili işlemin çalışma modu (ör. çekirdek şekli, alçak/yüksek geçiren).',
    "tip-btn_prev": "İşlemi mevcut pipeline çıktısına uygulayıp sonucu <b>Sonra</b> panelinde gösterir; pipeline'a eklemez.",
    "tip-btn_apply": "İşlemi pipeline'a kalıcı adım olarak ekler; sonraki işlemler bu çıktının üzerine uygulanır. Pipeline çıktısı U-Net, ResNet-18 ve Simülasyon sekmelerinde model girdisi olarak seçilebilir.",
    "tip-btn_undo": "Pipeline'daki son adımı kaldırır.",
    "tip-btn_reset": 'Tüm adımları siler ve orijinal görüntüye döner.',
    "tip-dd_preset": 'Hazır işlem zincirleri (ör. belge ikilileştirme, frekans analizi, kenar haritası).',
    "tip-btn_preset": "Seçili preseti <b>orijinal</b> görüntüye uygular; mevcut pipeline'ın yerine geçer.",
    "tip-dd_unet": "Kullanılacak U-Net ağırlığı. Ad; eğitimde kullanılan GT'yi (bbox / piksel-değişim / glyph), giriş kanallarını (RGB ya da RGB + frekans) ve tohumu (s42/43/44) gösterir.",
    "tip-rd_src": 'Modele <b>orijinal görüntünün</b> mü yoksa <b>Ön işleme</b> sekmesinde kurduğunuz <b>pipeline çıktısının</b> mı verileceği.',
    "tip-rd_gt": 'Tahminin hangi referansla ölçüleceği.<br><b>resmî bbox</b>: veri setinin dikdörtgen etiketi.<br><b>piksel-değişim</b>: sahte ile authentic eş arasındaki fark eşiklenip (eps 8) morfolojiyle temizlenerek türetilen, yalnız gerçekten değişen pikselleri içeren GT. Authentic eşi gerekir.',
    "tip-sl_thr": 'Olasılık haritasını ikili maskeye çeviren eşik. Model ya da GT türü değişince, o model için doğrulama kümesinde seçilmiş en iyi eşik (τ*) otomatik gelir; elle değiştirebilirsiniz.',
    "tip-ck_otsu": 'Sabit eşik yerine, olasılık haritasının histogramından Otsu yöntemiyle otomatik eşik hesaplanır.',
    "tip-sl_morph": 'İkili maskeye önce açma (küçük gürültüleri siler), sonra kapama (boşlukları doldurur) uygulanır; değer çekirdek boyutudur (px). 0 = kapalı.',
    "tip-btn_unet": 'Seçili modeli çalıştırır: olasılık haritası, bindirme, ikili maske ve ölçütler (F1, IoU, AP…).',
    "tip-btn_cmp": "Aynı modeli hem orijinal görüntüde hem pipeline çıktısında çalıştırıp ölçütleri yan yana koyar; ön işlemenin etkisini gösterir. Pipeline'da en az bir adım olmalı.",
    "tip-dd_rn": "ResNet-18 ağırlığı. <b>ft</b>: ImageNet'ten tam ince ayar · <b>probe</b>: yalnız son katman eğitildi · <b>scratch</b>: sıfırdan eğitim. <b>h4</b> modelleri Faz 2 H4 deneylerine aittir.",
    "tip-rd_src2": 'Modele <b>orijinal görüntünün</b> mü yoksa <b>Ön işleme</b> sekmesinde kurduğunuz <b>pipeline çıktısının</b> mı verileceği.',
    "tip-sl_stride": "224 px'lik pencerenin her adımda kaç piksel kaydırılacağı. Küçük adım daha ince ısı haritası verir ama yavaştır (adım yarıya inince patch sayısı ~4 katına çıkar). 224 = örtüşmesiz.",
    "tip-btn_rn": "Görüntüyü 224 px patch'lerle tarar; patch skoru ısı haritası ve ölçütleri üretir. Görüntü kararı: en yüksek patch skoru ≥ 0.5 ise SAHTE.",
    "tip-btn_green": 'Mevcut görüntüde yeşil işaretçi piksellerini (G − max(R,B) > 50 ve G > 110) arar ve magenta ile gösterir. Authentic eşi varsa orijinalde de bulunan yeşiller çıkarılır.',
    "tip-x_src": "Açıklanacak görüntünün kaynağı.<br><b>Girdi sekmesi</b>: yüklenen ya da aktarılan orijinal görüntü.<br><b>Pipeline çıktısı</b>: Ön işleme sekmesindeki zincirin çıktısı.<br><b>Son simülasyon girdisi</b>: sekme 0'da modele verilen görüntü (senaryo + bozulma dahil).<br><b>Yeni yükleme</b>: buradan yeni görüntü (+ isteğe bağlı authentic eş ve GT).",
    "tip-x_up_img": 'Açıklanacak belge görüntüsü (PNG önerilir).',
    "tip-x_up_auth": 'Aynı belgenin değiştirilmemiş hali (aynı boyut). Verilirse SHAP referansı olarak kullanılır: ΣSHAP = f(sahte) − f(authentic).',
    "tip-x_up_gt": 'Değiştirilen bölgenin maskesi (beyaz = sahte). GT kutusu hedefi ve bölge kütlesi grafiği için kullanılır.',
    "tip-x_unet": 'Açıklanacak U-Net modelleri. Her biri kendi gerçek çıktısından türetilen hedefle ayrı açıklanır.',
    "tip-x_res": 'Açıklanacak ResNet-18 modelleri (224 px patch sınıflandırıcı).',
    "tip-b_xpred": 'Seçili modelleri uygulamanın kendi çıkarımıyla (değiştirilmeden) çalıştırır: görüntü skoru, eşik, karar ve tespit bölgeleri. SHAP hedefleri bu tahminlerden türetilir.',
    "tip-x_target": "Hangi skaler çıktının açıklanacağı:<br><b>Tam görüntü · karar</b> (önerilen): modelin uygulamadaki tam görüntü karar skoru (U-Net: tüm olasılık haritasının top-%1 ortalaması · ResNet: tüm ızgaranın maks patch skoru · YOLO11: en güvenli kutu). Açıklama orijinal görüntünün TAMAMINI tek hedef ve tek referansla kapsar; E[f]+ΣSHAP=f(x) doğrulanır.<br><b>Tam görüntü · tespitler</b>: ilk k bölge / patch / kutu, her biri tam görüntüde.<br><b>Tam görüntü · GT</b>: GT kutusu içi ortalama (ResNet: kesişen patch'lerin maks skoru · YOLO: GT ile IoU≥0.5 kutu güveni).<br><b>Döşemeli mozaik</b>: 512 px örtüşen pencerelerin yerel açıklamaları orijinal koordinatlarda birleşir — tabanlar farklı olduğundan tam görüntü SHAP DEĞİLDİR.<br><b>Pencere</b> hedefleri: önceki 512 px pencere / 224 patch açıklamaları (değişmedi).<br><b>Ortak pencere</b>: karşılaştırma (③).",
    "tip-x_k": 'Tespitler hedefinde model başına açıklanacak en fazla bölge / patch sayısı.',
    "tip-x_method": "<b>Partition</b>: modelden bağımsız, hiyerarşik Owen değerleri; referansla maskeleme yapar, toplamsallık kesindir (ΣSHAP = f(x) − f(referans)).<br><b>Gradient</b>: beklenen gradyanlar (fp32); piksel düzeyinde ince doku izi, 5 kanallı U-Net'te RGB / FFT-HP / Laplacian grup katkısı. Toplamsallık yaklaşıktır.",
    "tip-x_ref": "Gizlenen piksellerin neyle doldurulacağı (SHAP'ın karşı-olgusal referansı).<br><b>authentic eş</b>: orijinal belge — en anlamlı referans; ΣSHAP = f(sahte) − f(orijinal).<br><b>blur(32,32)</b>: görüntünün bulanık hali.<br><b>inpaint_telea</b>: çevreden doldurma (yalnız Partition).",
    "tip-x_evals": "Partition açıklayıcısının model değerlendirme bütçesi. Yüksek değer daha ince bölümleme ve daha ayrıntılı harita verir ama süre doğrusal artar (ZeroGPU'da çağrı başına ≤ 120 s).",
    "tip-b_xshap": 'Seçilen hedefleri SHAP ile açıklar. Tahminden bağımsızdır; aynı görüntü, model, hedef, yöntem, referans ve bütçe önbellekteyse yeniden hesaplanmaz. Sonuçlar PNG · JSON · NPZ · MD olarak indirilebilir.',
    "tip-b_xcmp": 'Tüm seçili modelleri aynı pencerede, aynı referans ve bütçeyle açıklar; top-%5 pozitif bölge IoU/Dice, 16×16 blok Spearman korelasyonu ve ortak / modele özgü bölge haritası üretir. Farklı hedef türleri yalnız normalize ölçütlerle karşılaştırılır.',
    "tip-x_show": 'Ayrıntılı panelleri gösterilecek açıklama (model · hedef).',
}

CSS += (XAI.viewer.ZOOM_CSS if XAI_OK else "")  # büyük, en-boy korumalı, yakınlaştırılabilir XAI görüntüleri

TIP_JS = """
() => {
  const tips = """ + json.dumps(TIPS, ensure_ascii=False) + """;
  if (window.__aicTips) return; window.__aicTips = true;
  const box = document.createElement('div'); box.id = 'aic-tip'; document.body.appendChild(box);
  let cur = null;
  const find = (el) => { while (el && el !== document.body) { if (el.id && tips[el.id]) return el; el = el.parentElement; } return null; };
  const place = (t) => {
    const r = t.getBoundingClientRect(), bw = box.offsetWidth, bh = box.offsetHeight, vw = innerWidth, vh = innerHeight;
    let x = r.right + 12, y = r.top;
    if (x + bw > vw - 8) { x = Math.max(8, Math.min(r.left, vw - bw - 8)); y = r.bottom + 8; if (y + bh > vh - 8) y = r.top - bh - 8; }
    y = Math.max(8, Math.min(y, vh - bh - 8));
    box.style.left = x + 'px'; box.style.top = y + 'px';
  };
  document.addEventListener('mouseover', (e) => {
    const t = find(e.target);
    if (t === cur) return;
    cur = t;
    if (!t) { box.classList.remove('show'); return; }
    box.innerHTML = tips[t.id]; box.classList.add('show'); place(t);
  });
  document.addEventListener('scroll', () => { if (cur) place(cur); }, true);
  document.addEventListener('mousedown', () => { cur = null; box.classList.remove('show'); });
}
"""

# ================================================================== XAI · KATMAN FİLTRELERİ (sekme 8 › Filter Visualization)
# SHAP'tan bağımsız ikinci XAI özelliği: app/xai_filters.py — ilk conv katmanlarının checkpoint'teki gerçek kernel'leri
# + modelin kendi forward'ından alınan aktivasyon haritaları (bağımsız F.conv2d ile doğrulanır). Ayrıntı: docs/XAI_FILTERS.md
try:
    import xai_filters as XFILT
    XFILT_OK, _XFILT_ERR = True, ""
except Exception as _e:  # paket yoksa diğer sekmeler etkilenmez
    XFILT, XFILT_OK, _XFILT_ERR = None, False, str(_e)

preload_models()
xai_setup()
_IDX = dataset_index()
_UNETS = [n for n in list_unet_ckpts() if n in _REG or not _REG]
_RESNETS = [n for n in list_resnet_ckpts() if n in _REG or not _REG]
_DEF_UNETS = [n for n in ("PHASE2/unet_bbox_s42", "PHASE2/unet_pixel_eps8_morph_s42", "PHASE2/unet_glyph_v2_s42",
                             "YOLO/aiforge-tiles-yolo11s-img1024-conserv-s42") if n in _UNETS] or _UNETS[:1]
_DEF_RESNET = [n for n in ("PHASE1/resnet18_ft_best",) if n in _RESNETS]
_SOURCES = ["hepsi"] + sorted(x for x in _IDX["source_dataset"].unique() if x)
_TOOLS = ["hepsi"] + sorted(x for x in _IDX["tool"].unique() if x)
_AREAS = ["hepsi"] + [a for a in ("<1%", "1-5%", "5-20%", ">20%") if a in set(_IDX["area_bin"])]

with gr.Blocks(title="AIC515 · AIForge-Doc Sahtecilik Lokalizasyonu") as demo:
    gr.Markdown("# AIC515 Görüntü İşleme · AIForge-Doc — Belge Sahteciliği Tespiti ve Lokalizasyonu")

    st_orig = gr.State(None); st_cur = gr.State(None); st_mask = gr.State(None); st_auth = gr.State(None); st_steps = gr.State([])
    st_sim = gr.State(None)

    # ---------------------------------------------------------------- 0 · SİMÜLASYON
    with gr.Tab("0 · Simülasyon"):
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("#### 1) Örnek seçimi")
                with gr.Row():
                    f_split = gr.Dropdown(["hepsi", "eğitim (training)", "test (testing)"], value="test (testing)", label="Bölüm", elem_id="tip-f_split")
                    f_source = gr.Dropdown(_SOURCES, value="hepsi", label="Kaynak veri seti", elem_id="tip-f_source")
                with gr.Row():
                    f_tool = gr.Dropdown(_TOOLS, value="hepsi", label="Inpainting üreticisi", elem_id="tip-f_tool")
                    f_area = gr.Dropdown(_AREAS, value="hepsi", label="Kurcalanan alan oranı", elem_id="tip-f_area")
                with gr.Row():
                    f_marker = gr.Dropdown(["hepsi", "var", "yok"], value="hepsi", label="Yeşil işaretçi kalıntısı", elem_id="tip-f_marker")
                    f_mp = gr.Slider(0.1, 40, value=12, step=0.1, label="En fazla megapiksel", elem_id="tip-f_mp")
                f_pair = gr.Checkbox(False, label="Yalnız authentic eşi olanlar (piksel-GT ve işaretçi senaryoları için gerekli)", elem_id="tip-f_pair")
                with gr.Row():
                    b_filter = gr.Button("Filtrele", elem_id="tip-b_filter")
                    b_rand = gr.Button("🎲 Rastgele seç", variant="secondary", elem_id="tip-b_rand")
                f_count = gr.Markdown(" ", elem_classes="small-md")
                dd_key = gr.Dropdown(choices=sample_choices(filter_index("test (testing)", max_mp=12)), label="Örnek (bölüm:kimlik)", filterable=True, allow_custom_value=True, elem_id="tip-dd_key")
                gr.Markdown("#### 2) Senaryo")
                rd_input = gr.Radio(INPUT_TYPES, value=INPUT_TYPES[0], label="Girdi senaryosu", elem_id="tip-rd_input")
                dd_deg = gr.Dropdown(DEGRADATIONS, value=DEGRADATIONS[0], label="Test-zamanı bozulma", elem_id="tip-dd_deg")
                ck_pipe = gr.Checkbox(False, label="Ön işleme sekmesindeki pipeline'ı da uygula", elem_id="tip-ck_pipe")
                gr.Markdown("#### 3) Modeller")
                cg_unet = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS, label="Lokalizasyon modelleri (U-Net · DeepLabV3+ · YOLO11 kutu tespiti)", elem_id="tip-cg_unet")
                cg_res = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _RESNETS], value=_DEF_RESNET, label="ResNet-18 (patch sınıflandırma)", elem_id="tip-cg_res")
                sl_stride_sim = gr.Slider(56, 224, value=112, step=8, label="ResNet kayan pencere adımı (px)", elem_id="tip-sl_stride_sim")
                b_sim = gr.Button("▶ Simülasyonu başlat", variant="primary", size="lg", elem_id="tip-b_sim")
                b_to_tabs = gr.Button("Bu örneği ön işleme / model sekmelerine aktar", elem_id="tip-b_to_tabs")
            with gr.Column(scale=2):
                with gr.Row():
                    pv_img = gr.Image(label="Seçilen örnek", type="numpy", height=300, interactive=False)
                    pv_gt = gr.Image(label="Resmî GT (mavi kutu)", type="numpy", height=300, interactive=False)
                pv_info = gr.Markdown("Bir örnek seçin ya da 🎲 ile rastgele seçin.")
                sim_log = gr.Markdown(" ")
                sim_verdict = gr.Markdown(" ", elem_id="verdict")
                sim_df = gr.Dataframe(label="Model bazında sonuçlar", wrap=True, interactive=False)
                sim_zoom = gr.Image(label="Yakın plan: alan kutusu çevresi", type="numpy", interactive=False)
                sim_gal = gr.Gallery(label="Adım adım çıktılar", columns=3, height=520, object_fit="contain", preview=False, format="jpeg")
                sim_files = gr.File(label="Simülasyon raporu (MD · JSON · CSV · PNG)", file_count="multiple")

    # ---------------------------------------------------------------- 1 · TOPLU TEST
    with gr.Tab("1 · Toplu test"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Row():
                    t_split = gr.Dropdown(["hepsi", "eğitim (training)", "test (testing)"], value="test (testing)", label="Bölüm", elem_id="tip-t_split")
                    t_source = gr.Dropdown(_SOURCES, value="hepsi", label="Kaynak", elem_id="tip-t_source")
                with gr.Row():
                    t_tool = gr.Dropdown(_TOOLS, value="hepsi", label="Üretici", elem_id="tip-t_tool")
                    t_area = gr.Dropdown(_AREAS, value="hepsi", label="Alan", elem_id="tip-t_area")
                with gr.Row():
                    t_marker = gr.Dropdown(["hepsi", "var", "yok"], value="hepsi", label="İşaretçi", elem_id="tip-t_marker")
                    t_mp = gr.Slider(0.1, 40, value=6, step=0.1, label="En fazla MP", elem_id="tip-t_mp")
                with gr.Row():
                    t_n = gr.Slider(1, 100, value=16, step=1, label="Örnek sayısı N", elem_id="tip-t_n")
                    t_seed = gr.Number(42, label="Tohum", precision=0, elem_id="tip-t_seed")
                t_mode = gr.Radio(BATCH_MODES, value=BATCH_MODES[1], label="Mod", elem_id="tip-t_mode")
                t_deg = gr.Dropdown(DEGRADATIONS, value=DEGRADATIONS[0], label="Bozulma", elem_id="tip-t_deg")
                t_unet = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS[:2], label="Lokalizasyon (U-Net · DeepLabV3+ · YOLO11)", elem_id="tip-t_unet")
                t_res = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _RESNETS], value=[], label="ResNet-18", elem_id="tip-t_res")
                b_batch = gr.Button("▶ Toplu testi başlat", variant="primary", size="lg", elem_id="tip-b_batch")
            with gr.Column(scale=2):
                t_prog = gr.Markdown(" ")
                t_agg = gr.Dataframe(label="Model bazında özet", wrap=True, interactive=False)
                t_fig = gr.Plot(label="Özet grafik")
                t_per = gr.Dataframe(label="Örnek bazında sonuçlar", wrap=True, interactive=False)
                t_files = gr.File(label="Toplu test çıktıları", file_count="multiple")

    # ---------------------------------------------------------------- 2 · GİRDİ
    with gr.Tab("2 · Girdi"):
        with gr.Row():
            with gr.Column(scale=1):
                in_img = gr.Image(label="Sahte (forged) görüntü — PNG/JPG", type="numpy", height=320, elem_id="tip-in_img")
                in_mask = gr.Image(label="GT maskesi (isteğe bağlı, 0/255)", type="numpy", height=160, image_mode="L", elem_id="tip-in_mask")
                in_auth = gr.Image(label="Authentic eş (isteğe bağlı; piksel-değişim GT için)", type="numpy", height=160, elem_id="tip-in_auth")
                btn_load = gr.Button("Yüklenenleri kullan", variant="primary", elem_id="tip-btn_load")
                dd_sample = gr.Dropdown(choices=sample_choices(), label="örnek (bölüm:kimlik)", filterable=True, allow_custom_value=True, elem_id="tip-dd_sample")
                btn_sample = gr.Button("Örneği yükle", elem_id="tip-btn_sample")
                with gr.Accordion("💾 Taslaklar (sonra kullanmak için kaydet)", open=True):
                    tx_draft = gr.Textbox(label="Taslak adı", placeholder="ör. cord fişi · indirim alanı", elem_id="tip-tx_draft")
                    btn_dsave = gr.Button("💾 Taslak olarak kaydet", variant="secondary", elem_id="tip-btn_dsave")
                    dd_draft = gr.Dropdown(choices=draft_list(), label="Kayıtlı taslaklar", filterable=True, elem_id="tip-dd_draft")
                    with gr.Row():
                        btn_dload = gr.Button("📂 Taslağı yükle", variant="primary", elem_id="tip-btn_dload")
                        btn_drefresh = gr.Button("↻", scale=0, min_width=48)
                        btn_ddel = gr.Button("🗑 Sil", variant="stop", scale=0, min_width=72, elem_id="tip-btn_ddel")
                    draft_md = gr.Markdown(" ", elem_classes="small-md")
            with gr.Column(scale=1):
                info_md = gr.Markdown("Görüntü yükleyin ya da örnek seçin.")
                view_orig = gr.Image(label="Orijinal", type="numpy", height=360, interactive=False)
                view_gt = gr.Image(label="GT maskesi", type="numpy", height=160, interactive=False)
                view_auth = gr.Image(label="Authentic eş", type="numpy", height=160, interactive=False)

    # ---------------------------------------------------------------- 3 · ÖN İŞLEME
    with gr.Tab("3 · Ön işleme (aşamalı pipeline)"):
        with gr.Row():
            with gr.Column(scale=1):
                dd_cat = gr.Dropdown(choices=CATEGORIES, value=CATEGORIES[0], label="Kategori (ders haftası)", elem_id="tip-dd_cat")
                first_ops = list(OPS[CATEGORIES[0]].keys())
                dd_op = gr.Dropdown(choices=first_ops, value=first_ops[0], label="İşlem", elem_id="tip-dd_op")
                sl1 = gr.Slider(visible=False, elem_id="tip-sl1"); sl2 = gr.Slider(visible=False, elem_id="tip-sl2"); sl3 = gr.Slider(visible=False, elem_id="tip-sl3")
                dd_mode = gr.Dropdown(choices=[], visible=False, label="mod", elem_id="tip-dd_mode")
                op_doc = gr.Markdown(" ", elem_classes="small-md")
                with gr.Row():
                    btn_prev = gr.Button("Önizle", elem_id="tip-btn_prev"); btn_apply = gr.Button("Pipeline'a ekle (uygula)", variant="primary", elem_id="tip-btn_apply")
                with gr.Row():
                    btn_undo = gr.Button("Geri al", elem_id="tip-btn_undo"); btn_reset = gr.Button("Sıfırla", elem_id="tip-btn_reset")
                dd_preset = gr.Dropdown(choices=list(PRESETS), value="— seçiniz —", label="Hazır pipeline (preset)", elem_id="tip-dd_preset")
                btn_preset = gr.Button("Preseti uygula", elem_id="tip-btn_preset")
                steps_box = gr.Markdown(steps_md([]))
            with gr.Column(scale=2):
                with gr.Row():
                    img_before = gr.Image(label="Önce (mevcut pipeline çıktısı)", type="numpy", height=380, interactive=False)
                    img_after = gr.Image(label="Sonra", type="numpy", height=380, interactive=False)
                hist_plot = gr.Plot(label="Histogram")
                status = gr.Markdown(" ")
                with gr.Accordion("Klasik öznitelikler (56; LBP · GLCM · DCT · Laplacian · FFT) — 256 px, alan kutusu merkezli", open=False):
                    btn_feat = gr.Button("Öznitelikleri hesapla")
                    feat_df = gr.Dataframe(label="öznitelik vektörü", wrap=True)

    # ---------------------------------------------------------------- 4 · U-NET
    with gr.Tab("4 · U-Net lokalizasyonu"):
        with gr.Row():
            with gr.Column(scale=1):
                dd_unet = gr.Dropdown(choices=[(friendly(n), n) for n in _UNETS], value=_DEF_UNETS[0] if _DEF_UNETS else None, label="Model (U-Net · DeepLabV3+ · YOLO11)", elem_id="tip-dd_unet")
                rd_src = gr.Radio(["Orijinal görüntü", "Pipeline çıktısı"], value="Orijinal görüntü", label="Model girdisi", elem_id="tip-rd_src")
                rd_gt = gr.Radio(["resmî bbox maskesi", "piksel-değişim (authentic'ten türet, eps8_morph)"], value="resmî bbox maskesi", label="Değerlendirme GT'si", elem_id="tip-rd_gt")
                sl_thr = gr.Slider(0.05, 0.95, value=default_tau(_DEF_UNETS[0], "bbox") if _DEF_UNETS else 0.5, step=0.05, label="eşik τ", elem_id="tip-sl_thr")
                ck_otsu = gr.Checkbox(False, label="Eşik yerine Otsu (olasılık haritası üzerinde)", elem_id="tip-ck_otsu")
                sl_morph = gr.Slider(0, 15, value=0, step=1, label="Son-işleme morfoloji (açma+kapama) çekirdeği; 0 = kapalı", elem_id="tip-sl_morph")
                btn_unet = gr.Button("U-Net çalıştır", variant="primary", elem_id="tip-btn_unet")
                btn_cmp = gr.Button("Karşılaştır: orijinal vs pipeline çıktısı", elem_id="tip-btn_cmp")
                unet_df = gr.Dataframe(label="ölçütler", wrap=True)
                unet_note = gr.Markdown(" ")
            with gr.Column(scale=2):
                with gr.Row():
                    unet_heat = gr.Image(label="Olasılık haritası", type="numpy", height=360, interactive=False)
                    unet_ov = gr.Image(label="Bindirme (kırmızı tahmin · mavi GT · yeşil tahmin-bbox)", type="numpy", height=360, interactive=False)
                unet_bin = gr.Image(label="İkili maske", type="numpy", height=240, interactive=False)
                with gr.Accordion("Karşılaştırma sonucu (orijinal vs pipeline)", open=False):
                    with gr.Row():
                        cmp_a = gr.Image(label="Orijinal girdi → tahmin", type="numpy", height=320, interactive=False)
                        cmp_b = gr.Image(label="Pipeline çıktısı → tahmin", type="numpy", height=320, interactive=False)
                    cmp_df = gr.Dataframe(label="ölçüt karşılaştırması", wrap=True)
                    cmp_note = gr.Markdown(" ")
        st_dummy = gr.State(None)

    # ---------------------------------------------------------------- 5 · RESNET
    with gr.Tab("5 · ResNet-18 patch sınıflandırma"):
        with gr.Row():
            with gr.Column(scale=1):
                dd_rn = gr.Dropdown(choices=[(friendly(n), n) for n in _RESNETS], value=_DEF_RESNET[0] if _DEF_RESNET else (_RESNETS[0] if _RESNETS else None), label="ResNet-18 checkpoint", elem_id="tip-dd_rn")
                rd_src2 = gr.Radio(["Orijinal görüntü", "Pipeline çıktısı"], value="Orijinal görüntü", label="Model girdisi", elem_id="tip-rd_src2")
                sl_stride = gr.Slider(32, 224, value=112, step=16, label="kayan pencere adımı (patch 224, natif çözünürlük)", elem_id="tip-sl_stride")
                btn_rn = gr.Button("ResNet-18 çalıştır", variant="primary", elem_id="tip-btn_rn")
                rn_df = gr.Dataframe(label="ölçütler", wrap=True)
            with gr.Column(scale=2):
                rn_ov = gr.Image(label="Patch skor ısı haritası (mavi: GT bbox)", type="numpy", height=480, interactive=False)

    # ---------------------------------------------------------------- 6 · YEŞİL İŞARETÇİ
    with gr.Tab("6 · Yeşil işaretçi sondası"):
        gr.Markdown("Veri seti üretim hattının bıraktığı **yeşil işaretçi kalıntısı** (PHASE 2 Ek-C): G − max(R,B) > 50 ve G > 110. "
                    "Sahtelerin %75,4'ünde bbox iç kenarında görülür ve tüm modellerin kullandığı bir kısayoldur. "
                    "Simülasyondaki 'işaretçi onarılmış' ve 'yalnız işaretçi' senaryoları bu etkiyi ölçer.")
        btn_green = gr.Button("Mevcut görüntüde işaretçiyi ara", variant="primary", elem_id="tip-btn_green")
        with gr.Row():
            green_ov = gr.Image(label="İşaretçi pikselleri (magenta)", type="numpy", height=420, interactive=False)
            green_md = gr.Markdown(" ")

    # ---------------------------------------------------------------- 7 · SONUÇLAR
    with gr.Tab("7 · Sonuçlar ve model kartı"):
        gr.Markdown(results_markdown())
        gr.Markdown(
            "### Notlar\n"
            "- Modeller **ham PNG, natif çözünürlük, ImageNet normalizasyonu** ile eğitildi; ön işleme ve bozulmalar test-zamanı müdahaleleridir.\n"
            "- Tam görüntü çıkarımı 512 px pencere / 448 px adım / örtüşme ortalaması; uzun kenar > 4096 px ise 4096'ya küçültülür.\n"
            "- Metrikler boru hattıyla aynı tanımla hesaplanır: 8-bit nicemlenmiş olasılık, 256 kutulu histogram, tüm pikseller (alt-örnekleme yok).\n"
            "- Görüntü-düzeyi skor = olasılık haritasının en yüksek %1 piksel ortalaması; eşik val'de Youden-J ile seçildi.\n"
            "- Veri seti: AIForge-Doc v1 (Scam.AI, CC BY-NC-SA 4.0, kapılı); harici test: AIForge-Doc v2 (GPT-Image-2, aynı lisans), yalnız sonuç tabloları.")

    # ---------------------------------------------------------------- 8 · XAI ANALYSIS (SHAP + Filter Visualization)
    with gr.Tab("8 · XAI Analysis"):
        gr.Markdown("**XAI Analysis** — birbirinden bağımsız iki açıklanabilirlik özelliği: **SHAP Analysis** modelin tahmin "
                    "kararına görüntü bölgelerinin katkısını açıklar (`xai/`); **Filter Visualization** eğitilmiş modelin ilk "
                    "konvolüsyon katmanlarındaki gerçek filtre ağırlıklarını ve bunların görüntüye uygulanmasıyla oluşan aktivasyon "
                    "haritalarını gösterir (`xai_filters.py`; filtre eğitilmez/üretilmez, ağırlıklar ve mimari değişmez).", elem_classes="small-md")
        with gr.Tabs():
            # ------------------------------------------------ 8a · SHAP Analysis
            with gr.Tab("SHAP Analysis"):
                st_xpred = gr.State(None); st_xview = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        gr.Markdown("#### 1) Görüntü ve modeller")
                        x_src = gr.Radio(XAI_SOURCES, value=XAI_SOURCES[0], label="Görüntü kaynağı", elem_id="tip-x_src")
                        with gr.Group(visible=False) as x_up_grp:
                            x_up_img = gr.Image(label="Görüntü", type="numpy", height=200, elem_id="tip-x_up_img")
                            with gr.Row():
                                x_up_auth = gr.Image(label="Authentic eş (isteğe bağlı)", type="numpy", height=120, elem_id="tip-x_up_auth")
                                x_up_gt = gr.Image(label="GT maskesi (isteğe bağlı)", type="numpy", height=120, image_mode="L", elem_id="tip-x_up_gt")
                        x_unet = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _UNETS], value=[n for n in _DEF_UNETS if not n.startswith("YOLO/")], label="U-Net / DeepLabV3+ / YOLO11 (YOLO: yalnız tam görüntü hedefleri)", elem_id="tip-x_unet")
                        x_res = gr.CheckboxGroup(choices=[(friendly(n), n) for n in _RESNETS], value=_DEF_RESNET, label="ResNet-18", elem_id="tip-x_res")
                        gr.Markdown(xai_unsupported_md(), elem_classes="small-md")
                        b_xpred = gr.Button("① Tahmin yap", variant="primary", elem_id="tip-b_xpred")
                        gr.Markdown("#### 2) SHAP")
                        x_target = gr.Radio(XAI_TARGETS, value=XAI_TARGETS[0], label="Açıklanacak hedef", elem_id="tip-x_target")
                        x_k = gr.Slider(1, 3, value=3, step=1, label="Tespit sayısı k", elem_id="tip-x_k")
                        x_method = gr.Radio(XAI_METHODS, value=XAI_METHODS[0], label="Yöntem", elem_id="tip-x_method")
                        x_ref = gr.Dropdown(XAI_REFS, value=XAI_REFS[0], label="Referans (arka plan)", elem_id="tip-x_ref")
                        x_evals = gr.Slider(200, 3000, value=800, step=100, label="max_evals (Partition bütçesi)", elem_id="tip-x_evals")
                        b_xshap = gr.Button("② SHAP analizini başlat", variant="primary", elem_id="tip-b_xshap")
                        b_xcmp = gr.Button("③ Modelleri karşılaştır", elem_id="tip-b_xcmp")
                    with gr.Column(scale=2):
                        x_status = gr.Markdown("① ile tahmin yapın, ② ile seçilen hedefleri SHAP ile açıklayın, ③ ile modelleri aynı pencerede karşılaştırın.")
                        x_pred_df = gr.Dataframe(label="Tahminler (uygulamanın kendi çıkarımı)", wrap=True, interactive=False)
                        x_res_df = gr.Dataframe(label="SHAP sonuçları", wrap=True, interactive=False)
                        x_loc = gr.Image(label="Açıklanan alanların konumu (tam görüntü hedefinde çerçeve = görüntünün tamamı)", type="numpy", height=300, interactive=False)
                        x_show = gr.Dropdown(choices=[], label="Gösterilecek açıklama", elem_id="tip-x_show")
                        x_sum = gr.Markdown(" ")
                        gr.Markdown("Görüntüler en-boy oranı korunarak gösterilir · **fare tekerleği**: imleç çevresinde yakınlaştır (32×'e kadar) · "
                                    "**sürükle**: kaydır · **çift tık**: sıfırla · sağ üst düğmeler: tam ekran / indir. Natif çözünürlüklü "
                                    "PNG/NPY ve 300 dpi PNG + PDF yayın şekilleri aşağıdaki dışa aktarımdadır.", elem_classes="small-md")
                        x_slider = gr.ImageSlider(label="Yan yana karşılaştırma: orijinal ↔ SHAP bindirmesi (kaydırıcıyı sürükleyin)", type="numpy", height=680)
                        with gr.Row():
                            x_p0 = gr.Image(label="Orijinal — tam görüntü ya da açıklanan pencere (mavi: GT · magenta: işaretçi · yeşil: hedef)", type="numpy", height=560, interactive=False, elem_classes=["xai-zoom"])
                            x_p2 = gr.Image(label="Bindirme (aynı koordinat sistemi)", type="numpy", height=560, interactive=False, elem_classes=["xai-zoom"])
                        with gr.Row():
                            x_p1 = gr.Image(label="SHAP ısı haritası (kırmızı: SAHTE yönü · mavi: GERÇEK yönü)", type="numpy", height=560, interactive=False, elem_classes=["xai-zoom"])
                        with gr.Row():
                            x_p3 = gr.Image(label="Pozitif katkı (SAHTE'yi artıran bölgeler)", type="numpy", height=480, interactive=False, elem_classes=["xai-zoom"])
                            x_p4 = gr.Image(label="Negatif katkı (GERÇEK'e iten bölgeler)", type="numpy", height=480, interactive=False, elem_classes=["xai-zoom"])
                        x_p5 = gr.Image(label="Sınıf panelleri: SAHTE | GERÇEK (= −SAHTE)", type="numpy", height=480, interactive=False, elem_classes=["xai-zoom"])
                        x_dist = gr.Image(label="Açıklanan hedef sınıfına (SAHTE) katkı dağılımı: histogram · birikimli kütle · toplamlar (GT içi/dışı)", type="numpy", height=360, interactive=False)
                        x_mass = gr.Image(label="Pozitif SHAP kütlesinin bölgelere dağılımı (GT kutusu / kenar halkası / işaretçi / geri kalan)", type="numpy", interactive=False)
                        with gr.Accordion("Tüm paneller (galeri)", open=False):
                            x_gal = gr.Gallery(columns=3, height=520, object_fit="contain", preview=False, format="png")
                        with gr.Accordion("Model karşılaştırması (③)", open=True):
                            x_cmp_md = gr.Markdown(" ")
                            x_cmp_img = gr.Image(label="IoU · Spearman · ortak/özgü bölge", type="numpy", interactive=False)
                            x_cmp_df = gr.Dataframe(label="Karşılaştırılan açıklamalar", wrap=True, interactive=False)
                        x_files = gr.File(label="Dışa aktarım (PNG paneller · JSON · NPZ · MD rapor)", file_count="multiple")

            # ------------------------------------------------ 8b · Filter Visualization / Katman Filtreleri
            with gr.Tab("Filter Visualization / Katman Filtreleri"):
                if XFILT_OK and TORCH_OK:
                    def _xf_sample(key):
                        try:
                            return load_dataset_sample(key)
                        except Exception as e:
                            raise gr.Error(f"Örnek yüklenemedi: {e}")
                    XFILT.build_tab(gr, models_dir=MODELS, yolo_runs_dir=ROOT / "phase3_yolo" / "runs", gpu=gpu,
                                    freq_fn=project_freq_channels, load_sample=_xf_sample,
                                    sample_choices=sample_choices(filter_index("test (testing)", max_mp=12)),
                                    image_states=(st_orig, st_mask), default_key="PHASE2/unet_bbox_s42", store_fp16=True if ZERO_GPU else None)
                else:
                    gr.Markdown(f"⚠️ Katman filtreleri kullanılamıyor: `{_XFILT_ERR or _TORCH_ERR}`")


    # ---------------------------------------------------------------- olaylar
    filt_in = [f_split, f_source, f_tool, f_area, f_pair, f_marker, f_mp]
    b_filter.click(sim_filter, filt_in, [dd_key, f_count])
    b_rand.click(sim_random, filt_in, [dd_key, f_count])
    dd_key.change(sim_preview, [dd_key], [pv_img, pv_gt, pv_info])
    b_sim.click(simulate, [dd_key, rd_input, dd_deg, ck_pipe, st_steps, cg_unet, cg_res, sl_stride_sim],
                [sim_log, sim_gal, sim_df, sim_verdict, sim_zoom, sim_files, st_sim])
    b_to_tabs.click(sim_to_tabs, [st_sim], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth, steps_box])

    b_batch.click(batch_test, [t_split, t_source, t_tool, t_area, t_marker, t_n, t_seed, t_mode, t_deg, t_unet, t_res, t_mp],
                  [t_prog, t_per, t_agg, t_fig, t_files])

    btn_load.click(on_upload, [in_img, in_mask, in_auth], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth]).then(
        lambda: steps_md([]), None, steps_box)
    btn_sample.click(on_sample, [dd_sample], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth]).then(
        lambda: steps_md([]), None, steps_box)
    btn_dsave.click(draft_save, [tx_draft, st_orig, st_mask, st_auth, st_steps, in_img, in_mask, in_auth, dd_sample], [dd_draft, draft_md])
    btn_dload.click(draft_load, [dd_draft], [st_orig, st_cur, st_mask, st_auth, st_steps, info_md, view_orig, img_before, view_gt, view_auth,
                                             steps_box, in_img, in_mask, in_auth, draft_md]).then(lambda c: c, [st_cur], [img_after])
    btn_ddel.click(draft_delete, [dd_draft], [dd_draft, draft_md])
    btn_drefresh.click(lambda v: _draft_dd(v), [dd_draft], [dd_draft])
    demo.load(lambda: _draft_dd(), None, [dd_draft])

    dd_cat.change(op_choices, [dd_cat], [dd_op, sl1, sl2, sl3, dd_mode, op_doc])
    dd_op.change(param_updates, [dd_cat, dd_op], [sl1, sl2, sl3, dd_mode, op_doc])
    demo.load(param_updates, [dd_cat, dd_op], [sl1, sl2, sl3, dd_mode, op_doc])

    btn_prev.click(preview, [st_cur, dd_cat, dd_op, sl1, sl2, sl3, dd_mode, st_auth], [img_after, hist_plot, status])
    btn_apply.click(apply_step, [st_cur, st_orig, st_steps, dd_cat, dd_op, sl1, sl2, sl3, dd_mode, st_auth],
                    [st_cur, img_after, st_steps, steps_box, hist_plot, status]).then(lambda c: c, [st_cur], [img_before])
    btn_undo.click(undo, [st_orig, st_steps, st_auth], [st_cur, img_before, st_steps, steps_box, status]).then(lambda c: c, [st_cur], [img_after])
    btn_reset.click(reset, [st_orig], [st_cur, img_before, st_steps, steps_box, status]).then(lambda c: c, [st_cur], [img_after])
    btn_preset.click(apply_preset, [st_orig, dd_preset, st_auth], [st_cur, img_after, st_steps, steps_box, hist_plot, status]).then(lambda c: c, [st_cur], [img_before])
    btn_feat.click(classical_features, [st_cur, st_mask], [feat_df])

    btn_unet.click(run_unet, [st_orig, st_cur, st_mask, st_auth, dd_unet, rd_src, sl_thr, ck_otsu, sl_morph, rd_gt],
                   [unet_heat, unet_ov, unet_bin, unet_df, unet_note, st_dummy])
    btn_cmp.click(compare_unet, [st_orig, st_cur, st_mask, st_auth, dd_unet, sl_thr, ck_otsu, sl_morph, rd_gt, st_steps],
                  [cmp_a, cmp_b, cmp_df, cmp_note])
    dd_unet.change(lambda m, g: default_tau(m, "bbox" if g.startswith("resmî") else "pixel_eps8_morph") if m else 0.5, [dd_unet, rd_gt], [sl_thr])
    rd_gt.change(lambda m, g: default_tau(m, "bbox" if g.startswith("resmî") else "pixel_eps8_morph") if m else 0.5, [dd_unet, rd_gt], [sl_thr])

    btn_rn.click(run_resnet, [st_orig, st_cur, st_mask, dd_rn, rd_src2, sl_stride], [rn_ov, rn_df])
    btn_green.click(run_green, [st_cur, st_mask, st_auth], [green_ov, green_md])


    # ---------------------------------------------------------------- 8 · XAI Analysis › SHAP olayları
    x_src.change(lambda v: gr.update(visible=v == XAI_SOURCES[3]), [x_src], [x_up_grp])
    b_xpred.click(xai_predict, [x_src, x_unet, x_res, st_orig, st_cur, st_mask, st_auth, st_sim, x_up_img, x_up_auth, x_up_gt],
                  [x_pred_df, x_status, st_xpred])
    _x_panels = [x_sum, x_p0, x_p1, x_p2, x_p3, x_p4, x_p5, x_slider, x_dist]
    b_xshap.click(xai_run, [st_xpred, x_target, x_k, x_method, x_ref, x_evals],
                  [x_status, x_res_df, x_loc, x_show, st_xview, x_mass, x_gal, x_files] + _x_panels)
    x_show.change(_xai_show_outputs, [st_xview, x_show], _x_panels)
    b_xcmp.click(xai_compare, [st_xpred, x_method, x_ref, x_evals], [x_cmp_md, x_cmp_img, x_cmp_df, x_files])

    demo.load(None, None, None, js=TIP_JS)  # sol panel açıklamaları
    if XAI_OK:
        demo.load(None, None, None, js=XAI.viewer.ZOOM_JS)  # XAI görüntüleri: tekerlek yakınlaştırma · sürükleme · çift tık sıfırlama

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "7860"))
    host = os.environ.get("APP_HOST", "0.0.0.0" if IN_SPACE else "127.0.0.1")
    demo.launch(server_name=host, server_port=port, inbrowser=(os.environ.get("APP_NO_BROWSER") is None and not IN_SPACE),
                css=CSS, show_error=True)
