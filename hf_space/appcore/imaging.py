"""appcore.imaging — görüntü/frekans yardımcıları, yeşil işaretçi dedektörü, piksel-değişim GT türetimi."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar

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


# ================================================================== METRİKLER
def q_u8(prob):
    return np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)
