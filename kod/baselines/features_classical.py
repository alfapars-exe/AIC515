"""
features_classical.py — Klasik görüntü işleme öznitelikleri (AIC515 3.–11. hafta) (TR).

Gri tonlamalı 256×256 patch'ten el-yapımı öznitelik vektörü:
  * 8×8 blok DCT: ilk 15 zigzag AC katsayısının bloklar arası mean/std (30), DC std (1),
    enerji oranları: düşük (AC 1-15) / orta (16-35) / yüksek (36-63) bant enerjisinin toplam AC enerjisine oranı (3)
  * LBP (P=8, R=1, uniform) histogramı — 10 bin (10)
  * GLCM (32 gri seviye, d=1, 4 açı ortalaması): contrast, homogeneity, energy, correlation (4)
  * 3×3 Laplacian artığı: mean|r|, std, kurtosis, skew (4)
  * FFT radyal spektrum: düşük / orta / yüksek bant enerji oranları + log-spektrum eğimi (4)
Toplam 56 öznitelik. `feature_names()` sırayı verir.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.stats import kurtosis, skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ---- zigzag sırası (8×8)
def _zigzag_indices(n=8):
    idx = sorted(((i, j) for i in range(n) for j in range(n)),
                 key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 else p[0]))
    return idx

ZZ = _zigzag_indices(8)
_D = cv2.dct(np.eye(8, dtype=np.float32))  # ortonormal DCT-II matrisi (8×8)


def block_dct_features(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) - 128.0
    H, W = g.shape
    H8, W8 = H // 8 * 8, W // 8 * 8
    blocks = g[:H8, :W8].reshape(H8 // 8, 8, W8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    coef = _D @ blocks @ _D.T                      # (N,8,8)
    zz = np.stack([coef[:, i, j] for (i, j) in ZZ], axis=1)  # (N,64) zigzag
    ac = zz[:, 1:]
    feats = []
    feats.extend(ac[:, :15].mean(0)); feats.extend(ac[:, :15].std(0))
    feats.append(zz[:, 0].std())
    e = (ac ** 2).sum(0)
    tot = e.sum() + 1e-8
    feats.extend([e[:15].sum() / tot, e[15:35].sum() / tot, e[35:].sum() / tot])
    return np.asarray(feats, dtype=np.float32)


def lbp_features(gray_u8: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
    return hist.astype(np.float32)


def glcm_features(gray_u8: np.ndarray) -> np.ndarray:
    q = (gray_u8 // 8).astype(np.uint8)  # 32 seviye
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    return np.array([graycoprops(glcm, p).mean() for p in ("contrast", "homogeneity", "energy", "correlation")],
                    dtype=np.float32)


def laplacian_features(gray: np.ndarray) -> np.ndarray:
    r = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=1)  # 3×3 [0 1 0;1 -4 1;0 1 0]
    f = r.ravel()
    return np.array([np.abs(f).mean(), f.std(), kurtosis(f, fisher=True), skew(f)], dtype=np.float32)


def fft_radial_features(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    P = np.abs(F) ** 2
    H, W = g.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    rn = r / rmax  # 0..~1.41
    tot = P[rn <= 1].sum() + 1e-8
    low = P[(rn > 0) & (rn <= 0.1)].sum() / tot
    mid = P[(rn > 0.1) & (rn <= 0.4)].sum() / tot
    high = P[(rn > 0.4) & (rn <= 1.0)].sum() / tot
    # radyal ortalama log-spektrum eğimi (1/f benzeri davranış)
    rb = np.clip((r / rmax * 32).astype(int), 0, 40)
    prof = np.bincount(rb.ravel(), weights=P.ravel()) / (np.bincount(rb.ravel()) + 1e-8)
    k = np.arange(1, 32); y = np.log(prof[1:32] + 1e-8); x = np.log(k)
    slope = np.polyfit(x, y, 1)[0]
    return np.array([low, mid, high, slope], dtype=np.float32)


def feature_names():
    names = [f"dct_ac{i+1}_mean" for i in range(15)] + [f"dct_ac{i+1}_std" for i in range(15)] + ["dct_dc_std",
             "dct_E_low", "dct_E_mid", "dct_E_high"]
    names += [f"lbp_u{i}" for i in range(10)]
    names += ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
    names += ["lap_mean_abs", "lap_std", "lap_kurtosis", "lap_skew"]
    names += ["fft_low", "fft_mid", "fft_high", "fft_slope"]
    return names


def extract_features(patch_rgb_u8: np.ndarray) -> np.ndarray:
    gray_u8 = cv2.cvtColor(patch_rgb_u8, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32)
    return np.concatenate([block_dct_features(gray), lbp_features(gray_u8), glcm_features(gray_u8),
                           laplacian_features(gray), fft_radial_features(gray)])


def extract_many(patches: np.ndarray) -> np.ndarray:
    X = np.stack([extract_features(p) for p in patches])
    assert X.shape[1] == len(feature_names()), (X.shape, len(feature_names()))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
