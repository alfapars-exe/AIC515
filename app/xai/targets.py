"""Açıklama hedefleri: modelin gerçek tahmin çıktısından türetilen SKALER hedefler ve bunların konumu.

Koordinatlar
  prep="unet"   : U-Net'in gördüğü görüntü — uzun kenar > max_side ise ölçeklenmiş, kısa kenarı < pencere ise
                  sağ/alt kenardan replicate ile doldurulmuş (app._unet_prepare ile aynı).
  prep="resnet" : ResNet'in gördüğü görüntü — yalnız < 224 ise doldurulmuş, ölçekleme yok (app._predict_resnet).
Pencere (y, x, h, w) bu koordinatlardadır. `valid`: penceredeki gerçek (doldurulmamış) pikseller.

Hedef türleri (kind)
  patch              : tek 224 patch'in p(sahte)'si                         (ResNet)
  window_max_patch   : penceredeki uygulama ızgarası patch'lerinin maks skoru  (ResNet, karşılaştırma modu)
  window_image_score : penceredeki geçerli piksellerin top-%1 ortalaması     (U-Net; app.image_score tanımı)
  region_mean        : tespit bölgesi (bağlı bileşen) ortalama olasılığı      (U-Net)
  gt_bbox_mean       : GT kutusu içi ortalama olasılık                       (U-Net)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import numpy as np
import cv2


@dataclass
class TargetSpec:
    model: str
    kind: str
    prep: str                         # unet | resnet
    y: int
    x: int
    h: int
    w: int
    label: str = ""
    source: str = "karar"             # karar | tespit | gt | ortak
    confidence: Optional[float] = None
    region: Optional[np.ndarray] = None   # (h,w) bool — region_mean / gt_bbox_mean
    valid: Optional[np.ndarray] = None    # (h,w) bool — doldurulmamış pikseller
    patches: Optional[list] = None        # window_max_patch: pencere içi (dy, dx) patch konumları
    scale: float = 1.0                    # prep="unet" ölçek katsayısı (orijinal → hazırlanmış)
    notes: list = field(default_factory=list)

    @property
    def window(self):
        return (self.y, self.x, self.h, self.w)

    def key(self):
        rh = "" if self.region is None else hashlib.sha1(np.packbits(self.region).tobytes()).hexdigest()[:12]
        return f"{self.model}|{self.kind}|{self.prep}|{self.y},{self.x},{self.h},{self.w}|{rh}"

    def to_dict(self):
        return dict(model=self.model, kind=self.kind, prep=self.prep, window=[self.y, self.x, self.h, self.w], label=self.label,
                    source=self.source, confidence=self.confidence, scale=self.scale,
                    region_px=None if self.region is None else int(self.region.sum()), notes=list(self.notes))


# ------------------------------------------------------------------ geometri
def unet_geometry(H0, W0, win=512, max_side=4096):
    """app._unet_prepare ile aynı formül: (ölçek, (h, w) ölçeklenmiş, (Hp, Wp) doldurulmuş)."""
    s = 1.0
    h, w = H0, W0
    if max(H0, W0) > max_side:
        s = max_side / max(H0, W0)
        h, w = max(1, int(round(H0 * s))), max(1, int(round(W0 * s)))
    return s, (h, w), (max(h, win), max(w, win))


def resnet_geometry(H0, W0, patch=224):
    return 1.0, (H0, W0), (max(H0, patch), max(W0, patch))


def window_at(cy, cx, Hp, Wp, wh, ww):
    wh, ww = min(wh, Hp), min(ww, Wp)
    y = int(np.clip(int(round(cy - wh / 2)), 0, Hp - wh))
    x = int(np.clip(int(round(cx - ww / 2)), 0, Wp - ww))
    return y, x, wh, ww


def _valid(y, x, h, w, hv, wv):
    v = np.zeros((h, w), bool)
    v[:max(0, min(h, hv - y)), :max(0, min(w, wv - x))] = True
    return v


def _to_prep_mask(mask_orig, scale, hw, HpWp):
    """Orijinal koordinattaki ikili maske → hazırlanmış koordinat (en yakın komşu ölçek + sıfır dolgu)."""
    h, w = hw
    m = mask_orig.astype(np.uint8)
    if scale != 1.0:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros(HpWp, bool)
    out[:h, :w] = m > 0
    return out


def _bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resnet_grid(H0, W0, deps):
    """Uygulamanın ResNet kayan pencere ızgarası (resnet-hazırlanmış koordinatlar)."""
    _, _, (Hr, Wr) = resnet_geometry(H0, W0, deps.resnet_patch)
    return deps.windows(Hr, Wr, deps.resnet_patch, deps.resnet_stride)


def _patches_in(grid, y, x, h, w, P):
    return [(gy - y, gx - x) for (gy, gx) in grid if gy >= y and gx >= x and gy + P <= y + h and gx + P <= x + w]


# ------------------------------------------------------------------ hedef üreticiler
def _unet_window_target(name, cy0, cx0, H0, W0, deps, kind, source, label, conf=None, region_orig=None):
    s, (h, w), (Hp, Wp) = unet_geometry(H0, W0, deps.unet_win, deps.unet_max_side)
    y, x, wh, ww = window_at(cy0 * s, cx0 * s, Hp, Wp, deps.unet_win, deps.unet_win)
    t = TargetSpec(name, kind, "unet", y, x, wh, ww, label=label, source=source, confidence=conf, scale=s,
                   valid=_valid(y, x, wh, ww, h, w))
    if region_orig is not None:
        rp = _to_prep_mask(region_orig, s, (h, w), (Hp, Wp))
        t.region = rp[y:y + wh, x:x + ww].copy()
        if rp.sum() > t.region.sum():
            t.notes.append(f"bölgenin %{100 * (1 - t.region.sum() / max(1, rp.sum())):.0f}'i pencere dışında kaldı")
        if not t.region.any():
            t.notes.append("bölge pencerede boş")
    if s != 1.0:
        t.notes.append(f"görüntü U-Net için {s:.3f}× küçültüldü (uzun kenar > {deps.unet_max_side})")
    return t


def _resnet_window_target(name, cy, cx, H0, W0, deps, source, label, conf=None):
    _, _, (Hr, Wr) = resnet_geometry(H0, W0, deps.resnet_patch)
    y, x, wh, ww = window_at(cy, cx, Hr, Wr, deps.unet_win, deps.unet_win)
    t = TargetSpec(name, "window_max_patch", "resnet", y, x, wh, ww, label=label, source=source, confidence=conf,
                   valid=_valid(y, x, wh, ww, H0, W0))
    t.patches = _patches_in(resnet_grid(H0, W0, deps), y, x, wh, ww, deps.resnet_patch)
    return t


def top1_centroid(prob):
    u = prob.ravel()
    k = max(1, int(0.01 * u.size))
    thr = np.partition(u, -k)[-k]
    ys, xs = np.where(prob >= thr)
    wts = prob[ys, xs] + 1e-6
    return float((ys * wts).sum() / wts.sum()), float((xs * wts).sum() / wts.sum())


def decision_targets(preds: dict, specs: dict, shape, deps):
    """Model başına kararı belirleyen hedef: ResNet → maks patch; U-Net → top-%1 ağırlık merkezli pencerenin skoru."""
    H0, W0 = shape[:2]
    out = []
    for name, (m, info) in preds.items():
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            s, y, x = max(info["scores"], key=lambda t: t[0])
            P = deps.resnet_patch
            out.append(TargetSpec(name, "patch", "resnet", int(y), int(x), P, P, source="karar", confidence=float(s),
                                  label=f"maks patch p={s:.3f} @ ({x},{y})", valid=_valid(int(y), int(x), P, P, H0, W0)))
        else:
            cy, cx = top1_centroid(m)
            sc = deps.image_score(m)
            out.append(_unet_window_target(name, cy, cx, H0, W0, deps, "window_image_score", "karar",
                                           f"görüntü skoru {sc:.3f} · top-%1 merkezli pencere", conf=sc))
    return out


def unet_regions(prob, tau, deps, min_px=9):
    binm, _ = deps.postprocess(prob, tau, 0, False)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm.astype(np.uint8), connectivity=8)
    regs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        msk = lab == i
        x0, y0, bw, bh = stats[i, :4]
        regs.append(dict(mask=msk, conf=float(prob[msk].mean()), area=int(msk.sum()), bbox=(int(x0), int(y0), int(x0 + bw), int(y0 + bh))))
    regs.sort(key=lambda r: (-r["conf"], -r["area"]))
    return regs


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detection_targets(preds: dict, specs: dict, shape, deps, k=3):
    """'Nesne tespiti' eşdeğeri: U-Net bağlı bileşenleri (güven = ortalama olasılık) ve ResNet top-k patch'leri."""
    H0, W0 = shape[:2]
    out = []
    for name, (m, info) in preds.items():
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            P = deps.resnet_patch
            picked = []
            for s, y, x in sorted(info["scores"], key=lambda t: -t[0]):
                b = (x, y, x + P, y + P)
                if all(_iou(b, q) < 0.25 for q in picked):
                    picked.append(b)
                    out.append(TargetSpec(name, "patch", "resnet", int(y), int(x), P, P, source="tespit", confidence=float(s),
                                          label=f"tespit {len(picked)} · patch p={s:.3f} @ ({x},{y})",
                                          valid=_valid(int(y), int(x), P, P, H0, W0)))
                if len(picked) >= k:
                    break
        else:
            tau = deps.default_tau(name, sp.train_gt or "bbox")
            for j, r in enumerate(unet_regions(m, tau, deps)[:k], 1):
                x0, y0, x1, y1 = r["bbox"]
                t = _unet_window_target(name, (y0 + y1) / 2, (x0 + x1) / 2, H0, W0, deps, "region_mean", "tespit",
                                        f"tespit {j} · bölge güveni {r['conf']:.3f} · kutu ({x0},{y0},{x1},{y1}) · τ={tau:.2f}",
                                        conf=r["conf"], region_orig=r["mask"])
                out.append(t)
            if not any(t.model == name for t in out):
                pass  # eşik üstü bölge yok → bu model için tespit hedefi üretilmez (arayüzde not edilir)
    return out


def gt_targets(models, specs, shape, gt_mask, deps):
    """GT kutusu hedefi: U-Net → kutu içi ortalama; ResNet → kutu merkezli penceredeki maks patch."""
    H0, W0 = shape[:2]
    if gt_mask is None:
        return []
    g = gt_mask > 127 if gt_mask.dtype == np.uint8 else gt_mask.astype(bool)
    bb = _bbox(g)
    if bb is None:
        return []
    cy, cx = (bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2
    out = []
    for name in models:
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            out.append(_resnet_window_target(name, cy, cx, H0, W0, deps, "gt", "GT kutusu merkezli pencere · maks patch"))
        else:
            box = np.zeros((H0, W0), bool); box[bb[1]:bb[3], bb[0]:bb[2]] = True
            out.append(_unet_window_target(name, cy, cx, H0, W0, deps, "gt_bbox_mean", "gt", f"GT kutusu içi ortalama · {bb}",
                                           region_orig=box))
    return out


def common_center(preds, specs, shape, gt_mask, deps):
    if gt_mask is not None:
        g = gt_mask > 127 if gt_mask.dtype == np.uint8 else gt_mask.astype(bool)
        bb = _bbox(g)
        if bb is not None:
            return (bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2, "GT kutusu merkezi"
    for name, (m, info) in preds.items():
        sp = specs.get(name)
        if sp is not None and sp.family == "unet":
            cy, cx = top1_centroid(m)
            return cy, cx, f"{sp.label} top-%1 merkezi"
    for name, (m, info) in preds.items():
        s, y, x = max(info["scores"], key=lambda t: t[0])
        return y + deps.resnet_patch / 2, x + deps.resnet_patch / 2, "ResNet maks patch merkezi"
    return shape[0] / 2, shape[1] / 2, "görüntü merkezi"


def common_window_targets(models, preds, specs, shape, gt_mask, deps):
    """Karşılaştırma modu: tüm modeller AYNI fiziksel pencereyi açıklar (ResNet: pencere içi maks patch)."""
    H0, W0 = shape[:2]
    cy, cx, why = common_center(preds, specs, shape, gt_mask, deps)
    out = []
    for name in models:
        sp = specs.get(name)
        if sp is None or sp.status != "OK":
            continue
        if sp.family == "resnet":
            t = _resnet_window_target(name, cy, cx, H0, W0, deps, "ortak", f"ortak pencere ({why}) · maks patch")
        else:
            t = _unet_window_target(name, cy, cx, H0, W0, deps, "window_image_score", "ortak",
                                    f"ortak pencere ({why}) · pencere görüntü skoru")
        out.append(t)
    return out


def window_mask(mask_orig, t: TargetSpec, deps):
    """Orijinal koordinattaki ikili maskeyi (GT / işaretçi) hedefin hazırlanmış penceresine taşı."""
    if mask_orig is None:
        return None
    m = mask_orig > 127 if mask_orig.dtype == np.uint8 else mask_orig.astype(bool)
    H0, W0 = m.shape[:2]
    if t.prep == "unet":
        s, hw, HpWp = unet_geometry(H0, W0, deps.unet_win, deps.unet_max_side)
    else:
        s, hw, HpWp = resnet_geometry(H0, W0, deps.resnet_patch)
    full = _to_prep_mask(m, s, hw, HpWp)
    return full[t.y:t.y + t.h, t.x:t.x + t.w]
