"""
xai_filters.py — XAI · Katman Filtreleri (Filter Visualization / Convolutional Kernels)

SHAP sekmesinden (app/xai) BAĞIMSIZ ikinci açıklanabilirlik özelliği. Eğitilmiş modellerin KAYITLI ağırlıklarından ilk
konvolüsyon katmanlarının gerçek kernel matrislerini çıkarır, bunları modelin kendi ön işlemesinden geçmiş görüntüye
uygular ve oluşan aktivasyon (öznitelik) haritalarını gösterir.

İlkeler
  * Hiçbir filtre eğitilmez ya da elle üretilmez: kernel = checkpoint dosyasındaki tensör (state_dict) ya da Ultralytics
    checkpoint'indeki model nesnesinin parametresi. Konvolüsyon katmanı olmayan modelde filtre üretilmez (NoConvError).
  * Ağırlıklar ve mimari değiştirilmez (eval + no_grad; analiz öncesi/sonrası SHA-256 karşılaştırılır).
  * Ara çıktılar modelin KENDİ forward geçişinden ileri kancalarla (forward hook) alınır. Aynı katmanın gerçek girdisine
    checkpoint kernel'i F.conv2d ile (aynı stride / padding / dilation / groups / bias) bağımsız olarak da uygulanır ve
    modelin çıktısıyla karşılaştırılır (max|Δ|) → bağımsız hesaplamanın modelin konvolüsyonuyla tutarlılığı her
    katmanda doğrulanır.
  * "İlk katmanlar" sabit sayı değildir: forward yürütme sırasındaki Conv2d katmanları, çıkış çözünürlüğü girdinin en az
    1/4'ü (birikimli adım ≤ 4) olduğu sürece alınır (ilk conv her zaman dahil; en fazla max_layers). Sınır aşılınca forward
    durdurulur — derin katmanlar hiç hesaplanmaz (bellek/süre kontrolü).
  * Her conv için modelin uyguladığı sonraki işlemler (BatchNorm → ReLU/SiLU) yalnızca gerçekten zincirleniyorsa
    (girdi = önceki aşamanın çıktısı, birebir) ayrı aşama olarak kaydedilir; artık (residual) toplama gibi araya giren
    işlemler zinciri kapatır.

Desteklenen modeller (mimari checkpoint'ten otomatik tespit edilir)
  smp.Unet / smp.DeepLabV3Plus (ResNet kodlayıcı, 3 ya da 5 kanal; 2 sınıflı birleşik U-Net dahil) · torchvision ResNet ·
  Ultralytics YOLO (DetectionModel; tam checkpoint ultralytics ile, yoksa export_yolo_stem ile çıkarılmış ilk-katman
  dosyası — yalnız gerçek ağırlıklar, eşdeğerlik testle doğrulanır).
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

NORM_TYPES = (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.LeakyReLU, nn.ELU, nn.Mish, nn.Hardswish, nn.PReLU,
             nn.Sigmoid, nn.Tanh, nn.Hardsigmoid)

STAGES = OrderedDict([("conv", "Conv çıktısı (ham, BN öncesi)"),
                      ("norm", "Normalizasyon (BatchNorm) sonrası"),
                      ("act", "Aktivasyon sonrası (sonraki katmana giden)")])
NORMS = ["Harita başına min-max", "Harita başına %1–%99 kırpma", "Katman genelinde simetrik (0 = gri)",
         "Katman genelinde min-max (ölçek korunur)"]
REGION_MODES = ["Natif çözünürlük kırpma (sahte bölge / merkez)", "Tam görüntü (uzun kenar ≤ N, küçültme)"]
DEVICES = ["otomatik", "GPU (cuda)", "CPU"]
SOURCES = ["Veri setinden örnek", "Girdi sekmesindeki görüntü", "Yeni yükleme"]
MAX_REGION = 1536
PER_PAGE = 16

_GT_NAMES = {"bbox": "bbox-GT", "pixel_eps8_morph": "piksel-GT", "glyph_v2": "glif-GT"}
_TAGS = {"geminionly": "yalnız Gemini", "mr": "işaretçi-onarımlı", "nm": "işaretçisiz"}
_FIXED = {"resnet18_ft_best": "ResNet-18 · tam ince ayar (PHASE1)", "resnet18_probe_best": "ResNet-18 · linear probe (PHASE1)",
          "resnet18_scratch_best": "ResNet-18 · sıfırdan (PHASE1)", "h4_resnet18": "ResNet-18 · H4 global-iz sondası",
          "h4_resnet18_poscontrol": "ResNet-18 · H4 global-iz sondası (poz. kontrol)",
          "unet_rgb_best": "U-Net · RGB (PHASE1)", "unet_rgbfreq_best": "U-Net · RGB+FFT+Laplacian, 5 kanal (PHASE1)"}


class NoConvError(RuntimeError):
    """Modelde Conv2d katmanı yok → filtre görselleştirmesi üretilmez (sahte filtre üretilmez)."""


class _Stop(Exception):
    """İlk katmanlar toplandı; forward'ı derin katmanlara ilerlemeden durdur."""


# ====================================================================== model kaydı
@dataclass
class ModelSpec:
    key: str
    label: str
    path: str
    kind: str      # state_dict | yolo | yolo_stem
    group: str


def friendly_label(stem: str) -> str:
    if stem in _FIXED:
        return _FIXED[stem]
    for pre, fam in (("unetjoint_", "U-Net + kutu yardımcı başı"), ("unet_", "U-Net"), ("dlv3p_", "DeepLabV3+")):
        if stem.startswith(pre):
            rest = stem[len(pre):]
            m = re.match(r"(?P<gt>bbox|pixel_eps8_morph|glyph_v2)(?P<tags>(?:_[a-z]+)*)_s(?P<seed>\d+)(?P<ep>_e\d+)?$", rest)
            if not m:
                return f"{fam} · {rest}"
            tags = [_TAGS.get(t, t) for t in m.group("tags").split("_") if t]
            extra = (" · " + ", ".join(tags)) if tags else ""
            ep = f", {m.group('ep')[2:]} epoch" if m.group("ep") else ""
            return f"{fam} · {_GT_NAMES[m.group('gt')]}{extra} (s{m.group('seed')}{ep})"
    return stem


def discover_models(models_dir, yolo_runs_dir=None) -> list:
    """Kayıtlı ağırlık dosyalarını bulur. Etiket dosya adından, MİMARİ ise yüklemede checkpoint içeriğinden belirlenir."""
    models_dir = Path(models_dir)
    out = []
    for p in sorted(glob.glob(str(models_dir / "phase2" / "*.pt"))) + sorted(glob.glob(str(models_dir / "*.pt"))):
        stem = Path(p).stem
        grp = "PHASE2" if Path(p).parent.name == "phase2" else "PHASE1"
        out.append(ModelSpec(f"{grp}/{stem}", f"{friendly_label(stem)} [{grp}]", p, "state_dict", grp))
    stems = {}
    for p in sorted(glob.glob(str(models_dir / "phase3_yolo" / "*_stem.pt"))):
        run = Path(p).stem[:-5]
        stems[run] = p
    seen = set()
    import importlib.util
    has_ultra = importlib.util.find_spec("ultralytics") is not None
    if has_ultra and yolo_runs_dir and Path(yolo_runs_dir).exists():
        for p in sorted(glob.glob(str(Path(yolo_runs_dir) / "*" / "weights" / "best.pt"))):
            run = Path(p).parent.parent.name
            done = (Path(p).parent.parent / "summary.json").exists()
            tag = "" if done else " · eğitim sürüyor (son best.pt)"
            out.append(ModelSpec(f"YOLO/{run}", f"YOLO11 · {run}{tag} [PHASE3]", p, "yolo", "PHASE3"))
            seen.add(run)
    for run, p in stems.items():
        if run in seen:  # tam checkpoint (6 ilk katman) varken çıkarılmış kopyayı yineleme
            continue
        out.append(ModelSpec(f"YOLOSTEM/{run}", f"YOLO11 · {run} · ilk katmanlar (best.pt'den çıkarılmış) [PHASE3]", p,
                             "yolo_stem", "PHASE3"))
    return out


@dataclass
class LoadedModel:
    spec: ModelSpec
    module: nn.Module
    arch: str
    family: str
    in_ch: int
    preprocess: str              # imagenet | imagenet+freq | yolo
    ckpt_conv: dict              # conv modül adı → (weight fp32, bias|None) — checkpoint'ten bağımsız kopya
    sha: str
    n_params: int
    note: str = ""
    channel_names: list = field(default_factory=list)


def state_sha(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _n_blocks(sd, prefix):
    idx = set()
    for k in sd:
        if k.startswith(prefix):
            idx.add(k[len(prefix):].split(".")[0])
    return len(idx)


def _resnet_name(sd, pre):
    n = (_n_blocks(sd, pre + "layer1."), _n_blocks(sd, pre + "layer2."), _n_blocks(sd, pre + "layer3."), _n_blocks(sd, pre + "layer4."))
    bottleneck = any(k.startswith(pre + "layer1.0.conv3") for k in sd)
    table = {(2, 2, 2, 2): "resnet18", (3, 4, 6, 3): "resnet50" if bottleneck else "resnet34"}
    if n not in table:
        raise RuntimeError(f"ResNet blok sayıları tanınmadı: {n}")
    return table[n]


class _YoloConv(nn.Module):
    """Ultralytics Conv bloğunun birebir karşılığı (conv → bn → act); yalnız export_yolo_stem dosyaları için."""

    def __init__(self, c1, c2, k, s, p, d, g, eps, act):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=eps)
        self.act = {"SiLU": nn.SiLU(), "ReLU": nn.ReLU(), "Identity": nn.Identity()}[act]

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _YoloStem(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Sequential(*[_YoloConv(**L["cfg"]) for L in layers])

    def forward(self, x):
        return self.model(x)


def _load_state_dict_model(spec: ModelSpec):
    import segmentation_models_pytorch as smp
    import torchvision
    sd = torch.load(spec.path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise RuntimeError(f"Beklenmeyen checkpoint biçimi: {type(sd).__name__}")
    note = ""
    if any(k.startswith("net.") for k in sd):  # PHASE1 UNetWrapper
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
        note = "PHASE1 sarmalayıcı (net.) öneki kaldırıldı. "
    if not any(torch.is_tensor(v) and v.ndim == 4 for v in sd.values()):
        raise NoConvError(f"{spec.label}: checkpoint'te 4-boyutlu (Conv2d) ağırlık yok — filtre görselleştirmesi üretilmez.")
    if "encoder.conv1.weight" in sd:
        enc = _resnet_name(sd, "encoder.")
        in_ch = int(sd["encoder.conv1.weight"].shape[1])
        classes = int(sd["segmentation_head.0.weight"].shape[0])
        if any(k.startswith("decoder.aspp") for k in sd):
            m = smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=classes)
            fam, arch = "dlv3p", f"smp.DeepLabV3Plus(encoder={enc}, in_channels={in_ch}, classes={classes})"
        elif any(k.startswith("decoder.blocks") for k in sd):
            m = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=in_ch, classes=classes)
            fam, arch = "unet", f"smp.Unet(encoder={enc}, in_channels={in_ch}, classes={classes})"
        else:
            raise RuntimeError("smp kod çözücüsü tanınmadı (Unet / DeepLabV3Plus değil).")
        prep = {3: "imagenet", 5: "imagenet+freq"}.get(in_ch)
        if prep is None:
            raise RuntimeError(f"{in_ch} giriş kanallı ön işleme tanımlı değil.")
    elif "conv1.weight" in sd and "fc.weight" in sd:
        name = _resnet_name(sd, "")
        m = getattr(torchvision.models, name)(weights=None)
        m.fc = nn.Linear(int(sd["fc.weight"].shape[1]), int(sd["fc.weight"].shape[0]))
        in_ch = int(sd["conv1.weight"].shape[1])
        fam, arch, prep = "resnet", f"torchvision.{name} (fc → {int(sd['fc.weight'].shape[0])})", "imagenet"
    else:
        raise RuntimeError("Mimari checkpoint anahtarlarından tanınmadı; forward kurulamadığı için katman çıktısı alınamaz.")
    m.load_state_dict(sd, strict=True)
    ckpt_conv = {}
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            b = sd.get(n + ".bias")
            ckpt_conv[n] = (sd[n + ".weight"].detach().float().clone(), None if b is None else b.detach().float().clone())
    names = ["R", "G", "B", "FFT-HP (D0=20)", "Laplacian"][:in_ch]
    return m, fam, arch, in_ch, prep, ckpt_conv, note, names


def _load_yolo(spec: ModelSpec):
    try:
        import ultralytics  # noqa: F401  (checkpoint pickle'ı ultralytics sınıflarını içerir)
    except Exception as e:
        raise RuntimeError(f"Tam YOLO checkpoint'i için ultralytics gerekli ({e}). export_yolo_stem ile ilk katmanları "
                           f"çıkarılmış dosyayı kullanın.")
    ck = torch.load(spec.path, map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck.get("model")
    if m is None:
        raise RuntimeError("YOLO checkpoint'inde model nesnesi yok.")
    dtype = str(next(m.parameters()).dtype).replace("torch.", "")
    ckpt_conv = {n: (mod.weight.detach().float().clone(), None if mod.bias is None else mod.bias.detach().float().clone())
                 for n, mod in m.named_modules() if isinstance(mod, nn.Conv2d)}
    m = m.float()  # fp16 → fp32 tür dönüşümü (değerler birebir korunur; conv+BN birleştirme YAPILMAZ)
    ver = ck.get("version", "?")
    arch = f"Ultralytics {type(m).__name__} ({getattr(m, 'yaml', {}).get('yaml_file', 'yolo11')}, ultralytics {ver}, kayıt {dtype})"
    return m, "yolo", arch, 3, "yolo", ckpt_conv, "Conv+BN birleştirilmedi (fuse yok); ham Conv/BN/SiLU katmanları. ", ["R", "G", "B"]


def _load_yolo_stem(spec: ModelSpec):
    d = torch.load(spec.path, map_location="cpu", weights_only=True)
    if d.get("format") != "ai515-yolo-stem-v1":
        raise RuntimeError("Tanınmayan YOLO ilk-katman dosyası.")
    m = _YoloStem(d["layers"])
    sd = {}
    for i, L in enumerate(d["layers"]):
        for k, v in L["state"].items():
            sd[f"model.{i}.{k}"] = v
    m.load_state_dict(sd, strict=True)
    ckpt_conv = {f"model.{i}.conv": (L["state"]["conv.weight"].float().clone(), None) for i, L in enumerate(d["layers"])}
    arch = f"Ultralytics YOLO ilk {len(d['layers'])} Conv bloğu (kaynak: {d.get('source_name', '?')}, sha256 {d.get('source_sha256', '')[:12]}…)"
    note = "best.pt'den export_yolo_stem ile çıkarılmış gerçek ağırlıklar (ultralytics gerektirmez). "
    return m, "yolo", arch, 3, "yolo", ckpt_conv, note, ["R", "G", "B"]


_MODEL_CACHE: "OrderedDict[str, LoadedModel]" = OrderedDict()
_LOCK = threading.Lock()
_PLOT_LOCK = threading.RLock()  # pyplot/mathtext iş parçacığı güvenli değil; Gradio olayları eşzamanlı çalışabilir


def _plot_locked(fn):
    import functools

    @functools.wraps(fn)
    def wrap(*a, **k):
        with _PLOT_LOCK:
            return fn(*a, **k)
    return wrap


def load_model(spec: ModelSpec, cache_size: int = 3) -> LoadedModel:
    with _LOCK:
        if spec.key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(spec.key)
            return _MODEL_CACHE[spec.key]
        loader = {"state_dict": _load_state_dict_model, "yolo": _load_yolo, "yolo_stem": _load_yolo_stem}[spec.kind]
        m, fam, arch, in_ch, prep, ckpt_conv, note, names = loader(spec)
        m = m.float().eval()
        for p in m.parameters():
            p.requires_grad_(False)
        if not ckpt_conv:
            raise NoConvError(f"{spec.label}: modelde Conv2d katmanı yok — filtre görselleştirmesi üretilmez.")
        lm = LoadedModel(spec, m, arch, fam, in_ch, prep, ckpt_conv, state_sha(m), sum(p.numel() for p in m.parameters()),
                         note, names)
        _MODEL_CACHE[spec.key] = lm
        while len(_MODEL_CACHE) > cache_size:
            _MODEL_CACHE.popitem(last=False)
        return lm


def export_yolo_stem(best_pt, out_path, n_layers: int = 2):
    """Ultralytics best.pt'den ilk n düz Conv bloğunun (conv+bn+act) GERÇEK ağırlıklarını taşınabilir dosyaya yazar
    (Space'te ultralytics olmadan kullanmak için). Ağırlıklar değiştirilmez; yalnız fp16 → fp32 tür dönüşümü."""
    import ultralytics  # noqa: F401
    ck = torch.load(best_pt, map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck.get("model")).float()
    layers = []
    for i in range(n_layers):
        blk = m.model[i]
        if type(blk).__name__ != "Conv":
            break
        c = blk.conv
        layers.append(dict(cfg=dict(c1=c.in_channels, c2=c.out_channels, k=c.kernel_size, s=c.stride, p=c.padding, d=c.dilation,
                                    g=c.groups, eps=float(blk.bn.eps), act=type(blk.act).__name__),
                           state={k: v.detach().clone() for k, v in blk.state_dict().items()}))
    sha = hashlib.sha256(Path(best_pt).read_bytes()).hexdigest()
    run = Path(best_pt).parent.parent.name
    torch.save(dict(format="ai515-yolo-stem-v1", source_name=f"{run}/weights/best.pt", source_sha256=sha, layers=layers), out_path)
    return out_path


# ====================================================================== bölge + ön işleme
def freq_channels_local(rgb_u8: np.ndarray, d0: float = 20.0) -> np.ndarray:
    """app.project_freq_channels ile birebir aynı tanım (FFT Gaussian yüksek-geçiren D0=20 + 3×3 Laplacian, örnek başına
    standartlaştırma). Uygulama kendi fonksiyonunu verir; bu yalnız bağımsız kullanım (rapor/test) içindir."""
    g = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32) / 255.0
    gu = (g * 255).astype(np.uint8).astype(np.float32)
    H, W = gu.shape
    fy = np.fft.fftfreq(H)[:, None] * H
    fx = np.fft.fftfreq(W)[None, :] * W
    D = np.sqrt(fy ** 2 + fx ** 2)
    Hf = 1.0 - np.exp(-(D ** 2) / (2.0 * d0 ** 2))
    hp = np.real(np.fft.ifft2(np.fft.fft2(gu) * Hf)) / 255.0
    lap = cv2.Laplacian(cv2.copyMakeBorder(g, 1, 1, 1, 1, cv2.BORDER_REPLICATE), cv2.CV_32F, ksize=1)[1:-1, 1:-1]
    out = [(c - c.mean()) / (c.std() + 1e-6) for c in (hp, lap)]
    return np.stack(out, 0).astype(np.float32)


def _to_rgb(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return img[..., :3].copy()
    return img


def select_region(img, mask, mode: str, size: int):
    """Analiz bölgesi. Natif kırpma adli izleri korur (modeller natif çözünürlükte eğitildi); tam görüntü küçültülür."""
    img = _to_rgb(img)
    H, W = img.shape[:2]
    size = int(max(32, min(MAX_REGION, size)))
    m = None if mask is None else (mask if mask.ndim == 2 else mask[..., 0])
    if m is not None and m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if mode == REGION_MODES[1]:
        s = min(1.0, size / max(H, W))
        if s < 1.0:
            nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
            reg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            mr = None if m is None else cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            reg, mr = img, m
        desc = f"tam görüntü {W}×{H} → {reg.shape[1]}×{reg.shape[0]} px (ölçek {s:.3f}, INTER_AREA)"
        return reg.copy(), (None if mr is None else mr.copy()), desc, dict(x=0, y=0, w=W, h=H, scale=s)
    if m is not None and (m > 127).any():
        ys, xs = np.nonzero(m > 127)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        how = "sahte bölge (GT kutusu) merkezli"
    else:
        cy, cx = H // 2, W // 2
        how = "görüntü merkezi"
    hh, ww = min(size, H), min(size, W)
    y0 = int(np.clip(cy - hh // 2, 0, H - hh)); x0 = int(np.clip(cx - ww // 2, 0, W - ww))
    reg = img[y0:y0 + hh, x0:x0 + ww].copy()
    mr = None if m is None else m[y0:y0 + hh, x0:x0 + ww].copy()
    desc = f"natif çözünürlük kırpma {ww}×{hh} px @ (x={x0}, y={y0}) — {how}; kaynak {W}×{H} px"
    return reg, mr, desc, dict(x=x0, y=y0, w=ww, h=hh, scale=1.0)


def preprocess(region_u8, how: str, freq_fn: Optional[Callable] = None, multiple: int = 32):
    """Modelin eğitimdeki ön işlemesi. Sağ/alt dolgu 32'nin katına (smp kodlayıcı şartı / YOLO adımı) — haritalar sonra
    dolgusuz alana kırpılır. Dönüş: (1,C,Hp,Wp) float32 tensör, açıklama."""
    h, w = region_u8.shape[:2]
    Hp, Wp = int(math.ceil(h / multiple) * multiple), int(math.ceil(w / multiple) * multiple)
    if how == "yolo":
        pad = cv2.copyMakeBorder(region_u8, 0, Hp - h, 0, Wp - w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = (pad.astype(np.float32) / 255.0).transpose(2, 0, 1)
        desc = "Ultralytics: RGB / 255 (ImageNet normalizasyonu yok), sağ/alt 114 gri dolgu (letterbox), fp32"
        return torch.from_numpy(np.ascontiguousarray(x))[None], desc
    pad = cv2.copyMakeBorder(region_u8, 0, Hp - h, 0, Wp - w, cv2.BORDER_REPLICATE)
    x = ((pad.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(x))[None]
    desc = "RGB / 255 → ImageNet ortalama/std normalizasyonu (eğitimdeki tanım), sağ/alt replicate dolgu, fp32"
    if how == "imagenet+freq":
        fc = torch.from_numpy((freq_fn or freq_channels_local)(pad))[None]
        mu = fc.mean(dim=(2, 3), keepdim=True); sd = fc.std(dim=(2, 3), keepdim=True) + 1e-6
        t = torch.cat([t, (fc - mu) / sd], 1)
        desc += " + 2 frekans kanalı (FFT-HP D0=20, Laplacian) pencere başına standartlaştırma (_predict_unet ile aynı)"
    return t, desc


# ====================================================================== çekirdek analiz
@dataclass
class LayerRecord:
    idx: int
    name: str
    conv: dict
    weight: np.ndarray                 # (out, in/groups, kh, kw) — checkpoint'ten
    bias: Optional[np.ndarray]
    out_stride: float
    out_shape: tuple
    stages: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict)
    stage_ops: dict = field(default_factory=dict)
    norm_params: Optional[dict] = None
    verify: dict = field(default_factory=dict)
    _stats: dict = field(default_factory=dict)

    @property
    def n_filters(self):
        return int(self.weight.shape[0])

    def stage_stats(self, st):
        if st not in self._stats:
            a = self.stages[st]
            self._stats[st] = (float(a.min()), float(a.max()), float(np.abs(a).max()))
        return self._stats[st]


@dataclass
class Analysis:
    token: str
    model_key: str
    model_label: str
    arch: str
    family: str
    preprocess: str
    prep_desc: str
    region: np.ndarray
    mask: Optional[np.ndarray]
    region_desc: str
    x: np.ndarray                     # modele verilen (dolgulu) girdi tensörü (C,Hp,Wp)
    valid_hw: tuple
    layers: list
    input_equal: bool
    device: str
    sec: float
    peak_gpu_mb: float
    stored_mb: float
    store_dtype: str
    sha_before: str
    sha_after: str
    channel_names: list
    note: str = ""


def pick_device(choice: str = "otomatik", min_free_gb: float = 1.5) -> str:
    if choice == "CPU" or not torch.cuda.is_available():
        return "cpu"
    if str(choice).startswith("GPU"):
        return "cuda"
    try:
        free, _ = torch.cuda.mem_get_info()
        return "cuda" if free / 1e9 >= min_free_gb else "cpu"
    except Exception:
        return "cuda"


@torch.no_grad()
def analyze(lm: LoadedModel, x: torch.Tensor, valid_hw, device: str = "cpu", max_layers: int = 6, max_stride: float = 4.0,
            store_dtype=np.float32):
    """Modelin kendi forward'ını çalıştırır; ilk konvolüsyon katmanlarının çıktılarını kancalarla toplar, her conv'u
    checkpoint kernel'iyle bağımsız F.conv2d ile yeniden hesaplayıp doğrular. Dönüş: (katmanlar, girdi_eşit, süre, tepe GPU MB)."""
    dev = torch.device(device)
    mod = lm.module
    h, w = valid_hw
    H_in = x.shape[-2]
    layers: list = []
    st = dict(open=None, prev=None, pending=None, n=0, input_equal=None)
    x_dev = x.to(dev, torch.float32)

    def keep(t, ratio):
        hh, ww = int(math.ceil(h / ratio)), int(math.ceil(w / ratio))
        # copy=True ŞART: CPU'da .cpu()/.numpy() belleği paylaşır; sonraki yerinde (inplace) ReLU BN çıktısını ezerdi.
        return t[0, :, :hh, :ww].detach().float().cpu().numpy().astype(store_dtype, copy=True)

    def close():
        st["open"] = None
        st["prev"] = None

    def conv_hook(name):
        def fn(m, inp, out):
            if not (torch.is_tensor(out) and out.dim() == 4):
                return
            ratio = H_in / out.shape[-2]
            if st["n"] >= max_layers or (st["n"] > 0 and ratio > max_stride + 1e-6):
                raise _Stop()
            xin = inp[0]
            Wt, b = lm.ckpt_conv[name]
            Wt = Wt.to(xin.device)
            b = None if b is None else b.to(xin.device)
            xi = xin.float()
            pad = m.padding
            if m.padding_mode != "zeros":
                xi = F.pad(xi, m._reversed_padding_repeated_twice, mode=m.padding_mode)
                pad = 0
            y = F.conv2d(xi, Wt, b, m.stride, pad, m.dilation, m.groups)
            o = out.float()
            diff = float((y - o).abs().max())
            scale = float(o.abs().max()) + 1e-12
            if st["n"] == 0:
                st["input_equal"] = bool(xin.shape == x_dev.shape and torch.equal(xin.float(), x_dev))
            rec = LayerRecord(
                idx=st["n"], name=name,
                conv=dict(type=type(m).__name__, in_ch=m.in_channels, out_ch=m.out_channels, kernel=tuple(m.kernel_size),
                          stride=tuple(m.stride), padding=m.padding if isinstance(m.padding, str) else tuple(m.padding),
                          dilation=tuple(m.dilation), groups=m.groups, bias=m.bias is not None, padding_mode=m.padding_mode),
                weight=Wt.detach().cpu().numpy().copy(), bias=None if b is None else b.cpu().numpy().copy(),
                out_stride=float(ratio), out_shape=(int(out.shape[1]), int(math.ceil(h / ratio)), int(math.ceil(w / ratio))))
            rec.verify = dict(max_abs=diff, rel=diff / scale, out_absmax=scale,
                              weight_is_checkpoint=bool(torch.equal(m.weight.detach().float(), Wt)),
                              input_shape=tuple(xin.shape), output_shape=tuple(out.shape))
            rec.stages["conv"] = keep(out, ratio)
            layers.append(rec)
            st["n"] += 1
            st["open"] = rec
            st["prev"] = o.clone()
        return fn

    def pre_hook(name):
        def fn(m, inp):
            p = st["prev"]
            ok = p is not None and torch.is_tensor(inp[0]) and inp[0].shape == p.shape and torch.equal(inp[0].float(), p)
            st["pending"] = (name, ok)
        return fn

    def post_hook(name, kind):
        def fn(m, inp, out):
            pend, st["pending"] = st["pending"], None
            rec = st["open"]
            if rec is None or pend is None or pend[0] != name or not pend[1] or kind in rec.stages or \
                    (kind == "norm" and "act" in rec.stages) or not (torch.is_tensor(out) and out.dim() == 4):
                close()
                return
            ratio = H_in / out.shape[-2]
            rec.stages[kind] = keep(out, ratio)
            parent = rec.name.rsplit(".", 1)[0] + "." if "." in rec.name else ""
            shared = "" if name.startswith(parent) else ", paylaşımlı modül"
            rec.stage_ops[kind] = f"{name} ({type(m).__name__}{shared})"
            if kind == "norm" and isinstance(m, nn.BatchNorm2d):
                rec.norm_params = dict(op=name, eps=float(m.eps), gamma=m.weight.detach().float().cpu().numpy().copy(),
                                       beta=m.bias.detach().float().cpu().numpy().copy(),
                                       mean=m.running_mean.detach().float().cpu().numpy().copy(),
                                       var=m.running_var.detach().float().cpu().numpy().copy())
            if kind == "act":
                close()
            else:
                st["prev"] = out.detach().float().clone()
        return fn

    def other_hook(m, inp, out):
        close()

    handles = []
    for name, m in mod.named_modules():
        if len(list(m.children())) > 0:
            continue
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook(name)))
        elif isinstance(m, NORM_TYPES) or isinstance(m, ACT_TYPES):
            kind = "norm" if isinstance(m, NORM_TYPES) else "act"
            handles.append(m.register_forward_pre_hook(pre_hook(name)))
            handles.append(m.register_forward_hook(post_hook(name, kind)))
        else:
            handles.append(m.register_forward_hook(other_hook))
    t0 = time.time()
    peak = 0.0
    try:
        mod.to(dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        try:
            mod(x_dev)
        except _Stop:
            pass
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    finally:
        for hd in handles:
            hd.remove()
        mod.to("cpu")
        del x_dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if not layers:
        raise NoConvError(f"{lm.spec.label}: forward sırasında hiçbir Conv2d çalışmadı — filtre üretilmez.")
    return layers, bool(st["input_equal"]), time.time() - t0, peak


_RESULTS: "OrderedDict[str, Analysis]" = OrderedDict()


def run_analysis(spec: ModelSpec, img, mask=None, region_mode=REGION_MODES[0], size=512, device_choice="otomatik",
                 max_layers=6, freq_fn=None, store_fp16: Optional[bool] = None, cache_size=2) -> Analysis:
    lm = load_model(spec)
    region, mreg, rdesc = select_region(img, mask, region_mode, size)[:3]
    x, pdesc = preprocess(region, lm.preprocess, freq_fn)
    if x.shape[1] != lm.in_ch:
        raise RuntimeError(f"Ön işleme {x.shape[1]} kanal üretti, model {lm.in_ch} kanal bekliyor.")
    dev = pick_device(device_choice)
    if store_fp16 is None:
        store_fp16 = region.shape[0] * region.shape[1] > 768 * 768
    dt = np.float16 if store_fp16 else np.float32
    sha0 = lm.sha
    layers, ieq, sec, peak = analyze(lm, x, region.shape[:2], dev, int(max_layers), 4.0, dt)
    sha1 = state_sha(lm.module)
    stored = sum(a.nbytes for L in layers for a in L.stages.values()) / 2 ** 20
    an = Analysis(uuid.uuid4().hex[:12], spec.key, spec.label, lm.arch, lm.family, lm.preprocess, pdesc, region, mreg, rdesc,
                  x[0].numpy(), region.shape[:2], layers, ieq, dev, sec, peak, stored, np.dtype(dt).name, sha0, sha1,
                  lm.channel_names, lm.note)
    _RESULTS[an.token] = an
    while len(_RESULTS) > cache_size:
        _RESULTS.popitem(last=False)
    return an


def get_analysis(token):
    return _RESULTS.get(token) if token else None


def probe_layers(lm: LoadedModel, max_layers=6, size=128):
    """Görüntü olmadan mimari analizi: sıfır girdiyle kısa forward → tespit edilen ilk conv katmanları."""
    x = torch.zeros(1, lm.in_ch, size, size)
    layers, _, _, _ = analyze(lm, x, (size, size), "cpu", max_layers, 4.0, np.float16)
    return layers


def channel_contributions(an: Analysis, L: LayerRecord, f: int):
    """İlk katman (girdi = görüntü kanalları): filtre f'nin her giriş kanalına ayrı uygulanması W[f,c] ⋆ x_c.
    Toplam + bias = conv çıktısı (kanal birleştirmesinin açık gösterimi). Dönüş: (katkılar (C,h,w), toplam+bias)."""
    c = L.conv
    if L.idx != 0 or c["groups"] != 1 or L.weight.shape[1] != an.x.shape[0]:
        return None, None
    x = torch.from_numpy(an.x.astype(np.float32))[None]
    Wf = torch.from_numpy(L.weight[f:f + 1].astype(np.float32))
    pad = c["padding"]
    contrib = []
    for ch in range(x.shape[1]):
        y = F.conv2d(x[:, ch:ch + 1], Wf[:, ch:ch + 1], None, c["stride"], pad, c["dilation"], 1)
        contrib.append(y[0, 0].numpy())
    hh, ww = L.out_shape[1], L.out_shape[2]
    contrib = np.stack(contrib)[:, :hh, :ww]
    total = contrib.sum(0) + (float(L.bias[f]) if L.bias is not None else 0.0)
    return contrib, total


# ====================================================================== sezgisel filtre etiketi
def describe_filter(w: np.ndarray, channel_names=None) -> str:
    """İlk katman kernel'i için SEZGİSEL düşük-seviye özellik etiketi (kenar / çizgi / renk karşıtlığı / doku / düşük geçiren).
    Yalnız kernel geometrisinden hesaplanır; kesin sınıflandırma değildir."""
    a = np.abs(w).sum() + 1e-12
    dc = abs(w.sum()) / a
    rgb = w[:3] if w.shape[0] >= 3 else None
    if rgb is not None:
        s = rgb.reshape(3, -1).sum(1)
        if (s.max() > 0 > s.min()) and min(abs(s.max()), abs(s.min())) > 0.25 * np.abs(rgb).sum() / 3:
            hi, lo = int(np.argmax(s)), int(np.argmin(s))
            names = channel_names or ["R", "G", "B"]
            return f"renk karşıtlığı ({names[hi]}+ / {names[lo]}−)"
    g = w.mean(0)
    if dc > 0.6:
        return "düşük geçiren / ortalama parlaklık"
    gy, gx = np.gradient(g)
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)
    if coh > 0.45:
        theta = 0.5 * math.degrees(math.atan2(2 * jxy, jxx - jyy))  # gradyan yönü
        edge_dir = (theta + 90) % 180
        sym = float(np.corrcoef(g.ravel(), np.rot90(g, 2).ravel())[0, 1]) if g.std() > 0 else 0.0
        kind = "kenar" if sym < -0.3 else ("çizgi / çubuk" if sym > 0.3 else "kenar / çizgi")
        return f"{kind} (yön ≈ {edge_dir:.0f}°)"
    return "doku / nokta (yüksek frekans)"


# ====================================================================== görselleştirme
def fig_to_np(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _is_color_layer(L: LayerRecord):
    return L.idx == 0 and L.weight.shape[1] in (3, 5)


def kernel_image(L: LayerRecord, f: int, per_filter=True):
    """Kernel görüntüsü (0 = gri). İlk katmanda RGB bileşik (5 kanallıda ilk 3 kanal), diğerlerinde Σ giriş kanalı."""
    w = L.weight[f]
    if _is_color_layer(L):
        v = w[:3].transpose(1, 2, 0)
        s = np.abs(v).max() if per_filter else np.abs(L.weight[:, :3]).max()
        return np.clip(v / (2 * s + 1e-12) + 0.5, 0, 1)
    v = w.sum(0)
    s = np.abs(v).max() if per_filter else np.abs(L.weight.sum(1)).max()
    return np.clip(v / (2 * s + 1e-12) + 0.5, 0, 1)


def norm_map(a, mode, L: LayerRecord = None, stage=None):
    a = a.astype(np.float32)
    if mode == NORMS[1]:
        lo, hi = np.percentile(a, [1, 99])
    elif mode == NORMS[2] and L is not None:
        m = L.stage_stats(stage)[2]
        lo, hi = -m, m
    elif mode == NORMS[3] and L is not None:
        lo, hi = L.stage_stats(stage)[:2]
    else:
        lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def _thumb(a, side=192):
    h, w = a.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1:
        a = cv2.resize(a.astype(np.float32), (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return a


def _kern_ticks(ax, k):
    kh, kw = k
    ax.set_xticks([0, kw - 1]); ax.set_yticks([0, kh - 1])
    ax.tick_params(labelsize=6, length=2)


def _title_strip(text, width_px, line_h=16):
    """Türkçe karakterli başlık şeridi (yalnız tek metin → hızlı); genişliğe göre satırlara bölünür."""
    import textwrap
    lines = textwrap.wrap(text, max(30, int(width_px / 6.2)))
    h = line_h * len(lines) + 10
    fig = plt.figure(figsize=(width_px / 100, h / 100), dpi=100)
    fig.text(0.005, 0.5, "\n".join(lines), fontsize=8.5, va="center", ha="left", linespacing=1.3)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def _mosaic(tiles, labels, cols, tile_px, title="", nearest=False, gap=6, label_h=15):
    """Hızlı grid: [0,1] aralığında gri (H,W) ya da RGB (H,W,3) karolar → tek uint8 görüntü (matplotlib alt-grafiksiz)."""
    n = len(tiles)
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))
    cw, ch = tile_px + gap, tile_px + gap + label_h
    canvas = np.full((rows * ch + gap, cols * cw + gap, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        t = np.clip(np.asarray(t, np.float32), 0, 1)
        h, w = t.shape[:2]
        s = tile_px / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        interp = cv2.INTER_NEAREST if (nearest or s >= 2) else cv2.INTER_AREA
        t = cv2.resize(t, (nw, nh), interpolation=interp)
        if t.ndim == 2:
            t = np.repeat(t[..., None], 3, 2)
        x0, y0 = gap + c * cw, gap + r * ch + label_h
        canvas[y0:y0 + nh, x0:x0 + nw] = (t * 255).astype(np.uint8)
        cv2.putText(canvas, str(labels[i]), (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    if title:
        strip = _title_strip(title, canvas.shape[1])
        if strip.shape[1] != canvas.shape[1]:
            strip = cv2.resize(strip, (canvas.shape[1], strip.shape[0]), interpolation=cv2.INTER_AREA)
        canvas = np.vstack([strip, canvas])
    return canvas


def fig_kernel_grid(L: LayerRecord, per_filter=True, cols=8):
    n = L.n_filters
    tiles = [kernel_image(L, i, per_filter) for i in range(n)]
    o, c, kh, kw = L.weight.shape
    kind = "RGB bileşik (5 kanallıda ilk 3 kanal)" if _is_color_layer(L) else "Σ giriş kanalları (gri)"
    title = (f"Filtre ağırlıkları — {L.name} · {o}×{c}×{kh}×{kw} · {kind} · 0 = gri · "
             f"{'filtre başına' if per_filter else 'katman geneli'} ölçek")
    return _mosaic(tiles, [f"#{i}" for i in range(n)], cols if n <= 64 else 16, 72, title, nearest=True)


def fig_pairs(an: Analysis, L: LayerRecord, stage: str, page: int, norm: str, per_filter=True, cols_pairs=4):
    """Referans görseldeki düzen: 'filtre #i' kernel'i ile 'öznitelik haritası #i' yan yana."""
    n = L.n_filters
    a0 = max(0, (int(page) - 1) * PER_PAGE)
    idx = list(range(a0, min(n, a0 + PER_PAGE)))
    rows = int(math.ceil(len(idx) / cols_pairs))
    fig, axes = plt.subplots(rows, cols_pairs * 2, figsize=(cols_pairs * 2 * 1.75, rows * 1.85), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    maps = L.stages[stage]
    for j, f in enumerate(idx):
        r, c = divmod(j, cols_pairs)
        axk, axm = axes[r, 2 * c], axes[r, 2 * c + 1]
        im = kernel_image(L, f, per_filter)
        axk.axis("on"); axk.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1, interpolation="nearest")
        _kern_ticks(axk, L.conv["kernel"]); axk.set_title(f"filtre #{f}", fontsize=7)
        mp = norm_map(_thumb(maps[f], 220), norm, L, stage)
        axm.axis("on"); axm.imshow(mp, cmap="gray", vmin=0, vmax=1, extent=(0, maps.shape[2], maps.shape[1], 0))
        flat = " (sabit/0)" if float(maps[f].max()) == float(maps[f].min()) else ""
        axm.tick_params(labelsize=5, length=2); axm.set_title(f"öznitelik haritası #{f}{flat}", fontsize=7)
    fig.suptitle(f"{an.model_label} · {L.name} · {STAGES[stage]} · filtre {idx[0]}–{idx[-1]} / {n}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig_to_np(fig)


def fig_act_grid(an: Analysis, L: LayerRecord, stage: str, norm: str, cols=8):
    maps = L.stages[stage]
    n = maps.shape[0]
    tiles = [norm_map(_thumb(maps[i], 160), norm, L, stage) for i in range(n)]
    c, hh, ww = maps.shape
    title = f"Aktivasyon haritaları — {L.name} · {STAGES[stage]} · {c}×{hh}×{ww} · normalizasyon: {norm}"
    return _mosaic(tiles, [f"#{i}" for i in range(n)], cols if n <= 64 else 16, 150, title)


def _channel_label(an, L, c):
    if L.idx == 0 and c < len(an.channel_names):
        return an.channel_names[c]
    return f"giriş kanalı {c}"


def fig_kernel_detail(an: Analysis, L: LayerRecord, f: int):
    w = L.weight[f]
    cin, kh, kw = w.shape
    m = float(np.abs(w).max()) + 1e-12
    annotate = kh * kw <= 49 and cin <= 8
    b = f" · bias {float(L.bias[f]):+.4f}" if L.bias is not None else " · bias yok"
    if cin > 8:  # çok kanallı (2.+ katman): hızlı mozaik, RdBu (mavi −, kırmızı +), ortak simetrik ölçek
        cmap = plt.get_cmap("RdBu_r")
        tiles = [cmap((w[c] / m + 1) / 2)[..., :3] for c in range(cin)]
        title = (f"{L.name} · filtre #{f} · {cin}×{kh}×{kw} gerçek ağırlıklar (checkpoint){b} · her karo bir giriş kanalı · "
                 f"mavi −{m:.3f} … kırmızı +{m:.3f} · değerler tabloda")
        return _mosaic(tiles, [str(c) for c in range(cin)], 16, 48, title, nearest=True)
    if cin <= 8:
        cols = cin + (1 if _is_color_layer(L) else 0)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 2.3, 2.6), squeeze=False)
        axes = axes[0]
    else:
        cols = int(math.ceil(math.sqrt(cin)))
        rows = int(math.ceil(cin / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.95 + 0.6), squeeze=False)
        axes = axes.ravel()
    im = None
    for c in range(len(axes)):
        ax = axes[c]
        if c >= cin:
            if _is_color_layer(L) and c == cin:
                ax.imshow(kernel_image(L, f, True), interpolation="nearest")
                ax.set_title("RGB bileşik", fontsize=8); _kern_ticks(ax, (kh, kw))
            else:
                ax.axis("off")
            continue
        im = ax.imshow(w[c], cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
        if annotate:
            ax.set_title(_channel_label(an, L, c), fontsize=8)
            for yy in range(kh):
                for xx in range(kw):
                    ax.text(xx, yy, f"{w[c, yy, xx]:.2f}", ha="center", va="center", fontsize=6 if kw > 5 else 7,
                            color="white" if abs(w[c, yy, xx]) > 0.55 * m else "black")
            _kern_ticks(ax, (kh, kw))
        else:
            ax.set_title(str(c), fontsize=6, pad=1); ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.025, pad=0.02, label="ağırlık değeri")
    fig.suptitle(f"{L.name} · filtre #{f} · {cin}×{kh}×{kw} gerçek ağırlıklar (checkpoint){b}", fontsize=9)
    return fig_to_np(fig)


def kernel_values_df(an: Analysis, L: LayerRecord, f: int, max_rows=512):
    w = L.weight[f]
    cin, kh, kw = w.shape
    rows = []
    for c in range(cin):
        for yy in range(kh):
            rows.append([_channel_label(an, L, c), yy] + [round(float(v), 6) for v in w[c, yy]])
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows, columns=["giriş kanalı", "satır"] + [f"k{j}" for j in range(kw)])


def fig_act_detail(an: Analysis, L: LayerRecord, stage: str, f: int):
    a = L.stages[stage][f].astype(np.float32)
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    if a.min() < 0:
        m = float(np.abs(a).max()) + 1e-12
        im = ax.imshow(a, cmap="RdBu_r", vmin=-m, vmax=m)
    else:
        im = ax.imshow(a, cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="ham aktivasyon değeri")
    ax.set_title(f"{L.name} · filtre #{f} · {STAGES[stage]} · {a.shape[1]}×{a.shape[0]} (1/{L.out_stride:g})", fontsize=9)
    axh.hist(a.ravel(), bins=80, color="#4c72b0")
    axh.set_yscale("log")
    axh.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter())  # mathtext'siz etiketler
    axh.set_title(f"min {a.min():.3f} · maks {a.max():.3f}\nort {a.mean():.3f} · std {a.std():.3f} · >0 %{100 * (a > 0).mean():.1f}", fontsize=8)
    axh.tick_params(labelsize=7)
    fig.tight_layout()
    return fig_to_np(fig)


def _mask_at(an: Analysis, shape_hw):
    if an.mask is None or not (an.mask > 127).any():
        return None
    return cv2.resize((an.mask > 127).astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def stats_df(an: Analysis, L: LayerRecord, stage: str):
    maps = L.stages[stage]
    mk = _mask_at(an, maps.shape[1:])
    rows = []
    for f in range(maps.shape[0]):
        a = maps[f].astype(np.float32)
        r = dict(filtre=f, min=round(float(a.min()), 4), maks=round(float(a.max()), 4), ort=round(float(a.mean()), 4),
                 std=round(float(a.std()), 4), **{"%>0": round(100 * float((a > 0).mean()), 1)}, **{"ort|a|": round(float(np.abs(a).mean()), 4)})
        if mk is not None and mk.any() and (~mk).any():
            i, o = float(np.abs(a[mk]).mean()), float(np.abs(a[~mk]).mean())
            r.update({"GT içi ort|a|": round(i, 4), "GT dışı ort|a|": round(o, 4), "iç/dış oranı": round(i / (o + 1e-9), 3)})
        if _is_color_layer(L):
            r["sezgisel tür"] = describe_filter(L.weight[f], an.channel_names)
        rows.append(r)
    return pd.DataFrame(rows)


def _overlay(rgb, a01, alpha=0.5):
    hm = cv2.applyColorMap((cv2.resize(a01, (rgb.shape[1], rgb.shape[0])) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) * (1 - alpha) + hm.astype(np.float32) * alpha).astype(np.uint8)


def _draw_mask_box(rgb, mask):
    out = rgb.copy()
    if mask is not None and (mask > 127).any():
        ys, xs = np.nonzero(mask > 127)
        cv2.rectangle(out, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (40, 120, 255), max(1, out.shape[1] // 250))
    return out


def fig_compare(an: Analysis, L: LayerRecord, f: int, stage: str, norm: str):
    """D · Orijinal ↔ filtre çıktısı. İlk katmanda kanal katkıları W[f,c]⋆x_c ve toplamları da gösterilir."""
    reg = _draw_mask_box(an.region, an.mask)
    contrib, total = channel_contributions(an, L, f)
    stage_keys = list(L.stages.keys())
    if contrib is not None:
        nC = contrib.shape[0]
        cols = max(nC + 2, len(stage_keys) + 1)
        fig, axes = plt.subplots(2, cols, figsize=(cols * 2.6, 5.6), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].imshow(reg); axes[0, 0].set_title("Orijinal bölge (mavi: GT)", fontsize=8)
        im = kernel_image(L, f, True)
        axes[0, 1].imshow(im, interpolation="nearest"); axes[0, 1].set_title(f"filtre #{f} (RGB bileşik)", fontsize=8)
        m = float(np.abs(contrib).max()) + 1e-12
        for c in range(nC):
            axes[0, 2 + c].imshow(contrib[c], cmap="RdBu_r", vmin=-m, vmax=m)
            axes[0, 2 + c].set_title(f"W[{f},{_channel_label(an, L, c)}] ⋆ x_{_channel_label(an, L, c)}", fontsize=8)
        d = float(np.abs(total - L.stages["conv"][f].astype(np.float32)).max())
        axes[1, 0].imshow(norm_map(total, norm), cmap="gray"); axes[1, 0].set_title(f"Σ kanallar + bias\n(max|Δ| conv'a göre {d:.1e})", fontsize=8)
        for j, sk in enumerate(stage_keys):
            axes[1, 1 + j].imshow(norm_map(L.stages[sk][f], norm, L, sk), cmap="gray")
            axes[1, 1 + j].set_title(STAGES[sk], fontsize=8)
        a = norm_map(np.abs(L.stages[stage][f].astype(np.float32)), NORMS[1])
        axes[1, 1 + len(stage_keys)].imshow(_overlay(an.region, a)); axes[1, 1 + len(stage_keys)].set_title("|aktivasyon| bindirme", fontsize=8)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for ax in axes:
            ax.axis("off")
        axes[0].imshow(reg); axes[0].set_title("Orijinal bölge (mavi: GT)", fontsize=8)
        axes[1].imshow(norm_map(L.stages[stage][f], norm, L, stage), cmap="gray"); axes[1].set_title(f"filtre #{f} · {STAGES[stage]}", fontsize=8)
        a = norm_map(np.abs(L.stages[stage][f].astype(np.float32)), NORMS[1])
        axes[2].imshow(_overlay(an.region, a)); axes[2].set_title("|aktivasyon| bindirme", fontsize=8)
        e = np.abs(L.stages[stage].astype(np.float32)).mean(0)
        axes[3].imshow(_overlay(an.region, norm_map(e, NORMS[1]))); axes[3].set_title("katman enerjisi: ort|a| (tüm filtreler)", fontsize=8)
    fig.suptitle(f"{an.model_label} · {L.name} · filtre #{f}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig_to_np(fig)


def fig_top_filters(an: Analysis, L: LayerRecord, stage: str, norm: str, k=8):
    df = stats_df(an, L, stage)
    key = "iç/dış oranı" if "iç/dış oranı" in df.columns else "ort|a|"
    top = df.sort_values(key, ascending=False).head(k)
    cols = 3
    rows = int(math.ceil((len(top) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(_draw_mask_box(an.region, an.mask)); axes[0, 0].set_title("Orijinal bölge", fontsize=8)
    for j, r in enumerate(top.itertuples(index=False)):
        f = int(r[0])
        ax = axes.ravel()[j + 1]
        ax.imshow(norm_map(L.stages[stage][f], norm, L, stage), cmap="gray")
        lab = f"#{f} · {key} {df.loc[f, key]}"
        if "sezgisel tür" in df.columns:
            lab += f"\n{df.loc[f, 'sezgisel tür']}"
        ax.set_title(lab, fontsize=7)
    crit = "GT kutusu içi/dışı ort|a| oranı" if key == "iç/dış oranı" else "ortalama |aktivasyon|"
    fig.suptitle(f"En güçlü {len(top)} filtre ({crit}) — {L.name} · {STAGES[stage]}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig_to_np(fig)


def layers_df(an: Analysis):
    rows = []
    for L in an.layers:
        c = L.conv
        rows.append({"#": L.idx, "katman": L.name, "ağırlık şekli": "×".join(map(str, L.weight.shape)),
                     "filtre": L.n_filters, "kernel": f"{c['kernel'][0]}×{c['kernel'][1]}", "stride": str(c["stride"]),
                     "padding": str(c["padding"]), "dilation": str(c["dilation"]), "groups": c["groups"], "bias": c["bias"],
                     "çıkış (C×h×w)": "×".join(map(str, L.out_shape)), "çözünürlük": f"1/{L.out_stride:g}",
                     "aşamalar": " → ".join(["conv"] + [L.stage_ops[k] for k in L.stages if k != "conv"]),
                     "doğrulama max|Δ|": f"{L.verify['max_abs']:.2e}", "göreli": f"{L.verify['rel']:.1e}",
                     "ağırlık = checkpoint": L.verify["weight_is_checkpoint"]})
    return pd.DataFrame(rows)


def verify_ok(L: LayerRecord, tol_rel=1e-4):
    return L.verify["weight_is_checkpoint"] and L.verify["rel"] <= tol_rel


def verify_md(an: Analysis):
    ok_all = all(verify_ok(L) for L in an.layers)
    worst = max(L.verify["rel"] for L in an.layers)
    s = (f"**Doğrulama:** {'✅' if ok_all and an.input_equal and an.sha_before == an.sha_after else '⚠️'} "
         f"bağımsız F.conv2d (checkpoint kernel'i) = modelin kendi conv çıktısı — en kötü göreli fark **{worst:.1e}** "
         f"(tolerans 1e-4; fp32{' / cuDNN-TF32' if an.device == 'cuda' else ''}) · "
         f"ilk conv'un girdisi = ön işlenmiş tensör: **{'evet' if an.input_equal else 'HAYIR'}** · "
         f"ağırlık SHA-256 analiz öncesi/sonrası: **{'aynı' if an.sha_before == an.sha_after else 'FARKLI'}** "
         f"(`{an.sha_after[:16]}…`)  \n"
         f"**Hesap:** {an.device} · {an.sec:.2f} s · tepe GPU {an.peak_gpu_mb:.0f} MiB · saklanan haritalar "
         f"{an.stored_mb:.1f} MiB ({an.store_dtype}) · forward ilk katmanlardan sonra durduruldu (derin katmanlar hesaplanmadı).")
    return s


def arch_md(lm: LoadedModel, layers=None):
    s = (f"**Mimari:** `{lm.arch}` · {lm.n_params / 1e6:.2f} M parametre · giriş {lm.in_ch} kanal "
         f"({', '.join(lm.channel_names)}) · ön işleme `{lm.preprocess}`  \n**Kaynak:** `{Path(lm.spec.path).name}` "
         f"({lm.spec.group}) · {lm.note}")
    if layers:
        s += "  \n**Tespit edilen ilk conv katmanları** (yürütme sırası, çıkış ≥ girdinin 1/4'ü): " + \
             " · ".join(f"`{L.name}` {L.weight.shape[0]}×{L.weight.shape[1]}×{L.conv['kernel'][0]}×{L.conv['kernel'][1]} "
                        f"s{L.conv['stride'][0]} (1/{L.out_stride:g})" for L in layers)
    return s


def layer_md(an: Analysis, L: LayerRecord):
    c = L.conv
    w = L.weight
    s = (f"**Katman:** `{L.name}` ({c['type']}) · **filtre sayısı** {L.n_filters} · **kernel** {c['kernel'][0]}×{c['kernel'][1]} · "
         f"giriş {c['in_ch']} → çıkış {c['out_ch']} kanal · stride {c['stride']} · padding {c['padding']} ({c['padding_mode']}) · "
         f"dilation {c['dilation']} · groups {c['groups']} · bias {'var' if c['bias'] else 'yok'}  \n"
         f"**Ağırlıklar:** min {w.min():+.4f} · maks {w.max():+.4f} · ort {w.mean():+.5f} · std {w.std():.4f} · "
         f"‖W‖₂ {np.linalg.norm(w):.3f}")
    if L.norm_params is not None:
        s += f"  \n**Sonraki normalizasyon:** `{L.norm_params['op']}` (eps {L.norm_params['eps']:g}) — filtre başına γ, β, μ, σ² ayrıntıda."
    s += f"  \n**Kayıtlı aşamalar:** " + " → ".join([STAGES[k] + (f" `{L.stage_ops[k]}`" if k in L.stage_ops else "") for k in L.stages])
    s += f"  \n**Girdi bölgesi:** {an.region_desc}  \n**Ön işleme:** {an.prep_desc}"
    return s


def filter_md(an: Analysis, L: LayerRecord, f: int):
    s = f"**Filtre #{f}** · `{L.name}`"
    if _is_color_layer(L):
        s += f" · sezgisel tür: **{describe_filter(L.weight[f], an.channel_names)}** (kernel geometrisinden; kesin değildir)"
    if L.norm_params is not None:
        p = L.norm_params
        g, b, mu, var = p["gamma"][f], p["beta"][f], p["mean"][f], p["var"][f]
        s += (f"  \nBN: γ {g:+.4f} · β {b:+.4f} · μ {mu:+.4f} · σ² {var:.4f} → BN(z) = {g / math.sqrt(var + p['eps']):+.4f}·z "
              f"{b - g * mu / math.sqrt(var + p['eps']):+.4f}")
    return s


def export_raw(an: Analysis, L: LayerRecord, f: int):
    out = Path(tempfile.mkdtemp(prefix="xai_filters_"))
    base = f"{an.model_key.replace('/', '_')}__{L.name.replace('.', '_')}"
    npz = out / f"{base}.npz"
    np.savez_compressed(npz, weight=L.weight, bias=L.bias if L.bias is not None else np.zeros(0),
                        **{f"act_{k}": v.astype(np.float32) for k, v in L.stages.items()}, region=an.region)
    csv = out / f"{base}__filtre{f}.csv"
    kernel_values_df(an, L, f, max_rows=10 ** 6).to_csv(csv, index=False)
    js = out / f"{base}__ozet.json"
    js.write_text(json.dumps(dict(model=an.model_key, label=an.model_label, arch=an.arch, region=an.region_desc,
                                  preprocess=an.prep_desc, device=an.device, layers=layers_df(an).to_dict("records"),
                                  sha256=an.sha_after, input_equal=an.input_equal), ensure_ascii=False, indent=1, default=str),
                  encoding="utf-8")
    return [str(npz), str(csv), str(js)]


# ====================================================================== Gradio sekmesi
def build_tab(gr, *, models_dir, yolo_runs_dir=None, gpu=None, freq_fn=None, load_sample=None, sample_choices=None,
              image_states=None, default_key=None, store_fp16=None):
    """Bağımsız 'Filter Visualization / Katman Filtreleri' sekmesinin içeriğini kurar (çağıran gr.Tab bağlamında).
    gpu: @spaces.GPU sarmalayıcısı (ZeroGPU) · freq_fn: app.project_freq_channels · load_sample(key)→(img,mask,auth,meta)
    image_states: (st_orig, st_mask) — Girdi sekmesindeki görüntü."""
    specs = discover_models(models_dir, yolo_runs_dir)
    by_key = {s.key: s for s in specs}
    choices = [(s.label, s.key) for s in specs]
    dkey = default_key if default_key in by_key else (specs[0].key if specs else None)
    stage_by_label = {v: k for k, v in STAGES.items()}

    def _compute(key, img, mask, region_mode, size, dev, maxl):
        an = run_analysis(by_key[key], img, mask, region_mode, int(size), dev, int(maxl), freq_fn, store_fp16)
        return an

    compute = gpu(duration=60)(_compute) if gpu is not None else _compute

    def on_model(key):
        if not key:
            return "Model seçin."
        try:
            lm = load_model(by_key[key])
            return arch_md(lm, probe_layers(lm))
        except NoConvError as e:
            return f"⚠️ {e}"
        except Exception as e:
            return f"⚠️ Model yüklenemedi: `{e}`"

    def _source(src, key, up, orig, msk):
        if src == SOURCES[0]:
            if not key or load_sample is None:
                raise gr.Error("Veri setinden bir örnek seçin.")
            img, mask, _auth, _meta = load_sample(key)
            return img, mask, f"veri seti `{key}`"
        if src == SOURCES[1]:
            if orig is None:
                raise gr.Error("Girdi sekmesinde yüklü görüntü yok.")
            return orig, msk, "Girdi sekmesindeki görüntü"
        if up is None:
            raise gr.Error("Bir görüntü yükleyin.")
        return up, None, "yüklenen görüntü"

    def _stage_update(L, cur_label=None):
        labels = [STAGES[k] for k in L.stages]
        val = cur_label if cur_label in labels else labels[-1]
        return gr.update(choices=labels, value=val), val

    @_plot_locked
    def render_layer(token, li, stage_lab, norm, per_filter, page):
        an = get_analysis(token)
        if an is None:
            return [" "] + [None] * 5
        L = an.layers[int(li)]
        st = stage_by_label.get(stage_lab, list(L.stages)[-1])
        if st not in L.stages:
            st = list(L.stages)[-1]
        return [layer_md(an, L), fig_kernel_grid(L, bool(per_filter)), fig_pairs(an, L, st, int(page), norm, bool(per_filter)),
                fig_act_grid(an, L, st, norm), stats_df(an, L, st), fig_top_filters(an, L, st, norm)]

    @_plot_locked
    def render_filter(token, li, f, stage_lab, norm):
        an = get_analysis(token)
        if an is None:
            return [None, None, None, None, " "]
        L = an.layers[int(li)]
        f = int(min(max(0, int(f)), L.n_filters - 1))
        st = stage_by_label.get(stage_lab, list(L.stages)[-1])
        if st not in L.stages:
            st = list(L.stages)[-1]
        return [fig_kernel_detail(an, L, f), kernel_values_df(an, L, f), fig_act_detail(an, L, st, f),
                fig_compare(an, L, f, st, norm), filter_md(an, L, f)]

    @_plot_locked
    def render_pairs(token, li, stage_lab, norm, per_filter, page):
        an = get_analysis(token)
        if an is None:
            return None
        L = an.layers[int(li)]
        st = stage_by_label.get(stage_lab, list(L.stages)[-1])
        st = st if st in L.stages else list(L.stages)[-1]
        return fig_pairs(an, L, st, int(page), norm, bool(per_filter))

    def _layer_choices(an):
        return [(f"{L.idx} · {L.name} · {L.weight.shape[0]} filtre {L.conv['kernel'][0]}×{L.conv['kernel'][1]} · 1/{L.out_stride:g}", L.idx)
                for L in an.layers]

    with gr.Row():
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("#### A) Model ve görüntü seçimi")
            f_model = gr.Dropdown(choices, value=dkey, label="Model (kayıtlı eğitilmiş ağırlıklar)", filterable=True)
            f_arch = gr.Markdown(on_model(dkey) if dkey else "Model bulunamadı.", elem_classes="small-md")
            f_src = gr.Radio(SOURCES, value=SOURCES[0] if load_sample else SOURCES[2], label="Görüntü kaynağı")
            f_key = gr.Dropdown(choices=sample_choices or [], label="Örnek (bölüm:kimlik)", filterable=True,
                                allow_custom_value=True, visible=load_sample is not None,
                                value=(sample_choices[0][1] if sample_choices else None))
            f_up = gr.Image(label="Görüntü yükle (PNG/JPG)", type="numpy", height=220, visible=load_sample is None)
            f_region = gr.Radio(REGION_MODES, value=REGION_MODES[0], label="Analiz bölgesi")
            f_size = gr.Slider(64, MAX_REGION, value=512, step=32, label="Bölge boyutu N (px) — bellek kontrolü")
            with gr.Row():
                f_dev = gr.Radio(DEVICES, value="otomatik", label="Cihaz")
                f_maxl = gr.Slider(1, 8, value=6, step=1, label="En fazla ilk conv katmanı")
            b_run = gr.Button("▶ Filtreleri çıkar ve görüntüye uygula", variant="primary", size="lg")
            gr.Markdown("#### Görünüm")
            f_layer = gr.Dropdown([], label="Analiz edilecek ilk katman")
            f_filter = gr.Slider(0, 63, value=0, step=1, label="Filtre numarası")
            f_stage = gr.Radio(list(STAGES.values()), value=STAGES["act"], label="Aktivasyon aşaması")
            f_norm = gr.Dropdown(NORMS, value=NORMS[0], label="Harita normalizasyonu (yalnız görüntüleme; ham değerler tabloda/indirmede)")
            f_pf = gr.Checkbox(True, label="Kernel'leri filtre başına ölçekle (kapalı: katman geneli; 0 = gri)")
            f_page = gr.Slider(1, 4, value=1, step=1, label=f"Filtre–harita çiftleri sayfası ({PER_PAGE}'lık)")
            b_dl = gr.Button("Ham değerleri indir (NPZ · CSV · JSON)")
            f_files = gr.File(label="Ham çıktı dosyaları", file_count="multiple")
        with gr.Column(scale=2):
            f_status = gr.Markdown("Model ve görüntü seçip **▶** ile başlatın. Kernel'ler checkpoint'ten okunur; yeni filtre "
                                   "eğitilmez/üretilmez; derin katmanlara ilerlenmez.")
            with gr.Tabs():
                with gr.Tab("B · Kernel / filtre ağırlıkları"):
                    f_lmd = gr.Markdown(" ")
                    f_kgrid = gr.Image(label="İlk katmandaki tüm filtreler (gerçek kernel ağırlıkları)", type="numpy", interactive=False)
                    f_kdet = gr.Image(label="Seçili filtrenin ağırlık matrisi (değerler hücrelerde)", type="numpy", interactive=False)
                    f_kdf = gr.Dataframe(label="Seçili filtrenin ağırlık değerleri", wrap=True, interactive=False)
                with gr.Tab("C · Filtre aktivasyon haritaları"):
                    f_pairs = gr.Image(label="Filtre ↔ öznitelik haritası çiftleri", type="numpy", interactive=False)
                    f_agrid = gr.Image(label="Tüm filtrelerin aktivasyon haritaları (grid)", type="numpy", interactive=False)
                    f_adet = gr.Image(label="Seçili filtrenin aktivasyon haritası (ham değer ölçeği) + histogram", type="numpy", interactive=False)
                    f_adf = gr.Dataframe(label="Filtre bazında ham aktivasyon istatistikleri", wrap=True, interactive=False)
                with gr.Tab("D · Orijinal görüntü ile karşılaştırma"):
                    f_fmd = gr.Markdown(" ")
                    f_cmp = gr.Image(label="Orijinal ↔ filtre çıktısı (ilk katmanda kanal katkıları W⋆x)", type="numpy", interactive=False)
                    f_top = gr.Image(label="Görüntüde en güçlü yanıt veren filtreler (kenar / çizgi / doku)", type="numpy", interactive=False)
                with gr.Tab("Katman özeti ve doğrulama"):
                    f_vmd = gr.Markdown(" ")
                    f_ldf = gr.Dataframe(label="Tespit edilen ilk katmanlar", wrap=True, interactive=False)
    st_tok = gr.State(None)

    layer_out = [f_lmd, f_kgrid, f_pairs, f_agrid, f_adf, f_top]
    filt_out = [f_kdet, f_kdf, f_adet, f_cmp, f_fmd]

    def on_run(key, src, skey, up, orig, msk, region_mode, size, dev, maxl, stage_lab, norm, pf):
        if not key:
            raise gr.Error("Model seçin.")
        img, mask, sdesc = _source(src, skey, up, orig, msk)
        try:
            an = compute(key, img, mask, region_mode, size, dev, maxl)
        except NoConvError as e:
            raise gr.Error(str(e))
        _RESULTS[an.token] = an  # ZeroGPU alt sürecinden dönen sonucu ana süreç önbelleğine koy
        L = an.layers[0]
        su, sval = _stage_update(L, stage_lab)
        status = (f"**{an.model_label}** · `{an.arch}` · kaynak: {sdesc} · {len(an.layers)} ilk conv katmanı tespit edildi.  \n"
                  + verify_md(an))
        lu = gr.update(choices=_layer_choices(an), value=0)
        fu = gr.update(maximum=L.n_filters - 1, value=0)
        pu = gr.update(maximum=max(1, int(math.ceil(L.n_filters / PER_PAGE))), value=1)
        return ([an.token, status, lu, fu, pu, su, verify_md(an), layers_df(an)]
                + render_layer(an.token, 0, sval, norm, pf, 1) + render_filter(an.token, 0, 0, sval, norm))

    def on_layer(token, li, stage_lab, norm, pf):
        an = get_analysis(token)
        if an is None or li is None:
            return [gr.update(), gr.update(), gr.update()] + [None] * 6 + [None] * 5
        L = an.layers[int(li)]
        su, sval = _stage_update(L, stage_lab)
        fu = gr.update(maximum=L.n_filters - 1, value=0)
        pu = gr.update(maximum=max(1, int(math.ceil(L.n_filters / PER_PAGE))), value=1)
        return [fu, pu, su] + render_layer(token, li, sval, norm, pf, 1) + render_filter(token, li, 0, sval, norm)

    f_model.change(on_model, [f_model], [f_arch])
    f_src.change(lambda s: (gr.update(visible=s == SOURCES[0]), gr.update(visible=s == SOURCES[2])), [f_src], [f_key, f_up])
    run_in = [f_model, f_src, f_key, f_up, image_states[0] if image_states else st_tok, image_states[1] if image_states else st_tok,
              f_region, f_size, f_dev, f_maxl, f_stage, f_norm, f_pf]
    b_run.click(on_run, run_in, [st_tok, f_status, f_layer, f_filter, f_page, f_stage, f_vmd, f_ldf] + layer_out + filt_out)
    f_layer.input(on_layer, [st_tok, f_layer, f_stage, f_norm, f_pf], [f_filter, f_page, f_stage] + layer_out + filt_out)
    for comp in (f_stage, f_norm):
        comp.input(render_layer, [st_tok, f_layer, f_stage, f_norm, f_pf, f_page], layer_out).then(
            render_filter, [st_tok, f_layer, f_filter, f_stage, f_norm], filt_out)
    f_pf.input(render_layer, [st_tok, f_layer, f_stage, f_norm, f_pf, f_page], layer_out)
    f_filter.release(render_filter, [st_tok, f_layer, f_filter, f_stage, f_norm], filt_out)
    f_page.release(render_pairs, [st_tok, f_layer, f_stage, f_norm, f_pf, f_page], [f_pairs])
    b_dl.click(lambda t, li, f: export_raw(get_analysis(t), get_analysis(t).layers[int(li or 0)], int(f)) if get_analysis(t) else None,
               [st_tok, f_layer, f_filter], [f_files])
    return dict(model=f_model, run=b_run, status=f_status)
