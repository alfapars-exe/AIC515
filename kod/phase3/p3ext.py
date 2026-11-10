"""
p3ext.py — Faz 3 eklentileri (TR): Faz 2 kod tabanını DEĞİŞTİRMEDEN genişletir (import edildiğinde yama uygular).

* Mimari kaydı (model adı önekine göre):
    unet_*       → smp.Unet(resnet18)                      (Faz 2 ile aynı)
    dlv3p_*      → smp.DeepLabV3Plus(resnet18)             (W2: ikinci mimari)
    unetjoint_*  → smp.Unet(resnet18, classes=2); kanal 0 = birincil (piksel/glif) maske, kanal 1 = bbox yardımcı
                   → değerlendirmede predict_full_multi zaten [:, 0] kullanır.
* Yeni GT varyantları: '<v>_nm' (işaretçi pikselleri GT'den çıkarılmış; make_marker_repaired.py üretir).
* P3_MARKER_REPAIR=1 ortam değişkeni → TrainingSet görüntüleri (train+val) phase3/marker_repaired/images/<id>.png
  varsa ondan okunur (işaretçi kalıntısı eşleşen authentic'ten geri yazılmış). Yalnız EĞİTİM süreçlerinde açılır;
  değerlendirme süreçleri ham görüntüyü görür.
* Ek test-zamanı koşulları D.DEGRADATIONS'a eklenir: clahe, gray (Faz 2 §6.4'te "test edilmedi" olanlar).
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("USERNAME", "user")
HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "phase2" / "experiments"
sys.path.insert(0, str(EXP))

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import common.p2data as D
import infer_fullimage as I

OUT = D.PROJECT_ROOT / "phase3"            # çıktı kökü (results/, figures/, logs/, marker_repaired/)
YOLO_OUT = D.PROJECT_ROOT / "phase3_yolo"  # YOLO çıktı kökü (runs/, preds_boxes/, results/)
REPAIR_DIR = OUT / "marker_repaired" / "images"
for v in ("pixel_eps8_morph_nm", "glyph_v2_nm"):
    if v not in D.GT_VARIANTS:
        D.GT_VARIANTS.append(v)


def build_model(name_or_arch: str, pretrained: bool = True) -> torch.nn.Module:
    w = "imagenet" if pretrained else None
    a = name_or_arch
    if a.startswith("dlv3p"):
        return smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=w, in_channels=3, classes=1)
    if a.startswith("unetjoint"):
        return smp.Unet(encoder_name="resnet18", encoder_weights=w, in_channels=3, classes=2)
    return smp.Unet(encoder_name="resnet18", encoder_weights=w, in_channels=3, classes=1)


def load_model(name: str, device) -> torch.nn.Module:
    m = build_model(name, pretrained=False)
    sd = torch.load(D.MODELS_DIR / f"{name}.pt", map_location="cpu")
    m.load_state_dict(sd)
    return m.to(device).eval().to(memory_format=torch.channels_last)


I.load_model = load_model  # run_eval / probe betikleri I.load_model'ı çağırır

if os.environ.get("P3_MARKER_REPAIR") == "1":
    _orig_img_path = D.img_path

    def _img_path(new_id: str, split: str = "train"):
        if split != "test":
            p = REPAIR_DIR / f"{new_id}.png"
            if p.exists():
                return p
        return _orig_img_path(new_id, split)

    D.img_path = _img_path


def degrade_clahe(img_u8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def degrade_gray(img_u8: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)


D.DEGRADATIONS.setdefault("clahe", degrade_clahe)
D.DEGRADATIONS.setdefault("gray", degrade_gray)


def gmask(rgb: np.ndarray) -> np.ndarray:
    """Yeşil işaretçi pikseli (Faz 2 Ek-C ile aynı tanım)."""
    r, g, b = (rgb[..., k].astype(np.int16) for k in range(3))
    return (g - np.maximum(r, b) > 50) & (g > 110)
