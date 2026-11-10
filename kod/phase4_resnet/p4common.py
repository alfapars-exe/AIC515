"""
p4common.py — Faz 4 ortak katman (TR): ResNet-18 parça sınıflandırıcısı için tam veri parça indeksleri,
parça çıkarımı (ham ve işaretçi-onarımlı), model kurulumu ve tam görüntü kayan pencere skoru.

Faz 1 (`baselines/train_resnet.py`) tarifinin tam veriye taşınmış hâlidir; Faz 2 veri katmanı (`common.p2data`)
DEĞİŞTİRİLMEDEN kullanılır. Parça tanımları Faz 1 ile aynıdır:
  tampered     : maske bbox merkezli 256x256 (+-32 px kaydırma; bbox 256'dan genişse bbox içinde rastgele pencere)
  clean_same   : aynı sahte görüntüde bbox ile kesişmeyen rastgele 256x256
  clean_paired : eşleşen authentic görüntüde ilk tampered parça ile AYNI koordinat
Tam görüntü skoru uygulamadaki `_predict_resnet` ile aynı protokoldür: < 224 ise replicate dolgu, 224 pencere, adım 112.
"""
from __future__ import annotations
import os, sys, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

os.environ.setdefault("USERNAME", "user")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2" / "experiments"))

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision
from scipy import ndimage as ndi
import common.p2data as D  # noqa: E402

ROOT = D.PROJECT_ROOT
OUT = ROOT / "phase4_resnet"
RES, LOG, CACHE, PRED = OUT / "results", OUT / "logs", OUT / "cache", OUT / "preds"
MODELS = ROOT / "models" / "phase4"
MODELS_P1 = ROOT / "models"
REPAIR_DIR = ROOT / "phase3" / "marker_repaired" / "images"
PATCH, OUTSZ, STRIDE = 256, 224, 112
INDEX_SEED = {"train": 42, "val": 43}   # Faz 1 ile aynı kural (train: seed, val: seed + 1); tüm koşularda SABİT


def ensure_dirs():
    for d in (RES, LOG, CACHE, PRED, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- işaretçi (Faz 2 Ek-C / Faz 3 ile aynı tanım)
def gmask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., k].astype(np.int16) for k in range(3))
    return (g - np.maximum(r, b) > 50) & (g > 110)


def ring_of(mb: np.ndarray) -> np.ndarray:
    return ndi.binary_dilation(mb, iterations=6) & ~ndi.binary_erosion(mb, iterations=6)


def marker_residue(forged: np.ndarray, auth: np.ndarray, bbox_mask: np.ndarray) -> np.ndarray:
    """make_marker_repaired.py ile birebir: residue = gmask(forged) & ~gmask(auth) & ring(bbox, 6 px); 1 px genişletilmiş."""
    res = gmask(forged) & ~gmask(auth) & ring_of(bbox_mask > 0)
    return ndi.binary_dilation(res, iterations=1)


def forged_path(new_id: str, split: str, mr: bool) -> Path:
    """mr=True ve onarılmış görüntü varsa onu döndürür (yalnız TrainingSet: train + val)."""
    if mr and split != "test":
        p = REPAIR_DIR / f"{new_id}.png"
        if p.exists():
            return p
    return D.img_path(new_id, split)


# ---------------------------------------------------------------- parça indeksleri (Faz 1 örnekleyicileri)
def sample_bbox_window(bbox, W, H, size, rng, jitter=32) -> Tuple[int, int]:
    x0b, y0b, x1b, y1b = bbox
    bw, bh = x1b - x0b, y1b - y0b
    if bw > size:
        cx = rng.integers(x0b + size // 2, x1b - size // 2 + 1)
    else:
        cx = (x0b + x1b) // 2 + rng.integers(-jitter, jitter + 1)
    if bh > size:
        cy = rng.integers(y0b + size // 2, y1b - size // 2 + 1)
    else:
        cy = (y0b + y1b) // 2 + rng.integers(-jitter, jitter + 1)
    x0 = int(np.clip(cx - size // 2, 0, max(0, W - size)))
    y0 = int(np.clip(cy - size // 2, 0, max(0, H - size)))
    return x0, y0


def sample_clean_window(bbox, W, H, size, rng, max_try=200) -> Optional[Tuple[int, int]]:
    x0b, y0b, x1b, y1b = bbox
    if W < size or H < size:
        return None
    for _ in range(max_try):
        x0 = int(rng.integers(0, W - size + 1)); y0 = int(rng.integers(0, H - size + 1))
        if x0 + size <= x0b or x0 >= x1b or y0 + size <= y0b or y0 >= y1b:
            return x0, y0
    return None


def split_ids_table(split: str) -> pd.DataFrame:
    """train: dışlananlar hariç (Faz 2/3 ile aynı 2748 görüntü); val: 487; test: 812."""
    tab = D.split_table(split)
    if split == "train":
        tab = tab[~tab.index.isin(set(D.load_excluded()))]
    return tab


def build_index(split: str, n_tampered: int, limit: int = 0) -> pd.DataFrame:
    tab = split_ids_table(split)
    if limit:
        tab = tab.iloc[:limit]
    ids = list(tab.index)
    with ThreadPoolExecutor(8) as ex:
        masks = list(ex.map(lambda i: D.read_mask(D.bbox_mask_path(i, "train")), ids))
    rng = np.random.default_rng(INDEX_SEED[split])
    rows = []
    for new_id, m in zip(ids, masks):
        r = tab.loc[new_id]
        H, W = m.shape
        bbox = D.mask_bbox(m)
        base = dict(new_id=new_id, tool=r["tool"], source=r["source"], area_bin=r["area_bin"], group=r["group"], W=W, H=H)
        first = None
        for k in range(n_tampered):
            x0, y0 = sample_bbox_window(bbox, W, H, PATCH, rng)
            first = first or (x0, y0)
            rows.append(dict(base, x0=x0, y0=y0, label=1, patch_type="tampered", rep=k))
        xy = sample_clean_window(bbox, W, H, PATCH, rng)
        if xy is not None:
            rows.append(dict(base, x0=xy[0], y0=xy[1], label=0, patch_type="clean_same", rep=0))
        if bool(r["pair_accepted"]) and bool(r["pair_same_size"]):
            rows.append(dict(base, x0=first[0], y0=first[1], label=0, patch_type="clean_paired", rep=0))
    return pd.DataFrame(rows)


def extract_both(index: pd.DataFrame, split_tab: pd.DataFrame, threads: int = 8) -> Tuple[np.ndarray, np.ndarray, int]:
    """İndeksteki parçaları iki varyantta çıkarır: (ham, işaretçi-onarımlı, onarılmış görüntü sayısı).
    Koordinatlar iki varyantta AYNIDIR; yalnız sahte görüntünün kaynağı değişir (clean_paired her ikisinde authentic)."""
    raw = np.zeros((len(index), PATCH, PATCH, 3), np.uint8)
    mr = np.zeros_like(raw)
    groups = [(nid, g.index.values) for nid, g in index.groupby("new_id", sort=False)]

    def work(item):
        nid, pos = item
        f = D.pad_to_min(D.read_rgb(D.img_path(nid, "train")), PATCH)
        rp = REPAIR_DIR / f"{nid}.png"
        has_rep = rp.exists()
        fr = D.pad_to_min(D.read_rgb(rp), PATCH) if has_rep else f
        sub = index.loc[pos]
        a = None
        if (sub.patch_type == "clean_paired").any():
            a = D.pad_to_min(D.read_rgb(D.auth_path(split_tab.loc[nid, "pair_name"], "train")), PATCH)
        for p, r in zip(pos, sub.itertuples(index=False)):
            if r.patch_type == "clean_paired":
                c = D.crop(a, int(r.x0), int(r.y0), PATCH)
                raw[p] = c; mr[p] = c
            else:
                raw[p] = D.crop(f, int(r.x0), int(r.y0), PATCH)
                mr[p] = D.crop(fr, int(r.x0), int(r.y0), PATCH)
        return int(has_rep)

    with ThreadPoolExecutor(threads) as ex:
        n_rep = sum(ex.map(work, groups))
    return raw, mr, n_rep


def cache_paths(split: str, tag: str = ""):
    sfx = f"_{tag}" if tag else ""
    return (RES / f"patch_index_{split}{sfx}.csv", CACHE / f"patches_{split}_raw{sfx}.npy", CACHE / f"patches_{split}_mr{sfx}.npy")


def get_patches(split: str, variant: str, tag: str = ""):
    """(index_df, patches uint8) — build_patches_p4.py tarafından üretilmiş önbellekten."""
    idx_p, raw_p, mr_p = cache_paths(split, tag)
    idx = pd.read_csv(idx_p, dtype={"new_id": str})
    arr = np.load(mr_p if variant == "mr" else raw_p)
    assert len(idx) == len(arr), (len(idx), arr.shape)
    return idx, arr


# ---------------------------------------------------------------- model
def build_resnet(pretrained: bool = True) -> nn.Module:
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(512, 1)
    return m


def ckpt_of(stem: str) -> Path:
    """Faz 4 ağırlıkları models/phase4/, Faz 1 ağırlıkları (resnet18_*_best) models/ altında."""
    p = MODELS / f"{stem}.pt"
    return p if p.exists() else MODELS_P1 / f"{stem}.pt"


def load_resnet(stem: str, device) -> nn.Module:
    m = build_resnet(pretrained=False)
    m.load_state_dict(torch.load(ckpt_of(stem), map_location="cpu"))
    return m.to(device).eval()


# ---------------------------------------------------------------- tam görüntü kayan pencere (uygulama protokolü)
def windows(H, W, win=OUTSZ, stride=STRIDE):
    ys = list(range(0, max(H - win, 0) + 1, stride)); xs = list(range(0, max(W - win, 0) + 1, stride))
    if ys[-1] + win < H: ys.append(H - win)
    if xs[-1] + win < W: xs.append(W - win)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def score_image(models: List[nn.Module], rgb_u8: np.ndarray, device, batch: int = 256):
    """Her model için patch skorları (N,) ve pencere koordinatları (N,2 [y,x]). app._predict_resnet ile aynı hazırlık."""
    h, w = rgb_u8.shape[:2]
    ph, pw = max(0, OUTSZ - h), max(0, OUTSZ - w)
    img = cv2.copyMakeBorder(rgb_u8, 0, ph, 0, pw, cv2.BORDER_REPLICATE) if (ph or pw) else rgb_u8
    H, W = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img)).to(device).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(D.IMAGENET_MEAN, device=device).view(3, 1, 1); std = torch.tensor(D.IMAGENET_STD, device=device).view(3, 1, 1)
    x = (x - mean) / std
    wins = windows(H, W)
    outs = [[] for _ in models]
    for i in range(0, len(wins), batch):
        xb = torch.stack([x[:, y:y + OUTSZ, xx:xx + OUTSZ] for (y, xx) in wins[i:i + batch]])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            for k, m in enumerate(models):
                outs[k].append(torch.sigmoid(m(xb).float()).squeeze(1).cpu().numpy())
    return [np.concatenate(o).astype(np.float32) for o in outs], np.asarray(wins, np.int32)
