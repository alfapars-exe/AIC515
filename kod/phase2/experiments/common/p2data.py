"""
common/p2data.py — Faz 2 ortak veri katmanı (TR).

baselines/common/data.py'den türetildi; farklar:
  * GT varyantı seçilebilir: 'bbox' (masks/<id>.png), 'pixel_eps8_morph', 'glyph_v2', ... (phase2/gt/<variant>/)
  * Görüntüler DİSKTEN okunur (RAM önbellek yok; 2762 görüntü ≈ 26 GB). Decode darboğazı için thread havuzlu
    prefetch (ThreadedLoader) — sandbox'ta multiprocessing yok, cv2.imread GIL bırakır.
  * Eğitim crop örnekleme: %70 'GT-pozitif içeren' (eğitim GT'sinden rastgele pozitif piksel merkezli, ±size/4 jitter),
    %30 rastgele. Görüntü başına epoch'ta crops_per_image crop AYNI decode'dan alınır.
  * Test split (TestingSet/) desteği: split='test' → TestingSet/images|masks ve gt/<variant>/test/<id>.png
  * id çakışması: TrainingSet ve TestingSet new_id'leri aynı uzayı paylaşır → tüm çıktı yolları <split>/<id>.png
"""
from __future__ import annotations
import os, json, math, threading, queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import cv2

cv2.setNumThreads(2)

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
PROJECT_ROOT = Path(str(_ROOT))
DATA_ROOT = PROJECT_ROOT / "data" / "AIForge-Doc-v1"
TRAIN_DIR = DATA_ROOT / "TrainingSet"
TEST_DIR = DATA_ROOT / "TestingSet"
AUDIT_DIR = PROJECT_ROOT / "dataset_audit"
P2_DIR = PROJECT_ROOT / "phase2"
GT_DIR = P2_DIR / "gt"
CTRL_DIR = P2_DIR / "controls"
EXP_DIR = P2_DIR / "experiments"
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figures"
LOG_DIR = EXP_DIR / "logs"
PRED_DIR = EXP_DIR / "preds"
MODELS_DIR = PROJECT_ROOT / "models" / "phase2"

AREA_BINS = ["<1%", "1-5%", "5-20%", ">20%"]
TOOL_LABEL = {"gemini-nano": "gemini", "qwen-inpaint": "ideogram"}
GT_VARIANTS = ["bbox", "pixel_eps4", "pixel_eps8", "pixel_eps16", "pixel_eps8_morph", "glyph", "glyph_v2", "bbox_and_eps8"]
EVAL_GTS = ["bbox", "pixel_eps8_morph", "glyph_v2"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dirs():
    for d in (RESULTS_DIR, FIG_DIR, LOG_DIR, PRED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- tablolar
def load_splits() -> Dict[str, List[str]]:
    with open(AUDIT_DIR / "splits.json", "r", encoding="utf-8") as f:
        sp = json.load(f)
    return {k: list(v) for k, v in sp.items() if isinstance(v, list)}


def load_excluded() -> List[str]:
    with open(EXP_DIR / "excluded_ids.json", "r", encoding="utf-8") as f:
        return list(json.load(f)["all_excluded_ids"])


def load_samples(split_kind: str) -> pd.DataFrame:
    """samples_forged.csv → 'training' ya da 'testing' satırları; new_id 9 haneli string indeks."""
    df = pd.read_csv(AUDIT_DIR / "samples_forged.csv", dtype={"new_id": str}, low_memory=False)
    df = df[df["split"] == split_kind].copy()
    df["new_id"] = df["new_id"].astype(int).map(lambda x: f"{x:09d}")
    df["tool"] = df["assigned_tool"].map(TOOL_LABEL).fillna(df["assigned_tool"])
    df["source"] = df["source_dataset"]
    df = df.set_index("new_id", drop=False)
    return df


def split_table(split: str) -> pd.DataFrame:
    """split ∈ {train, val, test}: id listesi + metadata (tool, source, area_bin, group, pair_name, W, H, nn_train_any_dist)."""
    sp = load_splits()
    kind = "testing" if split == "test" else "training"
    tab = load_samples(kind)
    ids = [i for i in sp[split] if i in tab.index]
    t = tab.loc[ids].copy()
    t["split"] = split
    if split == "test":
        d = t["nn_train_any_dist"]
        t["clean_test"] = d.isna() | (d > 20)
    return t


def train_ids(train_filter: Optional[str] = None) -> List[str]:
    t = split_table("train")
    ex = set(load_excluded())
    t = t[~t.index.isin(ex)]
    if train_filter == "gemini_only":
        t = t[t["tool"] == "gemini"]
    return list(t.index)


# ---------------------------------------------------------------- yollar
def set_dir(split: str) -> Path:
    return TEST_DIR if split == "test" else TRAIN_DIR


def img_path(new_id: str, split: str = "train") -> Path:
    return set_dir(split) / "images" / f"{new_id}.png"


def bbox_mask_path(new_id: str, split: str = "train") -> Path:
    return set_dir(split) / "masks" / f"{new_id}.png"


def auth_path(pair_name: str, split: str = "train") -> Path:
    return set_dir(split) / "authentic" / pair_name


def gt_path(new_id: str, variant: str, split: str = "train") -> Optional[Path]:
    """GT maskesi yolu. 'bbox' → resmi maske; diğerleri phase2/gt/<variant>/[test/]<id>.png (çifti yoksa None)."""
    if variant == "bbox":
        return bbox_mask_path(new_id, split)
    p = GT_DIR / variant / ("test" if split == "test" else "") / f"{new_id}.png"
    return p if p.exists() else None


def control_img_path(kind: str, new_id: str) -> Path:
    return CTRL_DIR / kind / "images" / f"{new_id}.png"


# ---------------------------------------------------------------- okuma
def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def read_gt(new_id: str, variant: str, split: str = "train", shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    p = gt_path(new_id, variant, split)
    if p is None:
        return None
    m = read_mask(p)
    if shape is not None and m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------- crop yardımcıları
def pad_to_min(img: np.ndarray, size: int, value=None) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph == 0 and pw == 0:
        return img
    if value is None:
        return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=value)


def crop(img: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    return img[y0:y0 + size, x0:x0 + size]


def window_from_center(cx: int, cy: int, W: int, H: int, size: int) -> Tuple[int, int]:
    x0 = int(np.clip(cx - size // 2, 0, max(0, W - size)))
    y0 = int(np.clip(cy - size // 2, 0, max(0, H - size)))
    return x0, y0


def sample_gt_window(gt: np.ndarray, bbox, W, H, size, rng, jitter: int) -> Tuple[int, int]:
    """Eğitim GT'sinden rastgele pozitif piksel merkezli pencere (+-jitter). GT boşsa bbox merkezi; o da boşsa rastgele."""
    pos = np.flatnonzero(gt)
    if len(pos) > 0:
        k = int(rng.integers(0, len(pos)))
        cy, cx = divmod(int(pos[k]), gt.shape[1])
    elif bbox[2] > bbox[0]:
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    else:
        return sample_random_window(W, H, size, rng)
    cx += int(rng.integers(-jitter, jitter + 1)); cy += int(rng.integers(-jitter, jitter + 1))
    return window_from_center(cx, cy, W, H, size)


def sample_random_window(W, H, size, rng) -> Tuple[int, int]:
    x0 = int(rng.integers(0, max(1, W - size + 1)))
    y0 = int(rng.integers(0, max(1, H - size + 1)))
    return x0, y0


def photometric_jitter(img_u8: np.ndarray, rng: np.random.Generator, b=0.08, c=0.08) -> np.ndarray:
    x = img_u8.astype(np.float32) / 255.0
    alpha = 1.0 + rng.uniform(-c, c)
    beta = rng.uniform(-b, b)
    return np.clip((x - 0.5) * alpha + 0.5 + beta, 0, 1)


def norm_chw(x01: np.ndarray) -> np.ndarray:
    x = (x01 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)


# ---------------------------------------------------------------- eğitim örnekleyici (thread prefetch)
class SegTrainSampler:
    """
    Her epoch'ta her eğitim görüntüsü bir kez decode edilir ve crops_per_image crop üretilir
    (%p_gt GT-pozitif merkezli jitter'lı, aksi rastgele). Görüntü sırası epoch başına karıştırılır.
    Batch = images_per_batch görüntü × crops_per_image crop.
    """

    def __init__(self, ids: List[str], gt_variant: str, size: int = 512, crops_per_image: int = 2,
                 p_gt: float = 0.7, images_per_batch: int = 8, seed: int = 42, n_threads: int = 8, prefetch: int = 6):
        self.ids, self.gt_variant, self.size, self.cpi = list(ids), gt_variant, size, crops_per_image
        self.p_gt, self.ipb, self.n_threads, self.prefetch = p_gt, images_per_batch, n_threads, prefetch
        self.rng = np.random.default_rng(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.ids) // self.ipb

    def _item(self, new_id: str, seed: int):
        rng = np.random.default_rng(seed)
        img = read_rgb(img_path(new_id, "train"))
        H, W = img.shape[:2]
        gt = read_gt(new_id, self.gt_variant, "train", shape=(H, W))
        if gt is None:
            gt = np.zeros((H, W), np.uint8)
        bbox = mask_bbox(read_mask(bbox_mask_path(new_id, "train"))) if self.gt_variant != "bbox" else mask_bbox(gt)
        imgp = pad_to_min(img, self.size); gtp = pad_to_min(gt, self.size, value=0)
        xs, ys = [], []
        for _ in range(self.cpi):
            if rng.random() < self.p_gt:
                x0, y0 = sample_gt_window(gt, bbox, W, H, self.size, rng, jitter=self.size // 4)
            else:
                x0, y0 = sample_random_window(W, H, self.size, rng)
            im = crop(imgp, x0, y0, self.size); mk = crop(gtp, x0, y0, self.size)
            xs.append(np.ascontiguousarray(im)); ys.append(mk[None])
        # uint8 HWC crop + uint8 maske; fotometrik jitter (alpha, beta) GPU'da uygulanır
        ab = np.stack([[1.0 + rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)] for _ in range(self.cpi)]).astype(np.float32)
        return np.stack(xs), np.stack(ys), ab

    def __iter__(self):
        self.epoch += 1
        order = self.rng.permutation(len(self.ids))
        seeds = self.rng.integers(0, 2 ** 31 - 1, size=len(self.ids))
        batches = [order[i:i + self.ipb] for i in range(0, len(order) - self.ipb + 1, self.ipb)]
        ex = ThreadPoolExecutor(self.n_threads)
        try:
            pending = {}
            def _submit(k):
                pending[k] = [ex.submit(self._item, self.ids[j], int(seeds[j])) for j in batches[k]]
            for k in range(min(self.prefetch, len(batches))):
                _submit(k)
            for k in range(len(batches)):
                fl = pending.pop(k)
                if k + self.prefetch < len(batches):
                    _submit(k + self.prefetch)
                outs = [f.result() for f in fl]
                yield (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]),
                       np.concatenate([o[2] for o in outs]))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)


def build_val_crop_index(ids: List[str], gt_variant: str, size: int, seed: int) -> pd.DataFrame:
    """Val için sabit crop index'i: görüntü başına 1 GT-pozitif merkezli + 1 rastgele crop (eğitim GT'sine göre)."""
    rng = np.random.default_rng(seed)
    rows = []
    for new_id in ids:
        bm = read_mask(bbox_mask_path(new_id, "train")); H, W = bm.shape
        gt = bm if gt_variant == "bbox" else read_gt(new_id, gt_variant, "train", shape=(H, W))
        if gt is None:
            gt = np.zeros((H, W), np.uint8)
        x0, y0 = sample_gt_window(gt, mask_bbox(bm), W, H, size, rng, jitter=size // 8)
        rows.append(dict(new_id=new_id, x0=x0, y0=y0, crop_type="gt"))
        x0, y0 = sample_random_window(W, H, size, rng)
        rows.append(dict(new_id=new_id, x0=x0, y0=y0, crop_type="random"))
    return pd.DataFrame(rows)


class ValCropLoader:
    """Sabit val crop'larını (index) thread havuzuyla okur; (x[B,3,S,S], y[B,1,S,S], metas) batch'leri üretir."""

    def __init__(self, index: pd.DataFrame, gt_variant: str, size: int, batch: int = 16, n_threads: int = 8):
        self.index, self.gt_variant, self.size, self.batch, self.n_threads = index, gt_variant, size, batch, n_threads
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _item(self, r):
        img = read_rgb(img_path(r.new_id, "train")); H, W = img.shape[:2]
        gt = read_gt(r.new_id, self.gt_variant, "train", shape=(H, W))
        if gt is None:
            gt = np.zeros((H, W), np.uint8)
        im = crop(pad_to_min(img, self.size), int(r.x0), int(r.y0), self.size)
        mk = crop(pad_to_min(gt, self.size, value=0), int(r.x0), int(r.y0), self.size)
        return np.ascontiguousarray(im), mk[None]

    def __iter__(self):
        rows = list(self.index.itertuples(index=False))
        with ThreadPoolExecutor(self.n_threads) as ex:
            for i in range(0, len(rows), self.batch):
                chunk = rows[i:i + self.batch]
                outs = list(ex.map(self._item, chunk))
                yield np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs]), chunk


# ---------------------------------------------------------------- dayanıklılık dönüşümleri
def degrade_jpeg(img_u8: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade_rescale(img_u8: np.ndarray, factor: float = 0.75) -> np.ndarray:
    h, w = img_u8.shape[:2]
    small = cv2.resize(img_u8, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_blur(img_u8: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img_u8, (0, 0), sigma)


DEGRADATIONS = {"jpeg75": degrade_jpeg, "rescale075": degrade_rescale, "blur1": degrade_blur}


def gpu_normalize(x_u8: "torch.Tensor", ab: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    """(B,S,S,3) uint8 GPU tensörü → ImageNet-normalize (B,3,S,S) float32 (channels_last). ab: (B,2) [alpha, beta] jitter."""
    import torch
    x = x_u8.permute(0, 3, 1, 2).float() / 255.0
    if ab is not None:
        a = ab[:, 0].view(-1, 1, 1, 1); b = ab[:, 1].view(-1, 1, 1, 1)
        x = ((x - 0.5) * a + 0.5 + b).clamp_(0, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def set_seed(seed: int = 42):
    import random, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
