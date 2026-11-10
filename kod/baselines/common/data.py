"""
common/data.py — AIForge-Doc v1 için ortak veri katmanı (TR).

İçerik:
  * Yol sabitleri ve splits.json / samples_forged.csv yükleyicileri
  * Görüntü/maske okuma (RAM önbellekli; büyük SROIE PNG'leri bir kez decode edilir)
  * Patch index üretimi (A deneyi): tampered / clean-same-image / clean-paired-authentic 256×256
  * PatchArrayDataset: önceden kesilmiş patch dizisi üzerinden sınıflandırma örnekleri
  * SegCropDataset: 512×512 natif çözünürlüklü crop'lar (eğitimde rastgele, val'de sabit index)
  * Frekans kanalları (D2 varyantı): FFT Gaussian yüksek-geçiren (D0=20) + 3×3 Laplacian artığı

Not: Maske GT olarak HER ZAMAN masks/<id>.png dosyası kullanılır (metadata bbox_xyxy değil).
Test setine (TestingSet/) hiçbir fonksiyon erişmez.
"""
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

# ---------------------------------------------------------------- yollar
import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
PROJECT_ROOT = Path(str(_ROOT))
DATA_ROOT = PROJECT_ROOT / "data" / "AIForge-Doc-v1"
TRAIN_DIR = DATA_ROOT / "TrainingSet"
AUDIT_DIR = PROJECT_ROOT / "dataset_audit"
BASELINE_DIR = PROJECT_ROOT / "baselines"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = BASELINE_DIR / "results"
FIG_DIR = BASELINE_DIR / "figures"
LOG_DIR = BASELINE_DIR / "logs"
CACHE_DIR = BASELINE_DIR / "cache"

AREA_BINS = ["<1%", "1-5%", "5-20%", ">20%"]
TOOL_LABEL = {"gemini-nano": "gemini", "qwen-inpaint": "ideogram"}


def ensure_dirs():
    for d in (RESULTS_DIR, FIG_DIR, LOG_DIR, CACHE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_splits(path: Optional[Path] = None) -> Dict[str, List[str]]:
    path = path or (AUDIT_DIR / "splits.json")
    with open(path, "r", encoding="utf-8") as f:
        sp = json.load(f)
    return {k: list(v) for k, v in sp.items() if isinstance(v, list) and k != "id_convention"}


def load_samples_forged(path: Optional[Path] = None) -> pd.DataFrame:
    """samples_forged.csv → yalnızca TrainingSet satırları, new_id 9 haneli string indeks."""
    path = path or (AUDIT_DIR / "samples_forged.csv")
    df = pd.read_csv(path, dtype={"new_id": str}, low_memory=False)
    df = df[df["split"] == "training"].copy()
    df["new_id"] = df["new_id"].astype(int).map(lambda x: f"{x:09d}")
    df["tool"] = df["assigned_tool"].map(TOOL_LABEL).fillna(df["assigned_tool"])
    df = df.set_index("new_id", drop=False)
    return df


def split_table(split_name: str, splits=None, samples=None) -> pd.DataFrame:
    splits = splits or load_splits()
    samples = samples if samples is not None else load_samples_forged()
    ids = [i for i in splits[split_name] if i in samples.index]
    return samples.loc[ids]


# ---------------------------------------------------------------- okuma / önbellek
def img_path(new_id: str) -> Path:
    return TRAIN_DIR / "images" / f"{new_id}.png"


def mask_path(new_id: str) -> Path:
    return TRAIN_DIR / "masks" / f"{new_id}.png"


def auth_path(pair_name: str) -> Path:
    return TRAIN_DIR / "authentic" / pair_name


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


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) — x1/y1 dahil değil. Maske boşsa (0,0,0,0)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class ImageCache:
    """Decode edilmiş RGB/maske dizilerini RAM'de tutar (num_workers=0 ile kullanılır)."""

    def __init__(self):
        self.imgs: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}
        self.auth: Dict[str, np.ndarray] = {}

    def image(self, new_id: str) -> np.ndarray:
        if new_id not in self.imgs:
            self.imgs[new_id] = read_rgb(img_path(new_id))
        return self.imgs[new_id]

    def mask(self, new_id: str) -> np.ndarray:
        if new_id not in self.masks:
            self.masks[new_id] = read_mask(mask_path(new_id))
        return self.masks[new_id]

    def authentic(self, pair_name: str) -> np.ndarray:
        if pair_name not in self.auth:
            self.auth[pair_name] = read_rgb(auth_path(pair_name))
        return self.auth[pair_name]

    def nbytes(self) -> int:
        return sum(a.nbytes for d in (self.imgs, self.masks, self.auth) for a in d.values())


# ---------------------------------------------------------------- crop yardımcıları
def pad_to_min(img: np.ndarray, size: int, value=None) -> np.ndarray:
    """Görüntü/maske crop boyutundan küçükse sağ/alt kenardan doldur (görüntü: kenar kopyası, maske: 0)."""
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph == 0 and pw == 0:
        return img
    if value is None:
        return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=value)


def crop(img: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    return img[y0:y0 + size, x0:x0 + size]


def sample_bbox_window(bbox, W, H, size, rng, jitter=32) -> Tuple[int, int]:
    """bbox merkezli size×size pencere (+-jitter). bbox pencereden büyükse bbox içinde rastgele pencere."""
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
    """Maske bbox'ı ile kesişmeyen rastgele size×size pencere; bulunamazsa None."""
    x0b, y0b, x1b, y1b = bbox
    if W < size or H < size:
        return None
    for _ in range(max_try):
        x0 = int(rng.integers(0, W - size + 1))
        y0 = int(rng.integers(0, H - size + 1))
        if x0 + size <= x0b or x0 >= x1b or y0 + size <= y0b or y0 >= y1b:
            return x0, y0
    return None


def sample_random_window(W, H, size, rng) -> Tuple[int, int]:
    x0 = int(rng.integers(0, max(1, W - size + 1)))
    y0 = int(rng.integers(0, max(1, H - size + 1)))
    return x0, y0


# ---------------------------------------------------------------- A) patch index
def build_patch_index(split_name: str, n_tampered: int, seed: int, patch: int = 256,
                      jitter: int = 32, cache: Optional[ImageCache] = None,
                      samples: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Her forged görüntü için:
      - n_tampered adet 'tampered' patch (maske bbox merkezli, farklı jitter)
      - 1 adet 'clean_same'  (aynı forged görüntüde maskeyle kesişmeyen)
      - 1 adet 'clean_paired' (eşleşen authentic görüntüde (i) ile aynı koordinat; pair_accepted & aynı boyut)
    Döndürülen tablo: new_id, src_file, x0, y0, label, patch_type, tool, source, area_bin, W, H
    """
    rng = np.random.default_rng(seed)
    tab = split_table(split_name, samples=samples)
    cache = cache or ImageCache()
    rows = []
    for new_id, r in tab.iterrows():
        m = cache.mask(new_id)
        H, W = m.shape
        bbox = mask_bbox(m)
        base = dict(new_id=new_id, tool=r["tool"], source=r["source_dataset"], area_bin=r["area_bin"], W=W, H=H)
        first_xy = None
        for k in range(n_tampered):
            x0, y0 = sample_bbox_window(bbox, W, H, patch, rng, jitter)
            if first_xy is None:
                first_xy = (x0, y0)
            rows.append(dict(base, src_file=str(img_path(new_id)), x0=x0, y0=y0, label=1, patch_type="tampered", rep=k))
        xy = sample_clean_window(bbox, W, H, patch, rng)
        if xy is not None:
            rows.append(dict(base, src_file=str(img_path(new_id)), x0=xy[0], y0=xy[1], label=0, patch_type="clean_same", rep=0))
        if bool(r["pair_accepted"]) and bool(r["pair_same_size"]):
            rows.append(dict(base, src_file=str(auth_path(r["pair_name"])), x0=first_xy[0], y0=first_xy[1], label=0,
                             patch_type="clean_paired", rep=0))
    cols = ["new_id", "src_file", "x0", "y0", "label", "patch_type", "rep", "tool", "source", "area_bin", "W", "H"]
    return pd.DataFrame(rows)[cols]


def extract_patches(index: pd.DataFrame, patch: int = 256, cache: Optional[ImageCache] = None) -> np.ndarray:
    """Index tablosundaki tüm patch'leri (N, patch, patch, 3) uint8 dizisine çıkarır."""
    cache = cache or ImageCache()
    out = np.empty((len(index), patch, patch, 3), dtype=np.uint8)
    for i, r in enumerate(index.itertuples(index=False)):
        if r.patch_type == "clean_paired":
            img = cache.authentic(Path(r.src_file).name)
        else:
            img = cache.image(r.new_id)
        img = pad_to_min(img, patch)
        c = crop(img, int(r.x0), int(r.y0), patch)
        out[i] = c
    return out


def patch_cache_path(split_name: str, seed: int) -> Path:
    return CACHE_DIR / f"patches_{split_name}_seed{seed}.npz"


def get_or_build_patches(split_name: str, n_tampered: int, seed: int, patch: int = 256,
                         cache: Optional[ImageCache] = None, force: bool = False):
    """Patch index + diziyi (npz) önbellekten yükle ya da üret. Dönüş: (index_df, patches uint8)."""
    ensure_dirs()
    p = patch_cache_path(split_name, seed)
    idx_csv = RESULTS_DIR / f"patch_index_{'train' if 'train' in split_name else 'val'}.csv"
    if p.exists() and idx_csv.exists() and not force:
        z = np.load(p)
        idx = pd.read_csv(idx_csv, dtype={"new_id": str})
        if len(idx) == z["patches"].shape[0]:
            return idx, z["patches"]
    idx = build_patch_index(split_name, n_tampered, seed, patch, cache=cache)
    arr = extract_patches(idx, patch, cache=cache)
    idx.to_csv(idx_csv, index=False)
    np.savez_compressed(p, patches=arr)
    return idx, arr


# ---------------------------------------------------------------- torch dataset'leri
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # torch olmadan da (klasik ML) import edilebilsin
    torch = None
    Dataset = object

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def photometric_jitter(img_u8: np.ndarray, rng: np.random.Generator, b=0.08, c=0.08) -> np.ndarray:
    """Küçük parlaklık/kontrast jitter'ı (metin için flip YOK)."""
    x = img_u8.astype(np.float32) / 255.0
    alpha = 1.0 + rng.uniform(-c, c)
    beta = rng.uniform(-b, b)
    x = np.clip((x - 0.5) * alpha + 0.5 + beta, 0, 1)
    return x


def to_tensor_norm(x01: np.ndarray) -> "torch.Tensor":
    x = (x01 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class PatchArrayDataset(Dataset):
    """Önceden çıkarılmış 256×256 patch'ler → 224 center/random crop (yeniden örnekleme YOK) + jitter."""

    def __init__(self, patches: np.ndarray, labels: np.ndarray, train: bool, out: int = 224, seed: int = 42):
        self.p, self.y, self.train, self.out = patches, labels.astype(np.float32), train, out
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = self.p[i]
        S = img.shape[0]
        if self.train:
            x0, y0 = self.rng.integers(0, S - self.out + 1, size=2)
        else:
            x0 = y0 = (S - self.out) // 2
        img = img[y0:y0 + self.out, x0:x0 + self.out]
        if self.train:
            x = photometric_jitter(img, self.rng)
        else:
            x = img.astype(np.float32) / 255.0
        return to_tensor_norm(x), torch.tensor(self.y[i])


def build_val_crop_index(split_name: str, size: int, seed: int, cache: ImageCache,
                         samples: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Val için sabit crop index'i: görüntü başına 1 bbox-içeren + 1 rastgele crop."""
    rng = np.random.default_rng(seed)
    tab = split_table(split_name, samples=samples)
    rows = []
    for new_id, r in tab.iterrows():
        m = cache.mask(new_id)
        H, W = m.shape
        bbox = mask_bbox(m)
        x0, y0 = sample_bbox_window(bbox, W, H, size, rng, jitter=size // 8)
        rows.append(dict(new_id=new_id, x0=x0, y0=y0, crop_type="bbox", tool=r["tool"], source=r["source_dataset"],
                         area_bin=r["area_bin"], W=W, H=H))
        x0, y0 = sample_random_window(W, H, size, rng)
        rows.append(dict(new_id=new_id, x0=x0, y0=y0, crop_type="random", tool=r["tool"], source=r["source_dataset"],
                         area_bin=r["area_bin"], W=W, H=H))
    return pd.DataFrame(rows)


class SegCropDataset(Dataset):
    """
    Segmentasyon crop'ları (natif çözünürlük, size×size).
      train=True : her __getitem__ rastgele crop; p_bbox olasılıkla bbox-içeren (jitter'lı), aksi hâlde rastgele.
      train=False: index tablosundaki sabit crop'lar.
    Döndürür: img float tensor (3,S,S) ImageNet-normalize, mask float (1,S,S), meta dict.
    """

    def __init__(self, ids: List[str], cache: ImageCache, size: int = 512, train: bool = True,
                 p_bbox: float = 0.7, crops_per_image: int = 2, seed: int = 42,
                 val_index: Optional[pd.DataFrame] = None):
        self.ids, self.cache, self.size, self.train = list(ids), cache, size, train
        self.p_bbox, self.cpi = p_bbox, crops_per_image
        self.rng = np.random.default_rng(seed)
        self.val_index = val_index
        self.bboxes = {i: mask_bbox(cache.mask(i)) for i in self.ids}

    def __len__(self):
        return len(self.val_index) if not self.train else len(self.ids) * self.cpi

    def _get(self, new_id, x0, y0):
        img = pad_to_min(self.cache.image(new_id), self.size)
        m = pad_to_min(self.cache.mask(new_id), self.size, value=0)
        return crop(img, x0, y0, self.size), crop(m, x0, y0, self.size)

    def __getitem__(self, i):
        if self.train:
            new_id = self.ids[i % len(self.ids)]
            m = self.cache.mask(new_id)
            H, W = m.shape
            if self.rng.random() < self.p_bbox:
                x0, y0 = sample_bbox_window(self.bboxes[new_id], W, H, self.size, self.rng, jitter=self.size // 4)
            else:
                x0, y0 = sample_random_window(W, H, self.size, self.rng)
            im, mk = self._get(new_id, x0, y0)
            x = photometric_jitter(im, self.rng)
            meta = dict(new_id=new_id, x0=int(x0), y0=int(y0))
        else:
            r = self.val_index.iloc[i]
            im, mk = self._get(r.new_id, int(r.x0), int(r.y0))
            x = im.astype(np.float32) / 255.0
            meta = dict(new_id=r.new_id, x0=int(r.x0), y0=int(r.y0), crop_type=r.crop_type, tool=r.tool,
                        area_bin=r.area_bin, source=r.source)
        return to_tensor_norm(x), torch.from_numpy(mk[None].astype(np.float32)), meta


def seg_collate(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


# ---------------------------------------------------------------- frekans kanalları (D2)
def denorm_to_gray(x: "torch.Tensor") -> "torch.Tensor":
    """ImageNet-normalize (B,3,H,W) → gri [0,1] (B,1,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    rgb = x * std + mean
    return (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])


def freq_channels(x: "torch.Tensor", d0: float = 20.0) -> "torch.Tensor":
    """
    (B,3,H,W) normalize RGB → (B,2,H,W):
      kanal 0: FFT Gaussian yüksek-geçiren filtre (H = 1 - exp(-D^2 / 2 D0^2), D0=20 frekans örneği) → ters FFT (uzamsal)
      kanal 1: 3×3 Laplacian artığı
    Her kanal örnek başına standartlaştırılır (sıfır ortalama, birim std) — AMP kararlılığı için float32'de hesaplanır.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        g = denorm_to_gray(x.float())
        B, _, H, W = g.shape
        fy = torch.fft.fftfreq(H, device=g.device).view(1, 1, H, 1) * H
        fx = torch.fft.fftfreq(W, device=g.device).view(1, 1, 1, W) * W
        D2 = fy ** 2 + fx ** 2
        Hf = 1.0 - torch.exp(-D2 / (2.0 * d0 ** 2))
        F = torch.fft.fft2(g)
        hp = torch.fft.ifft2(F * Hf).real
        lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=g.device).view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(torch.nn.functional.pad(g, (1, 1, 1, 1), mode="replicate"), lap_k)
        out = torch.cat([hp, lap], dim=1)
        mu = out.mean(dim=(2, 3), keepdim=True)
        sd = out.std(dim=(2, 3), keepdim=True) + 1e-6
        return (out - mu) / sd


# ---------------------------------------------------------------- dayanıklılık dönüşümleri (E)
def degrade_jpeg(img_u8: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def degrade_rescale(img_u8: np.ndarray, factor: float = 0.75) -> np.ndarray:
    h, w = img_u8.shape[:2]
    small = cv2.resize(img_u8, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
