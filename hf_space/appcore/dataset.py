"""appcore.dataset — veri seti dizini, filtreleme ve örnek yükleme."""
from __future__ import annotations
from ._common import *  # noqa: F401,F403 — ortak içe aktarmalar
from .config import (  # noqa: F401
    AUDIT, DATA, META, ROOT
)

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
