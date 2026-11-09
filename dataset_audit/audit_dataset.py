#!/usr/bin/env python
"""
AIForge-Doc v1 — 14-dimensional dataset audit + group-aware split proposal.

Usage:
    python audit_dataset.py --data "<path>/AIForge-Doc-v1" --out "<path>/dataset_audit" [--workers 24]
                            [--dup-threshold 20] [--val-frac 0.15] [--seed 42] [--max-pairs 0]

Every number in audit_stats.json is computed from the local files; nothing is taken from the
dataset card except where explicitly stored under "card_claims" for cross-checking.

Outputs (in --out):
    audit_stats.json          all computed statistics
    samples_forged.csv        per-forged-image table (mask stats, hashes, pairing, pixel-diff)
    samples_authentic.csv     per-authentic-image table (size, hashes)
    phash_pairs.csv           near-duplicate pairs (hamming <= dup-threshold)
    splits.json               train/val/test ids + smoke subsets
    fig_*.png                 figures
"""
import argparse, hashlib, json, os, re, sys, time, random, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
PHASH_SIZE = 16   # 256-bit pHash

# --------------------------------------------------------------------------------------
# Workers (module level so they pickle under Windows 'spawn')
# --------------------------------------------------------------------------------------
def _hashes(img):
    ph = imagehash.phash(img, hash_size=PHASH_SIZE)
    dh = imagehash.dhash(img, hash_size=8)
    return str(ph), str(dh)

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _bbox_of(binary):
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # inclusive

def _rect_iou(a, b):
    """IoU of two inclusive [x1,y1,x2,y2] rectangles."""
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0

def forged_worker(rec):
    """rec: dict with keys image (abs path), mask (abs path), new_id, split, bbox_xyxy."""
    out = {"key": rec["key"]}
    try:
        with Image.open(rec["image"]) as im:
            out["img_mode"], out["img_format"] = im.mode, im.format
            out["W"], out["H"] = im.size
            im_rgb = im.convert("RGB")
            out["phash"], out["dhash"] = _hashes(im_rgb)
        out["img_md5"] = _md5(rec["image"])
        out["img_bytes"] = os.path.getsize(rec["image"])
    except Exception as e:
        out["img_error"] = repr(e)
        return out
    try:
        with Image.open(rec["mask"]) as mk:
            out["mask_mode"], out["mask_format"] = mk.mode, mk.format
            out["mW"], out["mH"] = mk.size
            arr = np.asarray(mk)
        if arr.ndim == 3:
            out["mask_channels_equal"] = bool(np.all(arr[..., 0:1] == arr[..., :3]))
            arr = arr[..., 0]
        else:
            out["mask_channels_equal"] = True
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        vals, cnts = np.unique(arr, return_counts=True)
        out["mask_unique"] = ";".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
        out["mask_n_unique"] = int(len(vals))
        out["mask_is_binary_0_255"] = bool(set(vals.tolist()) <= {0, 255})
        out["mask_is_binary_0_1"] = bool(set(vals.tolist()) <= {0, 1})
        tam = arr > 127
        area = int(tam.sum())
        out["tamper_px"] = area
        out["tamper_ratio"] = area / float(arr.shape[0] * arr.shape[1])
        bb = _bbox_of(tam)
        out["mask_bbox"] = json.dumps(bb)
        if bb is not None:
            bb_area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
            out["rectangularity"] = area / bb_area
            out["mask_bbox_w"], out["mask_bbox_h"] = bb[2] - bb[0] + 1, bb[3] - bb[1] + 1
        else:
            out["rectangularity"] = float("nan")
        lab, n = ndimage.label(tam, structure=np.ones((3, 3)))
        out["n_components"] = int(n)
        if n > 0:
            sizes = ndimage.sum(tam, lab, index=np.arange(1, n + 1))
            out["min_component_px"] = int(np.min(sizes))
            out["max_component_px"] = int(np.max(sizes))
        else:
            out["min_component_px"] = 0
            out["max_component_px"] = 0
        # metadata bbox vs mask bbox
        mb = rec.get("bbox_xyxy")
        if mb is not None and bb is not None:
            mb_incl = [mb[0], mb[1], mb[2], mb[3]]            # if x2,y2 inclusive
            mb_excl = [mb[0], mb[1], mb[2] - 1, mb[3] - 1]    # if x2,y2 exclusive
            out["iou_meta_incl"] = _rect_iou(bb, mb_incl)
            out["iou_meta_excl"] = _rect_iou(bb, mb_excl)
            out["meta_bbox_exact_incl"] = bool(bb == mb_incl)
            out["meta_bbox_exact_excl"] = bool(bb == mb_excl)
    except Exception as e:
        out["mask_error"] = repr(e)
    return out

def authentic_worker(rec):
    out = {"key": rec["key"], "path": rec["path"]}
    try:
        with Image.open(rec["path"]) as im:
            out["img_mode"], out["img_format"] = im.mode, im.format
            out["W"], out["H"] = im.size
            out["phash"], out["dhash"] = _hashes(im.convert("RGB"))
        out["img_md5"] = _md5(rec["path"])
        out["img_bytes"] = os.path.getsize(rec["path"])
    except Exception as e:
        out["img_error"] = repr(e)
    return out

def diff_worker(rec):
    """Pixel-change analysis for a forged/authentic pair of equal size.
    rec: forged (path), authentic (path), mask (path), bbox (metadata, [x1,y1,x2,y2]), key"""
    out = {"key": rec["key"]}
    try:
        f = np.asarray(Image.open(rec["forged"]).convert("RGB")).astype(np.int16)
        a = np.asarray(Image.open(rec["authentic"]).convert("RGB")).astype(np.int16)
        if f.shape != a.shape:
            out["diff_error"] = f"shape mismatch {f.shape} vs {a.shape}"
            return out
        H, W = f.shape[:2]
        D = np.abs(f - a).max(axis=2)
        mk = np.asarray(Image.open(rec["mask"]))
        if mk.ndim == 3:
            mk = mk[..., 0]
        mask = mk > 127
        # context window: bbox expanded 50% per side, minimum 100 px (interpretation of the card)
        x1, y1, x2, y2 = rec["bbox"]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        px, py = max(int(0.5 * bw), 100), max(int(0.5 * bh), 100)
        cx1, cy1, cx2, cy2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        ctx = np.zeros_like(mask); ctx[cy1:cy2, cx1:cx2] = True
        out["mask_px"] = int(mask.sum())
        out["ctx_px"] = int(ctx.sum())
        out["img_px"] = int(H * W)
        for eps in (0, 8):
            ch = D > eps
            n_ch = int(ch.sum())
            inside = int((ch & mask).sum())
            outside = n_ch - inside
            out[f"changed_px_eps{eps}"] = n_ch
            out[f"changed_inside_mask_eps{eps}"] = inside
            out[f"changed_outside_mask_eps{eps}"] = outside
            out[f"frac_mask_changed_eps{eps}"] = inside / out["mask_px"] if out["mask_px"] else float("nan")
            out[f"frac_outside_of_changed_eps{eps}"] = outside / n_ch if n_ch else 0.0
            out[f"outside_ratio_img_eps{eps}"] = outside / (H * W)
            out[f"outside_in_ctx_eps{eps}"] = int((ch & ~mask & ctx).sum())
            union = int((ch | mask).sum())
            out[f"iou_diff_mask_eps{eps}"] = inside / union if union else float("nan")
            # bbox of all changed pixels
            bb = _bbox_of(ch)
            out[f"changed_bbox_eps{eps}"] = json.dumps(bb)
        out["D_max"] = int(D.max())
        out["D_mean_outside_mask"] = float(D[~mask].mean()) if (~mask).any() else 0.0
        out["D_p99_outside_mask"] = float(np.percentile(D[~mask], 99)) if (~mask).any() else 0.0
    except Exception as e:
        out["diff_error"] = repr(e)
    return out

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def hex_to_bits(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))

def pairwise_hamming(bits_a, bits_b, chunk=1024):
    """bits_*: (n, nbits) uint8 arrays of 0/1 -> (na, nb) int16 hamming distances."""
    pa = np.packbits(bits_a, axis=1); pb = np.packbits(bits_b, axis=1)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = np.empty((pa.shape[0], pb.shape[0]), dtype=np.int16)
    for i in range(0, pa.shape[0], chunk):
        x = pa[i:i + chunk, None, :] ^ pb[None, :, :]
        out[i:i + chunk] = lut[x].sum(axis=2)
    return out

class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra

def area_bin(r):
    if r < 0.01: return "<1%"
    if r < 0.05: return "1-5%"
    if r < 0.20: return "5-20%"
    return ">20%"

def source_from_authentic_name(name):
    stem = os.path.splitext(name)[0]
    if stem.startswith("cord_"): return "cord"
    if stem.startswith("xfund_"): return "xfund"
    if stem.startswith("wildreceipt_"): return "wildreceipt"
    if re.match(r"^X\d+", stem): return "sroie"
    if re.fullmatch(r"[0-9a-f]{40}", stem) or re.fullmatch(r"\d+", stem): return "wildreceipt"   # WildReceipt uses sha1 / numeric names
    return "unknown"

def nan_to_none(o):
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    if isinstance(o, dict): return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [nan_to_none(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return nan_to_none(float(o))
    if isinstance(o, np.ndarray): return nan_to_none(o.tolist())
    return o

def describe(series):
    s = pd.Series(series).dropna()
    if len(s) == 0: return {}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(s)), "min": float(s.min()), "p05": float(q[0.05]), "p25": float(q[0.25]),
            "median": float(q[0.5]), "p75": float(q[0.75]), "p95": float(q[0.95]),
            "max": float(s.max()), "mean": float(s.mean())}

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv=None, style_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--dup-threshold", type=int, default=20, help="pHash-256 hamming threshold for near-duplicates")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=0, help="limit pixel-diff pairs (0 = all)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    DATA, OUT = args.data, args.out
    os.makedirs(OUT, exist_ok=True)
    stats = {"data_path": DATA, "run_started": time.strftime("%Y-%m-%d %H:%M:%S"), "params": vars(args)}

    # ---------------- Dim 1: directory structure & formats ----------------
    tree = {}
    total_bytes = 0
    ext_counter = collections.Counter()
    for root, dirs, files in os.walk(DATA):
        if ".cache" in root: continue
        rel = os.path.relpath(root, DATA).replace("\\", "/")
        if files:
            b = sum(os.path.getsize(os.path.join(root, f)) for f in files)
            tree[rel] = {"n_files": len(files), "bytes": b}
            total_bytes += b
            for f in files: ext_counter[os.path.splitext(f)[1].lower()] += 1
    stats["dim01_structure"] = {"tree": tree, "total_bytes": total_bytes, "total_GB": total_bytes / 1e9,
                                "extensions": dict(ext_counter),
                                "top_level_files": sorted(f for f in os.listdir(DATA) if os.path.isfile(os.path.join(DATA, f)))}
    print(f"[dim01] {sum(t['n_files'] for t in tree.values())} files, {total_bytes/1e9:.2f} GB")

    # ---------------- metadata ----------------
    md_path = os.path.join(DATA, "metadata.jsonl")
    md = pd.DataFrame([json.loads(l) for l in open(md_path, encoding="utf-8") if l.strip()])
    md["key"] = md["split"].str[:2] + "_" + md["new_id"]      # tr_000000001 / te_000000001
    stats["metadata_columns"] = list(md.columns)
    stats["metadata_rows"] = int(len(md))

    # ---------------- Dim 2: image-mask matching ----------------
    dim02 = {}
    for split, folder in (("training", "TrainingSet"), ("testing", "TestingSet")):
        imgs = set(os.listdir(os.path.join(DATA, folder, "images")))
        msks = set(os.listdir(os.path.join(DATA, folder, "masks")))
        auth = set(os.listdir(os.path.join(DATA, folder, "authentic"))) if os.path.isdir(os.path.join(DATA, folder, "authentic")) else set()
        meta_ids = set(md.loc[md.split == split, "new_id"] + ".png")
        dim02[split] = {
            "n_images": len(imgs), "n_masks": len(msks), "n_authentic": len(auth),
            "images_without_mask": sorted(imgs - msks)[:50], "n_images_without_mask": len(imgs - msks),
            "masks_without_image": sorted(msks - imgs)[:50], "n_masks_without_image": len(msks - imgs),
            "n_images_without_metadata": len(imgs - meta_ids), "n_metadata_without_image": len(meta_ids - imgs),
            "n_metadata_rows": int((md.split == split).sum()),
            "n_duplicate_new_id_in_metadata": int(md.loc[md.split == split, "new_id"].duplicated().sum()),
            "authentic_name_patterns": dict(collections.Counter(re.sub(r"\d+", "#", a) for a in auth)),
        }
    dim02["n_spec_id_unique"] = int(md.spec_id.nunique())
    dim02["n_spec_id_duplicated_rows"] = int(md.spec_id.duplicated(keep=False).sum())
    stats["dim02_matching"] = dim02
    print("[dim02]", {k: (v["n_images"], v["n_masks"], v["n_authentic"]) for k, v in dim02.items() if isinstance(v, dict)})

    # ---------------- per-image workers ----------------
    recs = []
    for r in md.itertuples():
        folder = "TrainingSet" if r.split == "training" else "TestingSet"
        recs.append({"key": r.key, "image": os.path.join(DATA, folder, "images", r.new_id + ".png"),
                     "mask": os.path.join(DATA, folder, "masks", r.new_id + ".png"), "bbox_xyxy": r.bbox_xyxy})
    auth_recs = []
    for split, folder in (("training", "TrainingSet"), ("testing", "TestingSet")):
        d = os.path.join(DATA, folder, "authentic")
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                auth_recs.append({"key": f"{split[:2]}_auth_{f}", "path": os.path.join(d, f), "split": split, "name": f})
    print(f"[workers] forged={len(recs)} authentic={len(auth_recs)} workers={args.workers}")
    with ProcessPoolExecutor(args.workers) as ex:
        fres = list(ex.map(forged_worker, recs, chunksize=8))
        ares = list(ex.map(authentic_worker, auth_recs, chunksize=8))
    F = md.merge(pd.DataFrame(fres), on="key", how="left")
    A = pd.DataFrame(ares).merge(pd.DataFrame(auth_recs)[["key", "split", "name"]], on="key")
    A["source_from_name"] = A["name"].map(source_from_authentic_name)
    print(f"[workers] done in {time.time()-t0:.0f}s; img errors={F.get('img_error', pd.Series(dtype=object)).notna().sum()}")

    # ---------------- Dim 3: mask values & semantics ----------------
    dim03 = {
        "mask_modes": F["mask_mode"].value_counts(dropna=False).to_dict(),
        "mask_formats": F["mask_format"].value_counts(dropna=False).to_dict(),
        "mask_size_equals_image": int(((F.W == F.mW) & (F.H == F.mH)).sum()),
        "mask_size_mismatch": int(((F.W != F.mW) | (F.H != F.mH)).sum()),
        "n_unique_values_distribution": F["mask_n_unique"].value_counts().to_dict(),
        "all_binary_0_255": bool(F["mask_is_binary_0_255"].all()),
        "n_masks_not_0_255": int((~F["mask_is_binary_0_255"].fillna(False)).sum()),
        "n_masks_all_zero": int((F["tamper_px"] == 0).sum()),
        "unique_value_sets": F["mask_unique"].str.replace(r":\d+", "", regex=True).value_counts().to_dict(),
        "rectangularity": describe(F["rectangularity"]),
        "frac_rectangularity_ge_0_99": float((F["rectangularity"] >= 0.99).mean()),
        "frac_rectangularity_eq_1": float((F["rectangularity"] >= 0.999999).mean()),
        "n_components_distribution": F["n_components"].value_counts().sort_index().to_dict(),
        "meta_bbox_exact_match_inclusive": int(F["meta_bbox_exact_incl"].fillna(False).sum()),
        "meta_bbox_exact_match_exclusive": int(F["meta_bbox_exact_excl"].fillna(False).sum()),
        "iou_meta_bbox_vs_mask_bbox_incl": describe(F["iou_meta_incl"]),
        "iou_meta_bbox_vs_mask_bbox_excl": describe(F["iou_meta_excl"]),
        "n_iou_meta_lt_0_9": int((F[["iou_meta_incl", "iou_meta_excl"]].max(axis=1) < 0.9).sum()),
    }
    # metadata rows whose bbox does not match the mask (all of them are duplicate-spec_id rows -> swapped provenance)
    bad = F[F[["iou_meta_incl", "iou_meta_excl"]].max(axis=1) < 0.9]
    dup_ids = set(F.loc[F.spec_id.duplicated(keep=False), "spec_id"])
    dim03["metadata_bbox_mismatch_rows"] = bad[["key", "spec_id", "source_dataset", "bbox_xyxy", "mask_bbox"]].assign(bbox_xyxy=lambda d: d.bbox_xyxy.map(list)).to_dict("records")
    dim03["n_metadata_bbox_mismatch_rows"] = int(len(bad))
    dim03["n_mismatch_rows_with_duplicate_spec_id"] = int(bad.spec_id.isin(dup_ids).sum())
    stats["dim03_mask_values_semantics"] = dim03
    print("[dim03] binary 0/255:", dim03["all_binary_0_255"], "rect>=0.99:", round(dim03["frac_rectangularity_ge_0_99"], 4),
          "meta exact incl/excl:", dim03["meta_bbox_exact_match_inclusive"], dim03["meta_bbox_exact_match_exclusive"])

    # ---------------- Dims 4-8: metadata distributions ----------------
    def ct(col, by="split"):
        t = pd.crosstab(F[col], F[by], margins=True)
        return {str(k): {str(c): int(v) for c, v in row.items()} for k, row in t.iterrows()}
    stats["dim04_source_distribution"] = ct("source_dataset")
    stats["dim05_generator_distribution"] = ct("assigned_tool")
    stats["dim05_generator_by_source"] = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in pd.crosstab(F.source_dataset, F.assigned_tool).iterrows()}
    stats["dim06_generator_naming"] = {
        "assigned_tool_values": sorted(F.assigned_tool.unique().tolist()),
        "card_aliases": {"gemini-nano": "Gemini 2.5 Flash Image (card/dataset_info)", "qwen-inpaint": "Ideogram v2 Edit (card/dataset_info)"},
        "note": "metadata values are internal code names; the card maps them to different commercial model names — see report"}
    stats["dim07_doc_type_distribution"] = ct("doc_type")
    stats["dim07_language_distribution"] = ct("language")
    fn = F.field_name.value_counts()
    stats["dim08_field_distribution"] = {"n_unique_fields": int(len(fn)), "top30": {str(k): int(v) for k, v in fn.head(30).items()},
                                         "by_source_top5": {s: {str(k): int(v) for k, v in F[F.source_dataset == s].field_name.value_counts().head(5).items()} for s in F.source_dataset.unique()}}
    F["orig_len"] = F.original_value.astype(str).str.len(); F["forg_len"] = F.forged_value.astype(str).str.len()
    stats["dim08_value_length"] = {"original_len": describe(F.orig_len), "forged_len": describe(F.forg_len),
                                   "frac_same_length": float((F.orig_len == F.forg_len).mean()),
                                   "frac_identical_value": float((F.original_value == F.forged_value).mean())}

    # ---------------- Dim 9: resolution ----------------
    F["aspect"] = F.W / F.H
    F["megapixels"] = F.W * F.H / 1e6
    dim09 = {"W": describe(F.W), "H": describe(F.H), "aspect_w_over_h": describe(F.aspect), "megapixels": describe(F.megapixels),
             "by_source": {s: {"W": describe(g.W), "H": describe(g.H), "megapixels": describe(g.megapixels)} for s, g in F.groupby("source_dataset")},
             "image_modes": F.img_mode.value_counts().to_dict(), "image_formats": F.img_format.value_counts().to_dict(),
             "n_unique_sizes": int(F[["W", "H"]].drop_duplicates().shape[0]),
             "frac_portrait": float((F.H > F.W).mean())}
    stats["dim09_resolution"] = dim09

    # ---------------- Dim 10: tamper ratio ----------------
    F["area_bin"] = F.tamper_ratio.map(area_bin)
    bins_order = ["<1%", "1-5%", "5-20%", ">20%"]
    dim10 = {"tamper_ratio": describe(F.tamper_ratio), "tamper_px": describe(F.tamper_px),
             "bins": {b: int((F.area_bin == b).sum()) for b in bins_order},
             "bins_frac": {b: float((F.area_bin == b).mean()) for b in bins_order},
             "bins_by_split": {s: {b: int(((F.area_bin == b) & (F.split == s)).sum()) for b in bins_order} for s in ("training", "testing")},
             "bins_by_source": {s: {b: int(((F.area_bin == b) & (F.source_dataset == s)).sum()) for b in bins_order} for s in sorted(F.source_dataset.unique())},
             "n_components": describe(F.n_components), "n_components_counts": F.n_components.value_counts().sort_index().to_dict(),
             "min_component_px": describe(F.min_component_px),
             "frac_ratio_lt_0_1pct": float((F.tamper_ratio < 0.001).mean()),
             "mask_bbox_w": describe(F.mask_bbox_w), "mask_bbox_h": describe(F.mask_bbox_h),
             "tamper_ratio_by_tool": {t: describe(g.tamper_ratio) for t, g in F.groupby("assigned_tool")},
             "tamper_ratio_by_source": {t: describe(g.tamper_ratio) for t, g in F.groupby("source_dataset")}}
    stats["dim10_tamper_ratio"] = dim10
    print("[dim10] bins:", dim10["bins"], "median ratio:", round(dim10["tamper_ratio"]["median"], 5))

    # ---------------- Dim 11/12: hashes, pairing, duplicates, leakage ----------------
    # --- exact duplicates (md5)
    allimg = pd.concat([
        F[["key", "split", "phash", "dhash", "img_md5", "W", "H"]].assign(kind="forged"),
        A[["key", "split", "phash", "dhash", "img_md5", "W", "H"]].assign(kind="authentic")], ignore_index=True)
    allimg = allimg[allimg.phash.notna()].reset_index(drop=True)
    md5_groups = allimg.groupby("img_md5").key.apply(list)
    md5_dups = md5_groups[md5_groups.map(len) > 1]
    dim11 = {"n_exact_md5_duplicate_groups": int(len(md5_dups)), "n_images_in_md5_dups": int(sum(len(v) for v in md5_dups)),
             "md5_dup_examples": [v for v in md5_dups.head(10).tolist()],
             "n_exact_md5_dups_forged_forged": int(sum(1 for v in md5_dups if sum(k.split('_')[1] != 'auth' for k in v) > 1)),
             "n_exact_md5_dups_authentic_authentic": int(sum(1 for v in md5_dups if sum(k.split('_')[1] == 'auth' for k in v) > 1))}

    # --- pairwise pHash hamming
    bits = np.stack([hex_to_bits(h) for h in allimg.phash])
    Dm = pairwise_hamming(bits, bits)
    np.fill_diagonal(Dm, 999)
    T = args.dup_threshold
    kind = allimg.kind.values; split = allimg.split.values
    is_f = kind == "forged"; is_a = ~is_f
    is_tr = split == "training"; is_te = ~is_tr

    # forged -> authentic pairing (same split, same size required); top-k pHash candidates, then pixel verification
    A_by_key = A.set_index("key")
    cand_rows = []
    f_idx = np.where(is_f)[0]
    TOPK = 3
    for i in f_idx:
        cand = np.where(is_a & (split == split[i]))[0]
        same = cand[(allimg.W.values[cand] == allimg.W.values[i]) & (allimg.H.values[cand] == allimg.H.values[i])]
        if len(same) == 0:
            cand_rows.append({"key": allimg.key[i], "rank": 0, "pair_key": None, "pair_dist": None, "pair_same_size": False}); continue
        order = same[np.argsort(Dm[i, same])]
        ds = Dm[i, order]
        k = 1 if ds[0] <= T and (len(ds) == 1 or ds[1] - ds[0] >= 10) else min(TOPK, len(order))
        for rnk in range(k):
            j = order[rnk]
            cand_rows.append({"key": allimg.key[i], "rank": rnk, "pair_key": allimg.key[j], "pair_name": A_by_key.loc[allimg.key[j], "name"],
                              "pair_dist": int(ds[rnk]), "pair_dist_2nd": int(ds[1]) if len(ds) > 1 else None, "pair_same_size": True})
    C = pd.DataFrame(cand_rows)
    # pixel verification of every candidate pair
    Fk = F.set_index("key")
    diff_recs = []
    for r in C[C.pair_key.notna()].itertuples():
        fr = Fk.loc[r.key]; folder = "TrainingSet" if fr.split == "training" else "TestingSet"
        diff_recs.append({"key": f"{r.key}|{r.rank}", "forged": os.path.join(DATA, folder, "images", fr.new_id + ".png"),
                          "authentic": os.path.join(DATA, folder, "authentic", r.pair_name),
                          "mask": os.path.join(DATA, folder, "masks", fr.new_id + ".png"), "bbox": list(fr.bbox_xyxy)})
    print(f"[pairing] verifying {len(diff_recs)} candidate pairs by pixel difference")
    with ProcessPoolExecutor(args.workers) as ex:
        dres = list(ex.map(diff_worker, diff_recs, chunksize=4))
    Dd = pd.DataFrame(dres)
    Dd["key"], Dd["rank"] = zip(*[(k.split("|")[0], int(k.split("|")[1])) for k in Dd.key])
    C = C.merge(Dd, on=["key", "rank"], how="left")
    VERIFY_DMEAN = 25.0   # mean |diff| outside the mask must be small (allows mild global re-encoding drift)
    # rule: low mean drift outside the mask AND (pHash close with <5% of image changed, or pHash very close = same doc with global drift)
    C["pair_verified"] = C.D_mean_outside_mask.notna() & (C.D_mean_outside_mask < VERIFY_DMEAN) & (
        ((C.pair_dist <= 60) & (C.outside_ratio_img_eps8 < 0.05)) | (C.pair_dist <= 10))
    # choose best candidate per forged: verified first, then lowest outside drift
    C = C.sort_values(["key", "pair_verified", "D_mean_outside_mask"], ascending=[True, False, True])
    P = C.groupby("key", as_index=False).first()
    P["n_candidates_checked"] = C.groupby("key").size().reindex(P.key).values
    F = F.merge(P.drop(columns=["rank"]), on="key", how="left")
    F["pair_verified"] = F.pair_verified.fillna(False).astype(bool)
    F["pair_accepted"] = F.pair_verified          # alias used downstream
    F["pair_phash_only_accept"] = F.pair_same_size.fillna(False) & (F.pair_dist <= T)
    pair_counts = F.loc[F.pair_accepted, "pair_key"].value_counts()
    dim11.update({
        "pairing_threshold": T, "pairing_topk": TOPK, "pairing_verify_rule": f"same split, same (W,H); mean |forged-authentic| outside GT mask < {VERIFY_DMEAN} AND ((pHash<=60 AND <5% of image pixels changed by >8 outside mask) OR pHash<=10)",
        "pair_dist_describe_verified": describe(F.loc[F.pair_verified, "pair_dist"]),
        "pair_dist_hist_verified": {str(k): int(v) for k, v in F.loc[F.pair_verified, "pair_dist"].value_counts().sort_index().items()},
        "pair_dist_describe_unverified": describe(F.loc[~F.pair_verified, "pair_dist"]),
        "n_forged_with_same_size_authentic": int(F.pair_same_size.fillna(False).sum()),
        "n_pairs_phash_only": int(F.pair_phash_only_accept.sum()),
        "n_pairs_verified": int(F.pair_verified.sum()),
        "n_pairs_verified_by_split": F.groupby("split").pair_verified.sum().astype(int).to_dict(),
        "n_pairs_verified_by_split_source": {f"{s}/{src}": int(v) for (s, src), v in F.groupby(["split", "source_dataset"]).pair_verified.sum().items()},
        "n_pairs_phash_accept_but_pixel_reject": int((F.pair_phash_only_accept & ~F.pair_verified).sum()),
        "n_pairs_pixel_verified_but_phash_gt_T": int((F.pair_verified & (F.pair_dist > T)).sum()),
        "n_pairs_accepted": int(F.pair_accepted.sum()),
        "n_pairs_rejected": int((~F.pair_accepted).sum()),
        "n_authentic_used_once": int((pair_counts == 1).sum()),
        "n_authentic_used_multiple": int((pair_counts > 1).sum()),
        "max_forged_per_authentic": int(pair_counts.max()) if len(pair_counts) else 0,
        "n_authentic_unused": int(len(A) - len(pair_counts)),
        "forged_per_authentic_hist": {str(k): int(v) for k, v in pair_counts.value_counts().sort_index().items()},
        "pair_source_agreement": float((F.loc[F.pair_accepted, "source_dataset"] == F.loc[F.pair_accepted, "pair_name"].map(source_from_authentic_name)).mean()) if F.pair_accepted.any() else None,
        "authentic_source_from_name_by_split": {s: A[A.split == s].source_from_name.value_counts().to_dict() for s in ("training", "testing")},
        "authentic_vs_forged_source_counts": {s: {"forged_meta": F[F.split == s].source_dataset.value_counts().to_dict(),
                                                   "authentic_name": A[A.split == s].source_from_name.value_counts().to_dict()} for s in ("training", "testing")},
    })

    # --- near-duplicate clusters (pHash <= T) over all images
    ii, jj = np.where(np.triu(Dm <= T, 1))
    pairs = pd.DataFrame({"i": ii, "j": jj, "dist": Dm[ii, jj]})
    pairs["key_i"] = allimg.key.values[ii]; pairs["key_j"] = allimg.key.values[jj]
    pairs["kind_i"] = kind[ii]; pairs["kind_j"] = kind[jj]; pairs["split_i"] = split[ii]; pairs["split_j"] = split[jj]
    pairs.to_csv(os.path.join(OUT, "phash_pairs.csv"), index=False)
    ff = pairs[(pairs.kind_i == "forged") & (pairs.kind_j == "forged")]
    dim11.update({
        "n_near_dup_pairs_all": int(len(pairs)),
        "n_near_dup_pairs_forged_forged": int(len(ff)),
        "n_near_dup_pairs_forged_forged_same_split": int((ff.split_i == ff.split_j).sum()),
        "n_forged_in_any_forged_forged_pair": int(len(set(ff.key_i) | set(ff.key_j))),
        "hamming_threshold_sensitivity": {str(t): {"forged_forged_pairs": int(((Dm[np.ix_(is_f, is_f)] <= t).sum()) // 2),
                                                   "cross_split_forged_forged_pairs": int((Dm[np.ix_(is_f & is_tr, is_f & is_te)] <= t).sum())}
                                          for t in (0, 5, 10, 15, 20, 25, 30, 40)},
        "random_pair_hamming_describe": describe(Dm[np.triu_indices(len(allimg), 1)][::97].astype(float)),
    })

    # --- source-document groups: union of (forged<->accepted authentic) and (near-dup pairs <= T)
    dsu = DSU(len(allimg))
    key2idx = {k: i for i, k in enumerate(allimg.key)}
    for r in F[F.pair_accepted].itertuples():
        dsu.union(key2idx[r.key], key2idx[r.pair_key])
    for r in pairs.itertuples():
        dsu.union(int(r.i), int(r.j))
    allimg["group"] = [dsu.find(i) for i in range(len(allimg))]
    grp_split = allimg.groupby("group").split.agg(lambda s: set(s))
    cross_groups = grp_split[grp_split.map(len) > 1].index
    F = F.merge(allimg[["key", "group"]], on="key", how="left")
    A = A.merge(allimg[["key", "group"]], on="key", how="left")
    gsize_forged = F.groupby("group").size()
    F["group_size_forged"] = F.group.map(gsize_forged)
    tr_te = pairs[(pairs.split_i != pairs.split_j)]
    dim12 = {
        "threshold": T,
        "n_groups_total": int(allimg.group.nunique()),
        "n_groups_with_forged": int(F.group.nunique()),
        "forged_per_group_hist": {str(k): int(v) for k, v in gsize_forged.value_counts().sort_index().items()},
        "n_forged_in_multi_forged_groups": int((F.group_size_forged > 1).sum()),
        "n_groups_spanning_train_and_test": int(len(cross_groups)),
        "n_forged_train_in_cross_split_groups": int(F[(F.split == "training") & F.group.isin(cross_groups)].shape[0]),
        "n_forged_test_in_cross_split_groups": int(F[(F.split == "testing") & F.group.isin(cross_groups)].shape[0]),
        "frac_test_forged_in_cross_split_groups": float(F[(F.split == "testing")].group.isin(cross_groups).mean()),
        "n_cross_split_near_dup_pairs_any": int(len(tr_te)),
        "n_cross_split_pairs_forged_forged": int(((tr_te.kind_i == "forged") & (tr_te.kind_j == "forged")).sum()),
        "n_cross_split_pairs_forged_authentic": int(((tr_te.kind_i != tr_te.kind_j)).sum()),
        "n_cross_split_pairs_authentic_authentic": int(((tr_te.kind_i == "authentic") & (tr_te.kind_j == "authentic")).sum()),
        "cross_split_pair_examples": tr_te.sort_values("dist").head(15)[["key_i", "key_j", "dist"]].to_dict("records"),
        "cross_split_groups_by_source": F[F.group.isin(cross_groups) & (F.split == "testing")].source_dataset.value_counts().to_dict(),
        "n_exact_md5_cross_split": int(sum(1 for v in md5_dups if len({k[:2] for k in v}) > 1)),
    }
    # test FORGED vs any TRAIN image (forged or authentic) — the leakage that matters for localisation models
    te_f = np.where(is_f & is_te)[0]; tr_any = np.where(is_tr)[0]; tr_f_idx = np.where(is_f & is_tr)[0]; tr_a_idx = np.where(is_a & is_tr)[0]
    nn_any = Dm[np.ix_(te_f, tr_any)].min(1); nn_f = Dm[np.ix_(te_f, tr_f_idx)].min(1); nn_a = Dm[np.ix_(te_f, tr_a_idx)].min(1)
    te_keys = allimg.key.values[te_f]
    nn_df = pd.DataFrame({"key": te_keys, "nn_train_any_dist": nn_any, "nn_train_forged_dist": nn_f, "nn_train_auth_dist": nn_a})
    F = F.drop(columns=[c for c in ["nn_train_any_dist"] if c in F.columns]).merge(nn_df, on="key", how="left")
    groups_with_train_forged = set(F[F.split == "training"].group)
    dim12.update({
        "test_forged_near_train_any": {str(t): int((nn_any <= t).sum()) for t in (0, 5, 10, 20, 30)},
        "test_forged_near_train_forged": {str(t): int((nn_f <= t).sum()) for t in (0, 5, 10, 20, 30)},
        "test_forged_near_train_authentic": {str(t): int((nn_a <= t).sum()) for t in (0, 5, 10, 20, 30)},
        "n_test_forged_sharing_group_with_train_forged": int(F[(F.split == "testing")].group.isin(groups_with_train_forged).sum()),
        "test_forged_near_train_any_T_by_source": F[(F.split == "testing") & (F.nn_train_any_dist <= T)].source_dataset.value_counts().to_dict(),
    })
    stats["dim11_duplicates"] = dim11
    stats["dim12_leakage"] = dim12
    print(f"[dim11] accepted pairs={dim11['n_pairs_accepted']}/{len(F)}  near-dup f-f pairs={dim11['n_near_dup_pairs_forged_forged']}")
    print(f"[dim12] cross-split groups={dim12['n_groups_spanning_train_and_test']}  test forged affected={dim12['n_forged_test_in_cross_split_groups']}")

    # ---------------- Dim 13: authentic availability ----------------
    stats["dim13_authentic"] = {
        "authentic_present": bool(len(A) > 0),
        "n_authentic_train": int((A.split == "training").sum()), "n_authentic_test": int((A.split == "testing").sum()),
        "authentic_modes": A.img_mode.value_counts().to_dict(), "authentic_formats": A.img_format.value_counts().to_dict(),
        "n_unique_authentic_md5_train": int(A[A.split == "training"].img_md5.nunique()),
        "n_unique_authentic_md5_test": int(A[A.split == "testing"].img_md5.nunique()),
        "n_unique_authentic_phash_train": int(A[A.split == "training"].phash.nunique()),
        "n_unique_authentic_phash_test": int(A[A.split == "testing"].phash.nunique()),
        "note": "card says 'Authentic baseline images: 812 (mirror of test split)'; local files decide."}

    # ---------------- Test-authentic provenance (is TestingSet/authentic a mirror of test forged?) ----------------
    a_idx = np.where(is_a)[0]
    D_af = Dm[np.ix_(a_idx, f_idx)]
    nn_d = D_af.min(1); nn_j = f_idx[D_af.argmin(1)]
    A["nn_forged_key"] = allimg.key.values[nn_j]; A["nn_forged_dist"] = nn_d; A["nn_forged_split"] = allimg.split.values[nn_j]
    D_aa = Dm[np.ix_(a_idx, a_idx)]
    a_tr = np.where(allimg.split.values[a_idx] == "training")[0]; a_te = np.where(allimg.split.values[a_idx] == "testing")[0]
    md5_tr_auth = set(A[A.split == "training"].img_md5)
    A["md5_in_train_authentic"] = A.img_md5.isin(md5_tr_auth) & (A.split == "testing")
    A["nn_train_auth_dist"] = np.nan
    if len(a_te) and len(a_tr):
        A.loc[A.split == "testing", "nn_train_auth_dist"] = D_aa[np.ix_(a_te, a_tr)].min(1)
    verified_test_pairs = F[(F.split == "testing") & F.pair_verified]
    te_auth = A[A.split == "testing"]
    stats["test_authentic_provenance"] = {
        "n_test_authentic": int(len(te_auth)),
        "n_test_authentic_pixel_verified_as_source_of_a_test_forged": int(verified_test_pairs.pair_key.nunique()),
        "n_test_forged_with_verified_authentic_in_test": int(len(verified_test_pairs)),
        "n_test_authentic_byte_identical_to_train_authentic": int(te_auth.md5_in_train_authentic.sum()),
        "n_test_authentic_near_dup_of_train_authentic": {str(t): int((te_auth.nn_train_auth_dist <= t).sum()) for t in (0, 5, 10, 20)},
        "n_test_authentic_whose_nearest_forged_is_in_train": int((te_auth.nn_forged_split == "training").sum()),
        "n_test_authentic_near_dup_of_a_train_forged": {str(t): int(((te_auth.nn_forged_split == "training") & (te_auth.nn_forged_dist <= t)).sum()) for t in (0, 5, 10, 20)},
        "n_test_authentic_near_dup_of_a_test_forged": {str(t): int(((te_auth.nn_forged_split == "testing") & (te_auth.nn_forged_dist <= t)).sum()) for t in (0, 5, 10, 20)},
        "n_test_authentic_not_near_any_forged_T": int((te_auth.nn_forged_dist > T).sum()),
        "test_authentic_source_by_name": te_auth.source_from_name.value_counts().to_dict(),
        "test_forged_source_by_metadata": F[F.split == "testing"].source_dataset.value_counts().to_dict(),
        "card_claim": "Authentic baseline images: 812 (mirror of test split)",
    }
    print("[test-auth]", {k: v for k, v in stats["test_authentic_provenance"].items() if isinstance(v, int)})

    # ---------------- Pixel-change vs bbox analysis (verified pairs; rows already computed in pairing) ----------------
    F = F.rename(columns={c: c for c in F.columns})
    ok = F[F.pair_verified & F.get("diff_error", pd.Series(index=F.index, dtype=object)).isna() & F.changed_px_eps0.notna()]
    diff_recs = ok
    def diff_block(g):
        if len(g) == 0: return {}
        b = {"n_pairs": int(len(g))}
        for eps in (0, 8):
            b[f"eps{eps}"] = {
                "iou_diff_vs_mask": describe(g[f"iou_diff_mask_eps{eps}"]),
                "frac_mask_pixels_actually_changed": describe(g[f"frac_mask_changed_eps{eps}"]),
                "changed_outside_mask_px": describe(g[f"changed_outside_mask_eps{eps}"]),
                "n_pairs_with_any_change_outside_mask": int((g[f"changed_outside_mask_eps{eps}"] > 0).sum()),
                "frac_pairs_with_any_change_outside_mask": float((g[f"changed_outside_mask_eps{eps}"] > 0).mean()),
                "n_pairs_outside_gt_100px": int((g[f"changed_outside_mask_eps{eps}"] > 100).sum()),
                "n_pairs_outside_gt_1pct_image": int((g[f"outside_ratio_img_eps{eps}"] > 0.01).sum()),
                "frac_of_changed_pixels_that_are_outside_mask": describe(g[f"frac_outside_of_changed_eps{eps}"]),
                "outside_changes_within_context_window_frac": describe((g[f"outside_in_ctx_eps{eps}"] / g[f"changed_outside_mask_eps{eps}"].replace(0, np.nan))),
                "n_pairs_with_outside_change_beyond_context_window": int(((g[f"changed_outside_mask_eps{eps}"] - g[f"outside_in_ctx_eps{eps}"]) > 0).sum()),
                "n_pairs_zero_change_inside_mask": int((g[f"changed_inside_mask_eps{eps}"] == 0).sum()),
            }
        b["D_max"] = describe(g.D_max); b["D_mean_outside_mask"] = describe(g.D_mean_outside_mask); b["D_p99_outside_mask"] = describe(g.D_p99_outside_mask)
        return b
    ok = ok.copy(); ok["global_drift_flag"] = ok.outside_ratio_img_eps8 > 0.05
    stats["pixel_change_vs_bbox"] = {
        "n_pairs_verified": int(F.pair_verified.sum()), "n_pairs_ok": int(len(ok)),
        "n_pairs_with_global_drift_gt5pct_image_eps8": int(ok.global_drift_flag.sum()),
        "global_drift_examples": ok[ok.global_drift_flag][["key", "source_dataset", "assigned_tool", "pair_name", "outside_ratio_img_eps8", "D_mean_outside_mask"]].head(12).to_dict("records"),
        "errors_examples": Dd.loc[Dd.get("diff_error", pd.Series(dtype=object)).notna(), "diff_error"].head(5).tolist() if "diff_error" in Dd else [],
        "all": diff_block(ok),
        "by_tool": {t: diff_block(g) for t, g in ok.groupby("assigned_tool")},
        "by_source": {s: diff_block(g) for s, g in ok.groupby("source_dataset")},
        "by_split": {s: diff_block(g) for s, g in ok.groupby("split")},
        "context_window_definition": "metadata bbox expanded per side by max(0.5*side, 100 px), clipped to image (interpretation of card text)",
    }
    if len(ok):
        print(f"[diff] IoU(D>0,mask) median={ok.iou_diff_mask_eps0.median():.3f}; pairs w/ outside change eps8: {(ok.changed_outside_mask_eps8>0).mean():.3f}")

    # ---------------- Dim 14: license (from card; stored as claims) ----------------
    stats["dim14_license_claims_from_card"] = {
        "hf_card_license_field": "cc-by-nc-sa-4.0 (README front-matter)",
        "dataset_info_json_license": "CC BY 4.0 (dataset_info.json)",
        "readme_license_link": "URL points to creativecommons.org/licenses/by/4.0 while text says cc-by-nc-sa-4.0",
        "gating": "gated=true; non-commercial research + citation checkbox",
        "source_licenses_per_card": {"CORD v2": "CC BY 4.0", "WildReceipt": "Apache 2.0", "SROIE": "ICDAR 2019 research use", "XFUND": "CC BY 4.0"},
        "note": "copied verbatim from card files for cross-check; see report for assessment"}

    # ---------------- Group-aware split ----------------
    rng = random.Random(args.seed)
    tr = F[F.split == "training"].copy()
    te = F[F.split == "testing"].copy()
    groups = tr.groupby("group").new_id.apply(list)
    # exclude from val any train group that also touches test (keep them in train; they are leakage-flagged)
    cross = set(cross_groups)
    g_ids = list(groups.index); rng.shuffle(g_ids)
    target = int(round(args.val_frac * len(tr)))
    val_groups, n_val = [], 0
    for g in g_ids:
        if n_val >= target: break
        if g in cross: continue
        val_groups.append(g); n_val += len(groups[g])
    tr["subset"] = np.where(tr.group.isin(val_groups), "val", "train")
    val = tr[tr.subset == "val"]; trn = tr[tr.subset == "train"]
    # smoke subsets: stratified by area bin, balanced over (source, tool)
    def smoke(df, n):
        df = df.copy()
        bin_counts = df.area_bin.value_counts(normalize=True)
        want = {b: max(1, int(round(n * bin_counts.get(b, 0)))) for b in bins_order if bin_counts.get(b, 0) > 0}
        # fix rounding
        while sum(want.values()) > n: want[max(want, key=want.get)] -= 1
        while sum(want.values()) < n: want[max(want, key=want.get)] += 1
        chosen = []
        for b, k in want.items():
            sub = df[df.area_bin == b]
            cells = {c: g.new_id.sample(frac=1, random_state=args.seed).tolist() for c, g in sub.groupby(["source_dataset", "assigned_tool"])}
            keys = sorted(cells); picked = []
            while len(picked) < min(k, len(sub)):
                for c in keys:
                    if cells[c] and len(picked) < k: picked.append(cells[c].pop())
            chosen += picked
        return sorted(chosen)
    smoke_train = smoke(trn, 400); smoke_val = smoke(val, 100)
    def comp(df, ids=None):
        d = df if ids is None else df[df.new_id.isin(ids)]
        return {"n": int(len(d)), "source": d.source_dataset.value_counts().to_dict(), "tool": d.assigned_tool.value_counts().to_dict(),
                "doc_type": d.doc_type.value_counts().to_dict(), "area_bin": {b: int((d.area_bin == b).sum()) for b in bins_order},
                "n_groups": int(d.group.nunique())}
    splits = {
        "train": sorted(trn.new_id.tolist()), "val": sorted(val.new_id.tolist()), "test": sorted(te.new_id.tolist()),
        "smoke_train": smoke_train, "smoke_val": smoke_val,
        "id_convention": "9-digit new_id (filename stem); train/val/smoke_* refer to TrainingSet/, test refers to TestingSet/ (official test split untouched)",
        "grouping_method": f"source-document groups = connected components over (forged<->paired authentic edges, pHash-256 hamming<={T}) ∪ (pHash-256 near-duplicate edges, hamming<={T}) over all forged+authentic images of both splits; val sampled by whole groups (seed={args.seed}, target {args.val_frac:.0%} of train); groups touching the test split are kept in train and flagged",
        "notes": {"n_train_groups": int(tr.group.nunique()), "n_val_groups": len(val_groups), "val_frac_realised": float(len(val) / len(tr)),
                  "n_train_groups_touching_test_kept_in_train": int(len(cross & set(groups.index))),
                  "composition": {"train": comp(trn), "val": comp(val), "test": comp(te), "smoke_train": comp(trn, smoke_train), "smoke_val": comp(val, smoke_val)},
                  "smoke_selection": "stratified by tamper-area bin (<1%,1-5%,5-20%,>20%) proportional to the parent subset; within bin round-robin over (source_dataset, assigned_tool) cells",
                  "no_val_group_overlaps_train": bool(not (set(val.group) & set(trn.group)))},
    }
    with open(os.path.join(OUT, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(nan_to_none(splits), f, indent=1, ensure_ascii=False)
    stats["split_proposal"] = {k: v for k, v in splits.items() if k not in ("train", "val", "test", "smoke_train", "smoke_val")}
    stats["split_proposal"]["sizes"] = {k: len(splits[k]) for k in ("train", "val", "test", "smoke_train", "smoke_val")}
    print("[split]", stats["split_proposal"]["sizes"])

    # ---------------- Figures ----------------
    if not args.no_figures:
        make_figures(F, A, DATA, OUT, bins_order, style_fn)

    # ---------------- save ----------------
    F.to_csv(os.path.join(OUT, "samples_forged.csv"), index=False)
    A.to_csv(os.path.join(OUT, "samples_authentic.csv"), index=False)
    stats["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, "audit_stats.json"), "w", encoding="utf-8") as f:
        json.dump(nan_to_none(stats), f, indent=1, ensure_ascii=False)
    print(f"[done] {time.time()-t0:.0f}s -> {OUT}")
    return stats, F, A

# --------------------------------------------------------------------------------------
def make_figures(F, A, DATA, OUT, bins_order, style_fn=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if style_fn is not None:
        style_fn()
    else:
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                             "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False,
                             "figure.dpi": 150, "savefig.dpi": 300})
    src_colors = {"wildreceipt": "#1b9e77", "cord": "#d95f02", "sroie": "#7570b3", "xfund": "#e7298a"}
    # --- resolution
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for s, g in F.groupby("source_dataset"):
        ax.scatter(g.W, g.H, s=4, alpha=0.5, color=src_colors.get(s, "gray"), label=f"{s} (n={len(g)})", linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title("Resolution spans ~2 orders of magnitude")
    ax.legend(frameon=False, markerscale=3, loc="upper left")
    ax.margins(0.04)
    ax = axes[1]
    ax.hist(F.aspect, bins=50, color="#555555")
    ax.set_xlabel("Aspect ratio (W / H)"); ax.set_ylabel("Forged images")
    ax.set_title(f"{(F.H > F.W).mean():.0%} of images are portrait (H > W)")
    ax.margins(0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_resolution.png"), bbox_inches="tight"); plt.close(fig)
    # --- tamper ratio
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    r = F.tamper_ratio[F.tamper_ratio > 0]
    bins = np.logspace(np.log10(max(r.min(), 1e-6)), 0, 50)
    for t, g in F.groupby("assigned_tool"):
        ax.hist(g.tamper_ratio[g.tamper_ratio > 0], bins=bins, alpha=0.7, label=f"{t} (n={len(g)})", color={"gemini-nano": "#4477aa", "qwen-inpaint": "#ee6677"}.get(t, "gray"))
    ax.set_xscale("log")
    for x in (0.01, 0.05, 0.20):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Tampered area / image area"); ax.set_ylabel("Forged images")
    med = F.tamper_ratio.median()
    ax.set_title(f"Tampered regions are tiny: median {med*100:.2f}% of the image")
    ax.legend(frameon=False, loc="upper left"); ax.margins(0.04)
    ax = axes[1]
    counts = [int((F.area_bin == b).sum()) for b in bins_order]
    ax.bar(bins_order, counts, color="#555555")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c/len(F):.1%}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Forged images"); ax.set_xlabel("Tamper-area bin")
    ax.set_title(f"{counts[0]/len(F):.0%} of forgeries cover <1% of the image")
    ax.margins(0.04); ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_tamper_ratio.png"), bbox_inches="tight"); plt.close(fig)
    # --- example overlay grid (8 examples: 2 per source, both tools where possible)
    rng = np.random.RandomState(0)
    picks = []
    for s in sorted(F.source_dataset.unique()):
        g = F[(F.source_dataset == s) & (F.split == "training") & F.tamper_px.gt(0)]
        for t in sorted(g.assigned_tool.unique())[:2]:
            gg = g[g.assigned_tool == t]
            picks.append(gg.iloc[rng.randint(len(gg))])
        if len(g.assigned_tool.unique()) == 1:
            picks.append(g.iloc[rng.randint(len(g))])
    picks = picks[:8]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.6))
    for ax, r in zip(axes.ravel(), picks):
        folder = "TrainingSet" if r.split == "training" else "TestingSet"
        im = Image.open(os.path.join(DATA, folder, "images", r.new_id + ".png")).convert("RGB")
        mk = np.asarray(Image.open(os.path.join(DATA, folder, "masks", r.new_id + ".png")))
        if mk.ndim == 3: mk = mk[..., 0]
        arr = np.asarray(im).copy()
        sel = mk > 127
        arr[sel] = (0.45 * arr[sel] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
        x1, y1, x2, y2 = r.bbox_xyxy
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#0077bb", lw=0.8))
        ax.set_title(f"{r.source_dataset} · {r.assigned_tool}\n{r.field_name} · {r.tamper_ratio*100:.2f}% · {r.W}×{r.H}", fontsize=6.5)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.suptitle("Red = ground-truth mask (field bbox), blue = metadata bbox_xyxy", fontsize=8, y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_examples_overlay.png"), bbox_inches="tight"); plt.close(fig)
    # --- pixel-diff quads
    ok = F[F.pair_accepted & F.get("changed_px_eps0", pd.Series(index=F.index)).notna()]
    if len(ok):
        ex = []
        for t in sorted(ok.assigned_tool.unique()):
            g = ok[ok.assigned_tool == t].sort_values("changed_outside_mask_eps8")
            ex.append(g.iloc[len(g) // 2])          # typical
            ex.append(g.iloc[-1])                   # worst outside leakage
        ex = ex[:4]
        fig, axes = plt.subplots(len(ex), 4, figsize=(7.2, 1.9 * len(ex)))
        axes = np.atleast_2d(axes)
        for row, r in zip(axes, ex):
            folder = "TrainingSet" if r.split == "training" else "TestingSet"
            f = np.asarray(Image.open(os.path.join(DATA, folder, "images", r.new_id + ".png")).convert("RGB")).astype(np.int16)
            a = np.asarray(Image.open(os.path.join(DATA, folder, "authentic", r.pair_name)).convert("RGB")).astype(np.int16)
            mk = np.asarray(Image.open(os.path.join(DATA, folder, "masks", r.new_id + ".png")))
            if mk.ndim == 3: mk = mk[..., 0]
            D = np.abs(f - a).max(axis=2)
            x1, y1, x2, y2 = r.bbox_xyxy; H, W = D.shape
            pw, ph = max(int(0.5 * (x2 - x1)), 100), max(int(0.5 * (y2 - y1)), 100)
            cx1, cy1, cx2, cy2 = max(0, x1 - 2 * pw), max(0, y1 - 2 * ph), min(W, x2 + 2 * pw), min(H, y2 + 2 * ph)
            sl = (slice(cy1, cy2), slice(cx1, cx2))
            row[0].imshow(f[sl].astype(np.uint8)); row[0].set_title("forged", fontsize=7)
            row[1].imshow(a[sl].astype(np.uint8)); row[1].set_title("authentic", fontsize=7)
            row[2].imshow(mk[sl], cmap="gray", vmin=0, vmax=255); row[2].set_title("GT mask (bbox)", fontsize=7)
            row[3].imshow(D[sl] > 8, cmap="gray", vmin=0, vmax=1); row[3].set_title("|forged−authentic| > 8", fontsize=7)
            for ax in row:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
            row[0].set_ylabel(f"{r.assigned_tool}\n{r.source_dataset} {r.new_id}\noutside-mask px={int(r.changed_outside_mask_eps8)}", fontsize=6.5)
        fig.suptitle("Pixel-change vs. bbox ground truth (crop = 2× context window)", fontsize=8, y=0.995)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pixel_diff_quads.png"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    main()
