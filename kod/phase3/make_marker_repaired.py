"""
make_marker_repaired.py — İşaretçi kalıntısı nötralizasyonu (TR; W4/W5, PHASE 3 'zorunlu ön işleme').

TrainingSet (train+val) her eşleşmiş çift için (probe_green_marker.py ile birebir aynı tanım):
  residue = gmask(forged) & ~gmask(authentic) & ring(bbox, 6 px)  ;  resd = dilate(residue, 1)
  → phase3/marker_repaired/images/<id>.png : forged, resd pikselleri authentic'ten geri yazılmış (yalnız residue>0 ise)
  → phase2/gt/<v>_nm/<id>.png               : v ∈ {pixel_eps8_morph, glyph_v2}, resd pikselleri GT'den çıkarılmış
TestingSet: authentic eşi olan çiftlerde (gt/<v>/test/<id>.png mevcut) aynı _nm GT'si; tüm 812 test görüntüsü için
  'ring_green_px' = gmask(forged) & ring  (authentic gerekmeyen kalıntı göstergesi; yalnız ALT KÜME etiketi, model girdisi değil).
Çıktı: phase3/results/marker_residue.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3ext as X  # noqa: E402
D = X.D

OUT_RES = X.OUT / "results"; OUT_RES.mkdir(parents=True, exist_ok=True)
X.REPAIR_DIR.mkdir(parents=True, exist_ok=True)
VARIANTS = ["pixel_eps8_morph", "glyph_v2"]


def ring_of(mb):
    return ndi.binary_dilation(mb, iterations=6) & ~ndi.binary_erosion(mb, iterations=6)


def do_train(new_id, split, tab):
    f = D.read_rgb(D.img_path(new_id, split))
    mb = D.read_mask(D.bbox_mask_path(new_id, split)) > 0
    ring = ring_of(mb)
    row = dict(new_id=new_id, split=split, tool=tab.loc[new_id, "tool"], ring_green_px=int((X.gmask(f) & ring).sum()))
    ap = D.auth_path(tab.loc[new_id, "pair_name"], split) if isinstance(tab.loc[new_id, "pair_name"], str) else None
    a = D.read_rgb(ap) if ap is not None and ap.exists() else None
    if a is None or a.shape != f.shape:
        row.update(n_residue=-1, repaired=0); return row
    res = X.gmask(f) & ~X.gmask(a) & ring
    resd = ndi.binary_dilation(res, iterations=1)
    n = int(res.sum()); row.update(n_residue=n, n_residue_dil=int(resd.sum()), has_residue=int(n >= 20))
    if n > 0 and split != "test":
        rep = f.copy(); rep[resd] = a[resd]
        cv2.imwrite(str(X.REPAIR_DIR / f"{new_id}.png"), cv2.cvtColor(rep, cv2.COLOR_RGB2BGR))
        row["repaired"] = 1
    else:
        row["repaired"] = 0
    for v in VARIANTS:
        src = D.gt_path(new_id, v, split)
        if src is None:
            continue
        g = D.read_mask(src)
        if g.shape != resd.shape:
            continue
        removed = int((g.astype(bool) & resd).sum()); g[resd] = 0
        dst = D.GT_DIR / f"{v}_nm" / ("test" if split == "test" else "") / f"{new_id}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), g * 255)
        row[f"{v}_px_removed"] = removed
    return row


def main():
    rows, t0 = [], time.time()
    for split in ("train", "val", "test"):
        tab = D.split_table(split)
        ids = list(tab.index)
        with ThreadPoolExecutor(8) as ex:
            for k, r in enumerate(ex.map(lambda i: do_train(i, split, tab), ids)):
                rows.append(r)
                if (k + 1) % 400 == 0:
                    print(split, k + 1, len(ids), f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT_RES / "marker_residue.csv", index=False)
    s = df[df.n_residue >= 0].groupby(["split", "tool"]).agg(n=("new_id", "size"), residue_rate=("has_residue", "mean"),
                                                          repaired=("repaired", "sum")).reset_index()
    print(s.to_string(index=False))
    t = df[df.split == "test"]
    print("test ring_green_px>0 oranı (tool):", t.groupby("tool").ring_green_px.apply(lambda x: (x > 0).mean()).round(3).to_dict())
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
