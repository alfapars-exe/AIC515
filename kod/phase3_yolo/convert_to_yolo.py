"""
convert_to_yolo.py — AIForge-Doc v1 → Ultralytics YOLO tespit formatı (TR).

İki mod (tek değişkenli ablasyon için ikisi de üretilir):
  tiles : natif çözünürlükte TILE×TILE kareler (varsayılan 1024, stride 768). Pozitif kare = bbox'ın ≥%50'sini içeren
          kareler (hepsi) + görüntü başına NEG_PER_IMG rastgele negatif kare. Küçük hedefler (medyan bbox h = 32 px)
          için zorunlu yol; U-Net'in 512/448 kayan penceresiyle paralel.
  full  : görüntüler KOPYALANMAZ; Ultralytics'in "images→labels" kuralı gereği etiketler
          data/AIForge-Doc-v1/<Set>/labels/<id>.txt olarak yazılır ve bir liste .txt üretilir. imgsz eğitimde verilir.

Kurallar:
  * Dışlananlar (excluded_ids.json) ve split'ler phase2 ile birebir aynı (dataset_audit/splits.json).
  * --train-filter gemini_only → Faz 2 C aşamasının (held-out generator) YOLO karşılığı.
  * --auth-neg 1 → eşleşen authentic görüntüden 1 rastgele kare, boş etiketle (arka plan). Ablasyon faktörü.
  * PNG kaydı (kayıpsız) — JPEG kullanılmaz; sıkıştırma artefaktı adli sinyali bozar.
  * Tek sınıf: 0 = forged_region.

Kullanım:
  python convert_to_yolo.py --mode tiles --tile 1024 --stride 768 --neg-per-img 1 --auth-neg 0
  python convert_to_yolo.py --mode tiles --train-filter gemini_only --suffix _geminionly
  python convert_to_yolo.py --mode full
Çıktı: phase3_yolo/yolo_data/<mode>[suffix]/{images,labels}/{train,val} + data.yaml
"""
from __future__ import annotations
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np, cv2, pandas as pd
from concurrent.futures import ThreadPoolExecutor

import sys as _sys; from pathlib import Path as _Path
_sys.path.append(str(next(p for p in _Path(__file__).resolve().parents if (p / "_paths.py").is_file())))
from _paths import ROOT as _ROOT, KOD as _KOD, out_dir as _out_dir  # proje kökü · kod çatısı · çıktı klasörü
HERE = Path(__file__).resolve().parent
OUT = _out_dir(__file__)  # çıktı kökü: <proje>/phase3_yolo
sys.path.insert(0, str(HERE.parent / "phase2" / "experiments"))
from common import p2data as D  # noqa: E402

OUT_ROOT = OUT / "yolo_data"


def bbox_of(tab: pd.DataFrame, new_id: str):
    b = tab.loc[new_id, "bbox_xyxy"]
    if isinstance(b, str):
        b = json.loads(b)
    x0, y0, x1, y1 = [int(v) for v in b]
    return x0, y0, x1, y1


def write_label(path: Path, boxes_xyxy, W, H):
    lines = []
    for x0, y0, x1, y1 in boxes_xyxy:
        x0, x1 = max(0, x0), min(W, x1); y0, y1 = max(0, y0), min(H, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cx, cy, w, h = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def tile_grid(H, W, tile, stride):
    ys = list(range(0, max(H - tile, 0) + 1, stride)); xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    return [(x, y) for y in ys for x in xs]


def process_image(new_id, split, tab, args, out_img, out_lbl, rng, auth_neg):
    img = D.read_rgb(D.img_path(new_id, split))
    img = D.pad_to_min(img, args.tile)
    H, W = img.shape[:2]
    bx0, by0, bx1, by1 = bbox_of(tab, new_id)
    barea = max(1, (bx1 - bx0) * (by1 - by0))
    pos, neg = [], []
    for x, y in tile_grid(H, W, args.tile, args.stride):
        ix0, iy0, ix1, iy1 = max(x, bx0), max(y, by0), min(x + args.tile, bx1), min(y + args.tile, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        (pos if inter / barea >= 0.5 else neg).append((x, y))
    if not pos:  # bbox stride boşluğuna düşerse bbox-merkezli kare ekle
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        pos = [D.window_from_center(cx, cy, W, H, args.tile)]
    chosen = [(x, y, True) for x, y in pos] + [(x, y, False) for x, y in rng.sample(neg, min(args.neg_per_img, len(neg)))]
    n = 0
    for x, y, is_pos in chosen:
        name = f"{split}_{new_id}_{x}_{y}"
        crop = img[y:y + args.tile, x:x + args.tile]
        cv2.imwrite(str(out_img / f"{name}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        boxes = [(bx0 - x, by0 - y, bx1 - x, by1 - y)] if is_pos else []
        write_label(out_lbl / f"{name}.txt", boxes, args.tile, args.tile); n += 1
    if auth_neg:
        ap = D.auth_path(tab.loc[new_id, "pair_name"], split)
        if ap.exists():
            a = D.pad_to_min(D.read_rgb(ap), args.tile); aH, aW = a.shape[:2]
            x, y = D.sample_random_window(aW, aH, args.tile, np.random.default_rng(rng.randint(0, 2**31)))
            name = f"{split}_{new_id}_auth_{x}_{y}"
            cv2.imwrite(str(out_img / f"{name}.png"), cv2.cvtColor(a[y:y + args.tile, x:x + args.tile], cv2.COLOR_RGB2BGR))
            write_label(out_lbl / f"{name}.txt", [], args.tile, args.tile); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tiles", "full"], default="tiles")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--neg-per-img", type=int, default=1)
    ap.add_argument("--auth-neg", type=int, default=0)
    ap.add_argument("--train-filter", default=None, choices=[None, "gemini_only"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    out = OUT_ROOT / f"{args.mode}{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    vt = D.split_table("val")
    if args.train_filter == "gemini_only":  # Faz 2 C ile aynı: Ideogram val'de de held-out
        vt = vt[vt["tool"] == "gemini"]
    splits = {"train": D.train_ids(args.train_filter), "val": list(vt.index)}
    stats = {}
    for split, ids in splits.items():
        tab = D.split_table(split)
        if args.mode == "tiles":
            oi, ol = out / "images" / split, out / "labels" / split
            oi.mkdir(parents=True, exist_ok=True); ol.mkdir(parents=True, exist_ok=True)
            rng = random.Random(args.seed + (0 if split == "train" else 1))
            seeds = [random.Random(rng.randint(0, 2**31)) for _ in ids]
            with ThreadPoolExecutor(args.threads) as ex:
                counts = list(ex.map(lambda t: process_image(t[0], split, tab, args, oi, ol, t[1], args.auth_neg and split == "train"), zip(ids, seeds)))
            stats[split] = {"images": len(ids), "tiles": int(sum(counts))}
        else:
            lines = []
            for i in ids:
                p = D.img_path(i, split)
                W, H = int(tab.loc[i, "W"]), int(tab.loc[i, "H"])
                write_label(p.parent.parent / "labels" / f"{i}.txt", [bbox_of(tab, i)], W, H)
                lines.append(str(p))
            (out / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats[split] = {"images": len(ids)}
        print(split, stats[split], flush=True)

    yaml = {"path": str(out), "names": {0: "forged_region"},
            "train": "images/train" if args.mode == "tiles" else "train.txt",
            "val": "images/val" if args.mode == "tiles" else "val.txt"}
    (out / "data.yaml").write_text("\n".join(f"{k}: {v}" for k, v in yaml.items()) + "\n", encoding="utf-8")
    json.dump({"args": vars(args), "stats": stats}, open(out / "convert_stats.json", "w"), indent=2)
    print("data.yaml →", out / "data.yaml")


if __name__ == "__main__":
    main()
