"""
infer_to_maps.py — YOLO tespitlerini Faz 2 değerlendirme köprüsüne bağlar (TR).

Tam görüntüde dilimli çıkarım (natif çözünürlük, TILE/STRIDE pencereler) → kareler arası NMS →
(1) kutu listesi JSON  : phase3_yolo/preds_boxes/<model>/<suffix>/<id>.json   (tespit metrikleri için)
(2) uint8 olasılık haritası: phase2/experiments/preds/<model>/<suffix>/<id>.png  (kutu içi = conf·255, max-birleştirme)
    → phase2/experiments/evaluate_p2.py --models <model> --split val|test [--condition …] ile 3 GT altında AYNI
      metrikler (AUPRC/AUROC/F1@τ*/IoU/OracleF1) hesaplanır. τ* kalibrasyonu da aynı yoldan yapılır.

suffix kuralı infer_fullimage.py ile birebir: <split>[_<condition>][_<source>]
Bozulmalar p2data.DEGRADATIONS'tan (jpeg75, rescale075, blur1); kaynaklar forged|authentic|benign_pasteback|fullimage_resample.

Kullanım:
  python infer_to_maps.py --run aiforge-tiles1024-yolo11s-img1024-conserv-s42 --split val
  python infer_to_maps.py --run ... --split test --condition jpeg75
  python infer_to_maps.py --run ... --split val --source authentic
  python infer_to_maps.py --run aiforge-full-yolo11s-img1280-conserv-s42 --split val --full 1280   (full-resize modeli)
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np, cv2, pandas as pd, torch
from torchvision.ops import nms

os.environ.setdefault("USERNAME", "user")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2" / "experiments"))
from common import p2data as D  # noqa: E402

BOX_DIR = HERE / "preds_boxes"


def windows(H, W, tile, stride):
    ys = list(range(0, max(H - tile, 0) + 1, stride)); xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    return [(x, y) for y in ys for x in xs]


def detect_full(model, img, args):
    """→ (N,5) [x0,y0,x1,y1,conf] orijinal koordinatlarda."""
    H0, W0 = img.shape[:2]
    if args.full:
        r = model.predict(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), imgsz=args.full, conf=args.conf, iou=args.iou, verbose=False, half=True)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    img = D.pad_to_min(img, args.tile); H, W = img.shape[:2]
    wins = windows(H, W, args.tile, args.stride)
    boxes, confs = [], []
    for i in range(0, len(wins), args.batch):
        chunk = wins[i:i + args.batch]
        tiles = [cv2.cvtColor(img[y:y + args.tile, x:x + args.tile], cv2.COLOR_RGB2BGR) for x, y in chunk]
        for (x, y), r in zip(chunk, model.predict(tiles, imgsz=args.tile, conf=args.conf, iou=args.iou, verbose=False, half=True)):
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
            if len(b):
                b = b + np.array([x, y, x, y], dtype=np.float32); boxes.append(b); confs.append(c)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    b = np.concatenate(boxes); c = np.concatenate(confs)
    keep = nms(torch.from_numpy(b), torch.from_numpy(c), args.iou).numpy()
    b, c = b[keep], c[keep]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W0); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H0)
    return b, c


def boxes_to_map(b, c, H, W):
    m = np.zeros((H, W), np.float32)
    for (x0, y0, x1, y1), s in sorted(zip(b.tolist(), c.tolist()), key=lambda t: t[1]):
        x0, y0, x1, y1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], s)
    return (m * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="phase3_yolo/runs/<run> (weights/best.pt) — model adı olarak da kullanılır")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--condition", default="clean", choices=["clean"] + list(D.DEGRADATIONS))
    ap.add_argument("--source", default="forged", choices=["forged", "authentic", "benign_pasteback", "fullimage_resample"])
    ap.add_argument("--tile", type=int, default=1024); ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--full", type=int, default=0, help=">0 ise tam görüntü tek geçiş, bu imgsz ile")
    ap.add_argument("--conf", type=float, default=0.001); ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(HERE / "runs" / args.run / "weights" / "best.pt"))
    tab = D.split_table(args.split)
    ids = list(tab.index)[: args.limit or None]
    suffix = args.split + ("" if args.condition == "clean" else f"_{args.condition}") + ("" if args.source == "forged" else f"_{args.source}")
    out_png = D.PRED_DIR / args.run / suffix; out_box = BOX_DIR / args.run / suffix
    out_png.mkdir(parents=True, exist_ok=True); out_box.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for k, new_id in enumerate(ids):
        if args.source == "forged": p = D.img_path(new_id, args.split)
        elif args.source == "authentic": p = D.auth_path(tab.loc[new_id, "pair_name"], args.split)
        else: p = D.control_img_path(args.source, new_id)
        img = D.read_rgb(p)
        if args.condition != "clean":
            img = D.DEGRADATIONS[args.condition](img)
        H, W = img.shape[:2]
        b, c = detect_full(model, img, args)
        cv2.imwrite(str(out_png / f"{new_id}.png"), boxes_to_map(b, c, H, W))
        json.dump({"boxes_xyxy": b.round(1).tolist(), "conf": c.round(4).tolist(), "H": H, "W": W}, open(out_box / f"{new_id}.json", "w"))
        log.append(dict(new_id=new_id, n_boxes=len(b), top_conf=float(c.max()) if len(c) else 0.0, H=H, W=W))
        if (k + 1) % 50 == 0: print(f"[{suffix}] {k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(log).to_csv(HERE / "results" / f"infer_{suffix}_{args.run[:60]}.csv", index=False)
    print("done", suffix, f"{(time.time() - t0) / max(1, len(ids)) * 1000:.0f} ms/img")


if __name__ == "__main__":
    main()
