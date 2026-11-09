"""
detection_metrics.py — Tespit metriği (AP@0.5, AP@0.5:0.95, Recall@0.5 @ en iyi F1) iki model ailesi için AYNI tanımla (TR).

* YOLO: phase3_yolo/preds_boxes/<run>/<suffix>/<id>.json
* U-Net: phase2/experiments/preds/<model>/<suffix>/<id>.png → τ* (tau.json, bbox GT) ile eşik → bağlı bileşen →
  bileşen kutusu; skor = bileşen içi max olasılık. (Faz 2 E2 rect-fit ile aynı mantık.)
GT: her sahte görüntüde tam 1 kutu (samples_forged.csv bbox_xyxy). Alan dilimi (<1%, 1-5%, 5-20%) ve üretici (gemini/ideogram)
kırılımları da yazılır → phase3_yolo/results/detection_<suffix>.csv

Kullanım:
  python detection_metrics.py --yolo aiforge-tiles1024-yolo11s-img1024-conserv-s42,… --unet unet_bbox_s42,… --split val
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np, cv2, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2" / "experiments"))
from common import p2data as D  # noqa: E402

TAU = D.RESULTS_DIR / "tau.json"


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def ap_at(dets, n_gt, thr):
    """dets: list of (conf, iou_with_gt, img_id); tek GT/görüntü → aynı görüntüde ikinci eşleşme FP."""
    dets = sorted(dets, key=lambda t: -t[0]); matched = set(); tp = []
    for conf, io, img in dets:
        ok = io >= thr and img not in matched
        if ok: matched.add(img)
        tp.append(1 if ok else 0)
    tp = np.array(tp); fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(1, n_gt); prec = ctp / np.maximum(1, ctp + cfp)
    # COCO-tarzı 101 nokta interpolasyon
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r]; ap += (p.max() if len(p) else 0.0) / 101
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    return ap, float(rec[f1.argmax()]) if len(f1) else 0.0


def unet_boxes(png, tau):
    m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    if m is None: return []
    prob = m.astype(np.float32) / 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats((prob >= tau).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 4: continue
        out.append(((x, y, x + w, y + h), float(prob[lab == i].max())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=""); ap.add_argument("--unet", default="")
    ap.add_argument("--split", default="val", choices=["val", "test"]); ap.add_argument("--condition", default="clean")
    args = ap.parse_args()
    suffix = args.split + ("" if args.condition == "clean" else f"_{args.condition}")
    tab = D.split_table(args.split)
    tau = json.load(open(TAU)) if TAU.exists() else {}
    groups = {"all": tab.index, **{f"area_{b}": tab.index[tab["area_bin"] == b] for b in D.AREA_BINS[:3]},
              **{f"tool_{t}": tab.index[tab["tool"] == t] for t in ["gemini", "ideogram"]}}
    rows = []
    models = [(m, "yolo") for m in args.yolo.split(",") if m] + [(m, "unet") for m in args.unet.split(",") if m]
    for model, fam in models:
        per_img = {}
        for i in tab.index:
            gt = json.loads(tab.loc[i, "bbox_xyxy"]) if isinstance(tab.loc[i, "bbox_xyxy"], str) else list(tab.loc[i, "bbox_xyxy"])
            if fam == "yolo":
                f = HERE / "preds_boxes" / model / suffix / f"{i}.json"
                if not f.exists(): continue
                d = json.load(open(f)); cand = list(zip(d["boxes_xyxy"], d["conf"]))
            else:
                t = tau.get(model, {}).get("bbox", 0.5) if isinstance(tau.get(model), dict) else 0.5
                cand = unet_boxes(D.PRED_DIR / model / suffix / f"{i}.png", t)
            per_img[i] = [(c, iou(b, gt), i) for b, c in cand]
        for g, ids in groups.items():
            ids = [i for i in ids if i in per_img]
            if not ids: continue
            dets = [d for i in ids for d in per_img[i]]
            ap50, r50 = ap_at(dets, len(ids), 0.5)
            ap5095 = float(np.mean([ap_at(dets, len(ids), t)[0] for t in np.arange(0.5, 0.96, 0.05)]))
            rows.append(dict(model=model, family=fam, group=g, n=len(ids), AP50=round(ap50, 4), AP50_95=round(ap5095, 4),
                             Recall50_bestF1=round(r50, 4), mean_boxes_per_img=round(np.mean([len(per_img[i]) for i in ids]), 2)))
    out = pd.DataFrame(rows); (HERE / "results").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / f"detection_{suffix}.csv", index=False)
    print(out[out.group == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
