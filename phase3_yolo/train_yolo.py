"""
train_yolo.py — Ultralytics YOLO eğitim sarmalayıcısı; Faz 2 run adlandırma/log disiplinine uyar (TR).

Run adı: aiforge-<data>-<model>-img<imgsz>-<aug>-s<seed>   (ör. aiforge-tiles-yolo11s-img1024-conserv-s42; <data> = yolo_data alt klasörü)
Çıktı:  phase3_yolo/runs/<run>/weights/best.pt  + phase3_yolo/results/checkpoint_summary_yolo.csv (satır ekler)

Augmentasyon profilleri (ablasyon faktörü):
  conserv : mosaic=0, scale=0, translate=0.1, fliplr=0.5, hsv küçük (U-Net'in fotometrik jitter'ı ile parite;
            yeniden-örnekleme içeren augler adli izi bozabilir → varsayılan BU)
  default : Ultralytics varsayılanları (mosaic=1, scale=0.5 …)

Kullanım:
  python train_yolo.py --data yolo_data/tiles/data.yaml --model yolo11s.pt --imgsz 1024 --epochs 40 --seed 42
  python train_yolo.py ... --epochs 2 --fraction 0.15 --dry-run      (A aşaması: duman testi)
"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
import pandas as pd
import torch

os.environ.setdefault("USERNAME", "user")  # ENV_SETUP.md bilinen sorun
HERE = Path(__file__).resolve().parent
RUNS, RES = HERE / "runs", HERE / "results"
AUG = {
    "conserv": dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.0, translate=0.1, degrees=0.0, shear=0.0,
                    perspective=0.0, fliplr=0.5, flipud=0.0, hsv_h=0.0, hsv_s=0.1, hsv_v=0.08, erasing=0.0, close_mosaic=0),
    "default": dict(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", choices=list(AUG), default="conserv")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--lr0", type=float, default=0.001, help="AdamW için 1e-3 (0.01 SGD varsayılanıdır; AdamW ile NaN val kaybı verdi — duman testi A)")
    ap.add_argument("--workers", type=int, default=6, help="Gerçek Windows'ta çok-süreçli yükleme çalışır (Faz 2 sandbox'ında 0'dı)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    data_tag = Path(args.data).resolve().parent.name  # tiles / tiles_geminionly / full
    mtag = Path(args.model).stem
    name = args.name or f"aiforge-{data_tag}-{mtag}-img{args.imgsz}-{args.aug}-s{args.seed}" + ("-dryrun" if args.dry_run else "")
    RUNS.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA yok — cu128 torch kurulu mu? (ENV_SETUP.md)"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
                seed=args.seed, deterministic=True, workers=args.workers, project=str(RUNS), name=name, exist_ok=True,
                fraction=args.fraction, lr0=args.lr0, optimizer="AdamW", cos_lr=True, amp=True, cache=False,
                plots=True, verbose=True, single_cls=True, **AUG[args.aug])
    dt = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    rdir = RUNS / name
    res = pd.read_csv(rdir / "results.csv"); res.columns = [c.strip() for c in res.columns]
    m50 = res["metrics/mAP50(B)"]; best_ep = int(m50.idxmax()) + 1
    row = dict(run=name, data=data_tag, model=mtag, imgsz=args.imgsz, aug=args.aug, seed=args.seed,
               epochs=len(res), best_epoch=best_ep, best_val_tile_map50=round(float(m50.max()), 4),
               last_val_tile_map50=round(float(m50.iloc[-1]), 4), best_val_tile_map5095=round(float(res["metrics/mAP50-95(B)"].max()), 4),
               time_min=round(dt, 1), peak_gib=round(peak, 2), ckpt_mb=round((rdir / "weights" / "best.pt").stat().st_size / 2**20, 1),
               params_M=round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2), dry_run=args.dry_run)
    out = RES / "checkpoint_summary_yolo.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    json.dump(row, open(rdir / "summary.json", "w"), indent=2)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
