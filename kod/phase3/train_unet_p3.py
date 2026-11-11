"""
train_unet_p3.py — Faz 3 segmentasyon eğitimi (TR). train_unet_p2.py ile AYNI protokol (crop örnekleme, BCE+Dice, AdamW 1e-4
cosine, bf16, val-crop AUPRC ile seçim, patience 4); yalnız tek bir faktör değişir:

  --arch unet|dlv3p             (W2: ikinci mimari; DeepLabV3+ aynı ResNet-18 kodlayıcı)
  --aux-bbox W                  (PHASE3-a: birleşik kayıp; kanal 1 = bbox GT, kayıp = L(primary) + W·L(bbox); model adı unetjoint_*)
  --marker-repair               (W4: eğitim/val görüntülerinde işaretçi onarılmış; GT olarak *_nm varyantı önerilir)

Run adı: <arch>_<gt>[_geminionly][_mr]_s<seed>  (unetjoint_<gt>_s<seed> birleşik kayıpta). Çıktılar Faz 2 klasörlerine (models/phase2, logs, results).
Örnek:
  set P3_MARKER_REPAIR=1 & python train_unet_p3.py --gt pixel_eps8_morph_nm --marker-repair --seed 42
  python train_unet_p3.py --arch dlv3p --gt bbox --seed 42
  python train_unet_p3.py --gt pixel_eps8_morph --aux-bbox 0.5 --seed 42
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3ext as X  # noqa: E402  (yamalar)
import numpy as np, pandas as pd, torch, torch.nn as nn
import train_unet_p2 as T  # dice_loss, val_auprc
D = X.D


class JointSampler(D.SegTrainSampler):
    """Birincil GT etrafında crop örnekler; y = [birincil, bbox] (2 kanal)."""

    def _item(self, new_id: str, seed: int):
        rng = np.random.default_rng(seed)
        img = D.read_rgb(D.img_path(new_id, "train")); H, W = img.shape[:2]
        gt = D.read_gt(new_id, self.gt_variant, "train", shape=(H, W))
        if gt is None:
            gt = np.zeros((H, W), np.uint8)
        bm = D.read_mask(D.bbox_mask_path(new_id, "train"))
        bbox = D.mask_bbox(bm)
        imgp = D.pad_to_min(img, self.size); gtp = D.pad_to_min(gt, self.size, value=0); bmp = D.pad_to_min(bm, self.size, value=0)
        xs, ys = [], []
        for _ in range(self.cpi):
            if rng.random() < self.p_gt:
                x0, y0 = D.sample_gt_window(gt, bbox, W, H, self.size, rng, jitter=self.size // 4)
            else:
                x0, y0 = D.sample_random_window(W, H, self.size, rng)
            xs.append(np.ascontiguousarray(D.crop(imgp, x0, y0, self.size)))
            ys.append(np.stack([D.crop(gtp, x0, y0, self.size), D.crop(bmp, x0, y0, self.size)]))
        ab = np.stack([[1.0 + rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)] for _ in range(self.cpi)]).astype(np.float32)
        return np.stack(xs), np.stack(ys), ab


class Ch0(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x)[:, :1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, choices=D.GT_VARIANTS)
    ap.add_argument("--arch", default="unet", choices=["unet", "dlv3p"])
    ap.add_argument("--aux-bbox", type=float, default=0.0)
    ap.add_argument("--marker-repair", action="store_true")
    ap.add_argument("--train-filter", default=None, choices=[None, "gemini_only"])  # train_unet_p2.py ile aynı süzgeç
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--crops-per-image", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.marker_repair and os.environ.get("P3_MARKER_REPAIR") != "1":
        raise SystemExit("--marker-repair için P3_MARKER_REPAIR=1 ortam değişkeni gerekli (yama import anında uygulanır)")
    D.ensure_dirs(); D.set_seed(args.seed)
    device = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    arch = "unetjoint" if args.aux_bbox > 0 else args.arch
    run = (f"{arch}_{args.gt}" + ("_geminionly" if args.train_filter == "gemini_only" else "") + ("_mr" if args.marker_repair else "")
           + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else ""))
    tr_ids = D.train_ids(args.train_filter)
    val_tab = D.split_table("val"); val_ids = list(val_tab.index)
    if args.train_filter == "gemini_only":
        val_ids = list(val_tab[val_tab["tool"] == "gemini"].index)  # Ideogram tamamen held-out kalsın (train_unet_p2.py ile aynı)
    print(f"[{run}] train {len(tr_ids)} | val {len(val_ids)} | gt {args.gt} | arch {arch} | repair {args.marker_repair}", flush=True)
    idx_path = D.RESULTS_DIR / f"val_crop_index_{args.gt}.csv"
    if idx_path.exists():
        val_idx = pd.read_csv(idx_path, dtype={"new_id": str})
    else:
        val_idx = D.build_val_crop_index(list(val_tab.index), args.gt, args.crop, seed=1000 + 7); val_idx.to_csv(idx_path, index=False)
    val_idx = val_idx[val_idx.new_id.isin(set(val_ids))].reset_index(drop=True)  # train_unet_p2.py ile aynı: seçim kırpıntıları da süzülür
    ipb = max(1, args.batch // args.crops_per_image)
    S = JointSampler if args.aux_bbox > 0 else D.SegTrainSampler
    sampler = S(tr_ids, args.gt, args.crop, args.crops_per_image, 0.7, ipb, args.seed, args.threads)
    val_loader = D.ValCropLoader(val_idx, args.gt, args.crop, batch=16, n_threads=args.threads)
    n_batches = len(sampler) if args.max_batches == 0 else min(len(sampler), args.max_batches)
    model = X.build_model(arch).to(device).to(memory_format=torch.channels_last)
    val_model = Ch0(model) if args.aux_bbox > 0 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_batches)
    bce = nn.BCEWithLogitsLoss()
    ckpt = D.MODELS_DIR / f"{run}.pt"
    rng_val = np.random.default_rng(0)
    log_rows, best, best_ep, bad = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl, nb, te = 0.0, 0, time.time()
        for x, y, ab in sampler:
            xt = D.gpu_normalize(torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(ab).to(device))
            yt = torch.from_numpy(y).to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xt)
            logit = logit.float()
            if args.aux_bbox > 0:
                loss = (bce(logit[:, :1], yt[:, :1]) + T.dice_loss(logit[:, :1], yt[:, :1]) +
                        args.aux_bbox * (bce(logit[:, 1:], yt[:, 1:]) + T.dice_loss(logit[:, 1:], yt[:, 1:])))
            else:
                loss = bce(logit, yt) + T.dice_loss(logit, yt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
            if nb >= n_batches:
                break
        t_train = time.time() - te; tv = time.time()
        v = T.val_auprc(val_model, val_loader, device, rng_val)
        row = dict(epoch=ep, train_loss=tl / max(nb, 1), **v, lr=opt.param_groups[0]["lr"], epoch_train_s=t_train,
                   epoch_val_s=time.time() - tv, n_batches=nb)
        log_rows.append(row)
        print(f"[{run}] ep {ep:02d} loss {row['train_loss']:.4f} AUPRC {v['val_auprc']:.4f} F1 {v['val_f1']:.4f} "
              f"cleanFP {v['clean_fp_rate']:.4f} | train {t_train:.0f}s val {row['epoch_val_s']:.0f}s", flush=True)
        pd.DataFrame(log_rows).to_csv(D.LOG_DIR / f"train_{run}.csv", index=False)
        if v["val_auprc"] > best:
            best, best_ep, bad = v["val_auprc"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True); break
    res = dict(run=run, arch=arch, gt=args.gt, seed=args.seed, marker_repair=args.marker_repair, aux_bbox=args.aux_bbox, train_filter=args.train_filter,
               n_train=len(tr_ids), n_val=len(val_ids), n_val_crops=len(val_idx), best_epoch=best_ep, best_val_auprc=best, epochs_run=len(log_rows), epochs_planned=args.epochs,
               train_time_s=time.time() - t0, peak_gpu_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
               n_params=sum(p.numel() for p in model.parameters()), ckpt=str(ckpt),
               ckpt_mb=os.path.getsize(ckpt) / 2 ** 20 if ckpt.exists() else None, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    json.dump(res, open(D.RESULTS_DIR / f"train_{run}.json", "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
