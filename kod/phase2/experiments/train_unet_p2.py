"""
train_unet_p2.py — Faz 2: U-Net (ResNet18, ImageNet) eğitimi, seçilebilir GT (TR).

  python train_unet_p2.py --gt bbox|pixel_eps8_morph|glyph_v2 --seed 42 [--epochs 10] [--train-filter gemini_only]
                          [--batch 16] [--crop 512] [--crops-per-image 2] [--lr 1e-4] [--patience 4] [--tag dryrun]

Eğitim: tam train split (2762 − dışlananlar), 512 px natif crop (%70 GT-pozitif merkezli jitter'lı, %30 rastgele),
BCE+Dice, AdamW lr 1e-4 (cosine), batch 16 (= 8 görüntü × 2 crop), bf16 AMP, görüntüler diskten (thread prefetch).
Model seçimi: val (487) sabit crop'larında EĞİTİM GT'sine göre havuzlanmış Pixel-AUPRC; early-stop patience 4.
Çıktı: models/phase2/unet_<gt>[_geminionly]_s<seed>.pt, logs/train_<run>.csv, results/train_<run>.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from sklearn.metrics import average_precision_score

import common.p2data as D


def build_model() -> nn.Module:
    return smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)


def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    inter = (p * y).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def val_auprc(model, loader, device, rng) -> dict:
    """Val crop'larında havuzlanmış AUPRC (alt-örnek: tüm pozitif + crop başına ≤20k negatif), F1@0.5, temiz FP."""
    model.eval()
    S, Y, f1s, fps, n_gt = [], [], [], [], 0
    for x, y, _ in loader:
        xt = D.gpu_normalize(torch.from_numpy(x).to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(xt)
        prob = torch.sigmoid(logit.float()).cpu().numpy()[:, 0]
        gt = y[:, 0] > 0
        for i in range(len(prob)):
            g = gt[i].ravel(); p = prob[i].ravel()
            if g.any():
                n_gt += 1
                pos = np.flatnonzero(g); neg = np.flatnonzero(~g)
                sel = np.concatenate([pos, rng.choice(neg, min(len(neg), 20000), replace=False)])
                S.append(p[sel]); Y.append(g[sel])
                pb = p >= 0.5; tp = (pb & g).sum(); f1s.append(2 * tp / max(pb.sum() + g.sum(), 1))
            else:
                fps.append((p >= 0.5).mean())
    out = dict(val_auprc=float(average_precision_score(np.concatenate(Y), np.concatenate(S))) if S else float("nan"),
               val_f1=float(np.mean(f1s)) if f1s else float("nan"),
               clean_fp_rate=float(np.mean(fps)) if fps else float("nan"), n_gt_crops=n_gt, n_clean_crops=len(fps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, choices=D.GT_VARIANTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train-filter", default=None, choices=[None, "gemini_only"])
    ap.add_argument("--batch", type=int, default=16, help="crop sayısı (= images_per_batch × crops_per_image)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--crops-per-image", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-batches", type=int, default=0, help="dry-run: epoch başına batch sınırı (0 = tümü)")
    args = ap.parse_args()
    D.ensure_dirs(); D.set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    run = f"unet_{args.gt}" + ("_geminionly" if args.train_filter == "gemini_only" else "") + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    tr_ids = D.train_ids(args.train_filter)
    val_tab = D.split_table("val")
    val_ids = list(val_tab.index)
    if args.train_filter == "gemini_only":
        val_ids = list(val_tab[val_tab["tool"] == "gemini"].index)  # Ideogram tamamen held-out kalsın
    print(f"[{run}] train {len(tr_ids)} | val {len(val_ids)} | gt {args.gt}", flush=True)

    idx_path = D.RESULTS_DIR / f"val_crop_index_{args.gt}.csv"
    if idx_path.exists():
        val_idx = pd.read_csv(idx_path, dtype={"new_id": str})
    else:
        t = time.time(); val_idx = D.build_val_crop_index(list(val_tab.index), args.gt, args.crop, seed=1000 + 7)
        val_idx.to_csv(idx_path, index=False); print(f"val crop index built in {time.time() - t:.0f}s", flush=True)
    val_idx = val_idx[val_idx.new_id.isin(set(val_ids))].reset_index(drop=True)

    ipb = max(1, args.batch // args.crops_per_image)
    sampler = D.SegTrainSampler(tr_ids, args.gt, args.crop, args.crops_per_image, 0.7, ipb, args.seed, args.threads)
    val_loader = D.ValCropLoader(val_idx, args.gt, args.crop, batch=16, n_threads=args.threads)
    n_batches = len(sampler) if args.max_batches == 0 else min(len(sampler), args.max_batches)

    model = build_model().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_batches)
    bce = nn.BCEWithLogitsLoss()
    ckpt = D.MODELS_DIR / f"{run}.pt"
    rng_val = np.random.default_rng(0)
    log_rows, best, best_ep, bad = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl, nb, te, t_data = 0.0, 0, time.time(), 0.0
        td = time.time()
        for x, y, ab in sampler:
            t_data += time.time() - td
            xt = D.gpu_normalize(torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(ab).to(device))
            yt = torch.from_numpy(y).to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xt)
            logit = logit.float()
            loss = bce(logit, yt) + dice_loss(logit, yt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
            if nb >= n_batches:
                break
            td = time.time()
        t_train = time.time() - te; tv = time.time()
        v = val_auprc(model, val_loader, device, rng_val)
        row = dict(epoch=ep, train_loss=tl / max(nb, 1), **v, lr=opt.param_groups[0]["lr"],
                   epoch_train_s=t_train, epoch_val_s=time.time() - tv, data_wait_s=t_data, n_batches=nb)
        log_rows.append(row)
        print(f"[{run}] ep {ep:02d} loss {row['train_loss']:.4f} AUPRC {v['val_auprc']:.4f} F1 {v['val_f1']:.4f} "
              f"cleanFP {v['clean_fp_rate']:.4f} | train {t_train:.0f}s (data-wait {t_data:.0f}s) val {row['epoch_val_s']:.0f}s", flush=True)
        pd.DataFrame(log_rows).to_csv(D.LOG_DIR / f"train_{run}.csv", index=False)
        if v["val_auprc"] > best:
            best, best_ep, bad = v["val_auprc"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True); break
    train_time = time.time() - t0
    res = dict(run=run, gt=args.gt, seed=args.seed, train_filter=args.train_filter, n_train=len(tr_ids), n_val=len(val_ids),
               best_epoch=best_ep, best_val_auprc=best, epochs_run=len(log_rows), epochs_planned=args.epochs,
               train_time_s=train_time, peak_gpu_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
               n_params=sum(p.numel() for p in model.parameters()), batch=args.batch, crop=args.crop, lr=args.lr,
               crops_per_image=args.crops_per_image, n_train_crops_per_epoch=n_batches * args.batch, ckpt=str(ckpt),
               ckpt_mb=os.path.getsize(ckpt) / 2 ** 20 if ckpt.exists() else None, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    with open(D.RESULTS_DIR / f"train_{run}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
