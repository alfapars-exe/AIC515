"""
train_unet.py — D deneyi: U-Net (ResNet18 encoder, ImageNet) piksel lokalizasyonu (TR).

Varyantlar (--variant):
  rgb     : 3 kanal RGB
  rgbfreq : RGB + FFT Gaussian yüksek-geçiren (D0=20) + 3×3 Laplacian artığı = 5 kanal; ilk conv 5 kanala
            genişletilir (ilk 3 kanal ImageNet ağırlığı, ekstra 2 kanal 3 kanalın ortalama ağırlığı ile init).
Eğitim: 512×512 natif crop (%70 bbox-içeren jitter'lı, %30 rastgele), görüntü başına epoch'ta 2 crop,
loss = BCE + Dice, AdamW lr 1e-4, bf16 AMP, batch 8, early-stop (val Pixel-F1@0.5, patience 5).
Val: smoke_val her görüntü için 1 bbox-içeren + 1 rastgele crop (sabit seed) → piksel metrikleri,
alan dilimi / tool kırılımı, temiz crop FP oranı. Çıktı: results/unet_<variant>_results.json,
models/unet_<variant>_best.pt, logs/unet_<variant>.csv, cache/unet_<variant>_val_probs.npz
Kullanım: python train_unet.py --variant rgb [--epochs 15] [--batch 8]
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import common.data as D
from common.metrics import segmentation_summary, grouped_segmentation


class UNetWrapper(nn.Module):
    """RGB (B,3,H,W) alır; rgbfreq'te frekans kanallarını GPU'da ekler."""

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.net = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
        if variant == "rgbfreq":
            conv = self.net.encoder.conv1
            new = nn.Conv2d(5, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = conv.weight
                new.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
            self.net.encoder.conv1 = new

    def forward(self, x):
        if self.variant == "rgbfreq":
            x = torch.cat([x, D.freq_channels(x).to(x.dtype)], dim=1)
        return self.net(x)


def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    inter = (p * y).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def predict_records(model, loader, device, degrade=None):
    """Val crop'ları için (prob, gt, meta) kayıtları. degrade: uint8 RGB crop'u dönüştüren fonksiyon (E deneyi)."""
    model.eval()
    recs = []
    for x, y, metas in loader:
        if degrade is not None:
            # normalize tensörü → uint8 → bozulma → tekrar normalize
            mean = torch.tensor(D.IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(D.IMAGENET_STD).view(1, 3, 1, 1)
            u8 = ((x * std + mean).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
            u8 = np.stack([degrade(im) for im in u8])
            x = D.to_tensor_norm(u8.astype(np.float32) / 255.0) if u8.ndim == 3 else torch.stack(
                [D.to_tensor_norm(im.astype(np.float32) / 255.0) for im in u8])
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x)
        prob = torch.sigmoid(logit.float()).cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0].astype(np.uint8)
        for i, m in enumerate(metas):
            recs.append(dict(prob=prob[i].astype(np.float32), gt=gt[i], meta=m))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["rgb", "rgbfreq"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--crops-per-image", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    D.ensure_dirs(); D.set_seed(args.seed)
    device = torch.device("cuda")
    splits = D.load_splits(); samples = D.load_samples_forged()
    cache = D.ImageCache()
    t_load = time.time()
    for i in splits["smoke_train"] + splits["smoke_val"]:
        cache.image(i); cache.mask(i)
    print(f"cache: {cache.nbytes() / 2 ** 30:.2f} GiB in {time.time() - t_load:.0f}s", flush=True)

    val_idx_path = D.RESULTS_DIR / "seg_val_crop_index.csv"
    if val_idx_path.exists():
        val_idx = pd.read_csv(val_idx_path, dtype={"new_id": str})
    else:
        val_idx = D.build_val_crop_index("smoke_val", args.crop, args.seed + 7, cache, samples)
        val_idx.to_csv(val_idx_path, index=False)
    ds_tr = D.SegCropDataset(splits["smoke_train"], cache, args.crop, train=True, p_bbox=0.7,
                             crops_per_image=args.crops_per_image, seed=args.seed)
    ds_va = D.SegCropDataset(splits["smoke_val"], cache, args.crop, train=False, val_index=val_idx)
    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, generator=g, drop_last=True, collate_fn=D.seg_collate)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0, collate_fn=D.seg_collate)

    model = UNetWrapper(args.variant).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl_tr))
    bce = nn.BCEWithLogitsLoss()
    ckpt = D.MODELS_DIR / f"unet_{args.variant}_best.pt"
    log_rows, best, best_ep, bad = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl, nb, te = 0.0, 0, time.time()
        for x, y, _ in dl_tr:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)
            logit = logit.float()
            loss = bce(logit, y) + dice_loss(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
        recs = predict_records(model, dl_va, device)
        s = segmentation_summary(recs)
        vloss = float(np.mean([F.binary_cross_entropy(torch.tensor(r["prob"]).clamp(1e-6, 1 - 1e-6), torch.tensor(r["gt"], dtype=torch.float32)).item() for r in recs]))
        row = dict(epoch=ep, train_loss=tl / max(nb, 1), val_bce=vloss, val_f1=s.get("pixel_f1_at_0p5", float("nan")),
                   val_iou=s.get("iou_at_0p5", float("nan")), val_auprc=s.get("pixel_auprc_pooled_subsampled", float("nan")),
                   val_oracle_f1=s.get("oracle_f1", float("nan")), clean_fp_rate=s.get("clean_crop_fp_pixel_rate_at_0p5", float("nan")),
                   lr=opt.param_groups[0]["lr"], epoch_time_s=time.time() - te)
        log_rows.append(row)
        print(f"[{args.variant}] ep {ep:02d} loss {row['train_loss']:.4f} vbce {vloss:.4f} F1 {row['val_f1']:.4f} IoU {row['val_iou']:.4f} "
              f"AUPRC {row['val_auprc']:.4f} oracleF1 {row['val_oracle_f1']:.4f} cleanFP {row['clean_fp_rate']:.4f} ({row['epoch_time_s']:.0f}s)", flush=True)
        if row["val_f1"] > best:
            best, best_ep, bad = row["val_f1"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True); break
    train_time = time.time() - t0; peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30
    pd.DataFrame(log_rows).to_csv(D.LOG_DIR / f"unet_{args.variant}.csv", index=False)

    # --- en iyi checkpoint ile final değerlendirme (+ kırılımlar)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    recs = predict_records(model, dl_va, device)
    res = dict(overall=segmentation_summary(recs),
               by_crop_type=grouped_segmentation(recs, "crop_type"),
               by_tool=grouped_segmentation([r for r in recs if r["meta"]["crop_type"] == "bbox"], "tool"),
               by_area_bin=grouped_segmentation([r for r in recs if r["meta"]["crop_type"] == "bbox"], "area_bin"),
               by_source=grouped_segmentation([r for r in recs if r["meta"]["crop_type"] == "bbox"], "source"))
    # çıkarım süresi (512 crop, batch 1)
    xb = torch.zeros(1, 3, args.crop, args.crop, device=device).to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5): model(xb)
        torch.cuda.synchronize(); ti = time.time()
        for _ in range(30): model(xb)
        torch.cuda.synchronize()
    res["resources"] = dict(variant=args.variant, best_epoch=best_ep, epochs_run=len(log_rows), epochs_planned=args.epochs,
                            train_time_s=train_time, peak_gpu_mem_gib=peak_mem, n_params=n_params,
                            infer_ms_per_crop=(time.time() - ti) / 30 * 1000, ckpt_mb=os.path.getsize(ckpt) / 2 ** 20,
                            lr=args.lr, batch=args.batch, crop=args.crop, n_train_crops_per_epoch=len(ds_tr), n_val_crops=len(ds_va),
                            cache_gib=cache.nbytes() / 2 ** 30)
    with open(D.RESULTS_DIR / f"unet_{args.variant}_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    # val olasılık haritalarını sakla (figür ve dayanıklılık karşılaştırması için; uint8 0-255)
    np.savez_compressed(D.CACHE_DIR / f"unet_{args.variant}_val_probs.npz",
                        prob=np.stack([(r["prob"] * 255).round().astype(np.uint8) for r in recs]),
                        gt=np.stack([r["gt"] for r in recs]))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["overall"].items()}, ensure_ascii=False))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["resources"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
