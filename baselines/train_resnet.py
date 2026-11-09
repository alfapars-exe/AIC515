"""
train_resnet.py — C deneyi: ResNet18 patch sınıflandırma (TR).

Üç ayar (--setting):
  probe   : ImageNet-pretrained, backbone DONDURULMUŞ (BN eval), yalnızca yeni FC eğitilir (lr 1e-3)
  ft      : ImageNet-pretrained, tam fine-tune (lr 1e-4)
  scratch : aynı mimari, rastgele init (lr 1e-3)
Giriş: 256 natif patch → 224 random-crop (eğitim) / center-crop (val); yeniden örnekleme YOK.
Augment: yalnızca küçük parlaklık/kontrast jitter (hflip yok — metin). AdamW, cosine LR, BCE loss,
bf16 AMP, early-stop (val AUROC, patience 5). Çıktı: results/resnet_<setting>_*.json/csv,
models/resnet18_<setting>_best.pt (state_dict), logs/resnet_<setting>.csv (eğitim eğrisi).
Kullanım: python train_resnet.py --setting ft [--epochs 12] [--batch 32]
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import common.data as D
from common.metrics import classification_metrics, subset_auroc, safe_auroc

SETTINGS = {
    "probe": dict(pretrained=True, freeze=True, lr=1e-3),
    "ft": dict(pretrained=True, freeze=False, lr=1e-4),
    "scratch": dict(pretrained=False, freeze=False, lr=1e-3),
}


def build_model(setting: str) -> nn.Module:
    cfg = SETTINGS[setting]
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(512, 1)
    if cfg["freeze"]:
        for n, p in m.named_parameters():
            p.requires_grad = n.startswith("fc.")
    return m


def set_train_mode(m: nn.Module, setting: str):
    """probe'da backbone BN istatistikleri sabit kalsın (eval); yalnızca fc eğitilir."""
    if SETTINGS[setting]["freeze"]:
        m.eval()
    else:
        m.train()


@torch.no_grad()
def predict(m, loader, device):
    m.eval()
    scores, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = m(x).squeeze(1)
        scores.append(torch.sigmoid(logit.float()).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(scores), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=list(SETTINGS), required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    D.ensure_dirs(); D.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SETTINGS[args.setting]

    idx_tr, P_tr = D.get_or_build_patches("smoke_train", 2, args.seed)
    idx_va, P_va = D.get_or_build_patches("smoke_val", 1, args.seed + 1)
    ds_tr = D.PatchArrayDataset(P_tr, idx_tr.label.values, train=True, seed=args.seed)
    ds_va = D.PatchArrayDataset(P_va, idx_va.label.values, train=False)
    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, generator=g, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=args.workers)

    model = build_model(args.setting).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl_tr))
    crit = nn.BCEWithLogitsLoss()

    ckpt = D.MODELS_DIR / f"resnet18_{args.setting}_best.pt"
    log_rows, best_auc, best_ep, bad = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        set_train_mode(model, args.setting)
        tl, nb = 0.0, 0
        te = time.time()
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x).squeeze(1)
            loss = crit(logit.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
        s_va, y_va = predict(model, dl_va, device)
        m = classification_metrics(y_va, s_va)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(s_va).clamp(1e-6, 1 - 1e-6), torch.tensor(y_va, dtype=torch.float32)))
        row = dict(epoch=ep, train_loss=tl / max(nb, 1), val_loss=vloss, val_auroc=m["auroc"], val_acc=m["accuracy"],
                   val_bacc=m["balanced_accuracy"], val_f1=m["f1"], lr=opt.param_groups[0]["lr"], epoch_time_s=time.time() - te)
        log_rows.append(row)
        print(f"[{args.setting}] ep {ep:02d} loss {row['train_loss']:.4f} vloss {vloss:.4f} auroc {m['auroc']:.4f} "
              f"acc {m['accuracy']:.3f} bacc {m['balanced_accuracy']:.3f} f1 {m['f1']:.3f} ({row['epoch_time_s']:.1f}s)", flush=True)
        if m["auroc"] > best_auc:
            best_auc, best_ep, bad = m["auroc"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True); break
    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30
    pd.DataFrame(log_rows).to_csv(D.LOG_DIR / f"resnet_{args.setting}.csv", index=False)

    # --- en iyi checkpoint ile final değerlendirme
    model.load_state_dict(torch.load(ckpt, map_location=device))
    s_va, y_va = predict(model, dl_va, device)
    res = classification_metrics(y_va, s_va)
    pt = idx_va.patch_type.values
    res["auroc_tampered_vs_clean_paired"] = subset_auroc(y_va, s_va, pt, lambda g: g in ("tampered", "clean_paired"))
    res["auroc_tampered_vs_clean_same"] = subset_auroc(y_va, s_va, pt, lambda g: g in ("tampered", "clean_same"))
    for tool in sorted(idx_va.tool.unique()):
        keep = (y_va == 0) | (idx_va.tool.values == tool)
        res[f"auroc_tool_{tool}"] = safe_auroc(y_va[keep], s_va[keep])
    for ab in sorted(idx_va.area_bin.unique()):
        keep = (y_va == 0) | (idx_va.area_bin.values == ab)
        res[f"auroc_area_{ab}"] = safe_auroc(y_va[keep], s_va[keep])
    # çıkarım süresi (patch başına, batch 1, bf16)
    model.eval(); xb = torch.zeros(1, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10): model(xb)
        torch.cuda.synchronize(); ti = time.time()
        for _ in range(50): model(xb)
        torch.cuda.synchronize()
    infer_ms = (time.time() - ti) / 50 * 1000
    res.update(dict(setting=args.setting, best_epoch=best_ep, epochs_run=len(log_rows), epochs_planned=args.epochs,
                    train_time_s=train_time, peak_gpu_mem_gib=peak_mem, n_params=n_params, n_trainable=n_train,
                    infer_ms_per_patch=infer_ms, ckpt_mb=os.path.getsize(ckpt) / 2 ** 20, lr=cfg["lr"], batch=args.batch,
                    n_train_patches=len(ds_tr), n_val_patches=len(ds_va)))
    with open(D.RESULTS_DIR / f"resnet_{args.setting}_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    pred = idx_va[["new_id", "patch_type", "tool", "area_bin", "label"]].copy(); pred["score"] = s_va
    pred.to_csv(D.RESULTS_DIR / f"resnet_{args.setting}_val_predictions.csv", index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
