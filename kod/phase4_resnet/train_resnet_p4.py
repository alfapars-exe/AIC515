"""
train_resnet_p4.py — Faz 4: ResNet-18 parça sınıflandırıcısının TAM eğitim bölümüyle yeniden eğitimi (TR).

Tarif Faz 1 `ft` ayarıyla aynıdır (referans: baselines/train_resnet.py): ImageNet ön-eğitimli ResNet-18, fc -> 1,
tam ince ayar, AdamW lr 1e-4 wd 1e-4, cosine, batch 32, 12 epoch, bf16 AMP, erken durdurma (val parça AUROC, sabır 5),
256 parçadan rastgele 224 kesit + +-%8 parlaklık/kontrast (flip yok). Değişen TEK şey veridir:
  --variant raw : ham eğitim görüntüleri (2748 görüntü)                         [faktör: veri miktarı, Faz 1'e göre]
  --variant mr  : işaretçi-onarımlı sahte görüntüler (phase3/marker_repaired)   [faktör: işaretçi onarımı, raw'a göre]
Model seçimi varyantın kendi val parçalarıyla yapılır (Faz 3 P3_MARKER_REPAIR kuralı: train + val onarılır);
her model ayrıca iki val varyantında da raporlanır.
Çıktı: models/phase4/<stem>.pt, phase4_resnet/logs/train_<stem>.csv, phase4_resnet/results/train_<stem>.json,
       phase4_resnet/results/val_patch_preds_<stem>.csv
Kullanım: python train_resnet_p4.py --variant raw --seed 42
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import p4common as P
D = P.D


def stem_of(variant: str, seed: int, tag: str = "") -> str:
    return f"resnet18_ft_full{'_mr' if variant == 'mr' else ''}_s{seed}" + (f"_{tag}" if tag else "")


def batch_of(arr, sel, rng, train, device):
    B = len(sel)
    if train:
        ox = rng.integers(0, P.PATCH - P.OUTSZ + 1, B); oy = rng.integers(0, P.PATCH - P.OUTSZ + 1, B)
        ab = np.stack([1.0 + rng.uniform(-0.08, 0.08, B), rng.uniform(-0.08, 0.08, B)], 1).astype(np.float32)
    else:
        ox = oy = np.full(B, (P.PATCH - P.OUTSZ) // 2); ab = None
    x = np.stack([arr[i, oy[j]:oy[j] + P.OUTSZ, ox[j]:ox[j] + P.OUTSZ] for j, i in enumerate(sel)])
    xt = torch.from_numpy(x).to(device)
    return D.gpu_normalize(xt, torch.from_numpy(ab).to(device) if ab is not None else None)


@torch.no_grad()
def predict(model, arr, device, bs=256):
    model.eval(); out = []
    for i in range(0, len(arr), bs):
        x = batch_of(arr, np.arange(i, min(i + bs, len(arr))), None, False, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(x).float()[:, 0]).cpu().numpy())
    return np.concatenate(out)


def patch_metrics(y, s, ptype) -> dict:
    y = np.asarray(y).astype(int); pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    def sub(neg):
        k = (ptype == "tampered") | (ptype == neg)
        return float(roc_auc_score(y[k], s[k])) if len(np.unique(y[k])) > 1 else float("nan")
    return dict(n=int(len(y)), n_pos=int(y.sum()), auroc=float(roc_auc_score(y, s)), auprc=float(average_precision_score(y, s)),
                accuracy=(tp + tn) / len(y), tpr_at_0p5=tp / max(tp + fn, 1), fpr_at_0p5=fp / max(fp + tn, 1),
                auroc_tampered_vs_clean_paired=sub("clean_paired"), auroc_tampered_vs_clean_same=sub("clean_same"),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["raw", "mr"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tag", default="", help="önbellek/çıktı etiketi (duman testi: dryrun)")
    args = ap.parse_args()
    P.ensure_dirs(); D.set_seed(args.seed)
    device = torch.device("cuda")
    stem = stem_of(args.variant, args.seed, args.tag)
    run_name = f"aiforge-patch224-resnet18-ft-fulltrain-{'markerrepair' if args.variant == 'mr' else 'raw'}-s{args.seed}" + (f"-{args.tag}" if args.tag else "")
    idx_tr, Xtr = P.get_patches("train", args.variant, args.tag)
    idx_va, Xva = P.get_patches("val", args.variant, args.tag)
    _, Xva_other = P.get_patches("val", "raw" if args.variant == "mr" else "mr", args.tag)
    ytr = idx_tr.label.values.astype(np.float32); yva = idx_va.label.values.astype(int); pt = idx_va.patch_type.values
    print(f"[{run_name}] train {len(Xtr)} parça / {idx_tr.new_id.nunique()} görüntü · val {len(Xva)} parça / {idx_va.new_id.nunique()} görüntü", flush=True)

    model = P.build_resnet(pretrained=True).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_b = len(Xtr) // args.batch                                  # drop_last (Faz 1 ile aynı)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_b)
    crit = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(args.seed)
    ckpt = P.MODELS / f"{stem}.pt"
    log, best, best_ep, bad = [], -1.0, -1, 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; te = time.time()
        order = rng.permutation(len(Xtr))
        for b in range(n_b):
            sel = order[b * args.batch:(b + 1) * args.batch]
            x = batch_of(Xtr, sel, rng, True, device); y = torch.from_numpy(ytr[sel]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)[:, 0]
            loss = crit(logit.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item()
        sv = predict(model, Xva, device); m = patch_metrics(yva, sv, pt)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(sv).clamp(1e-6, 1 - 1e-6), torch.tensor(yva, dtype=torch.float32)))
        row = dict(epoch=ep, train_loss=tl / n_b, val_loss=vloss, val_auroc=m["auroc"], val_acc=m["accuracy"],
                   val_tpr=m["tpr_at_0p5"], val_fpr=m["fpr_at_0p5"], lr=opt.param_groups[0]["lr"], epoch_time_s=time.time() - te)
        log.append(row)
        print(f"[{stem}] ep {ep:02d} loss {row['train_loss']:.4f} vloss {vloss:.4f} auroc {m['auroc']:.4f} acc {m['accuracy']:.3f} "
              f"tpr {m['tpr_at_0p5']:.3f} fpr {m['fpr_at_0p5']:.3f} ({row['epoch_time_s']:.1f}s)", flush=True)
        if not np.isfinite(row["train_loss"]):
            raise RuntimeError("train loss sonlu değil")
        if m["auroc"] > best:
            best, best_ep, bad = m["auroc"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True); break
    train_time = time.time() - t0
    pd.DataFrame(log).to_csv(P.LOG / f"train_{stem}.csv", index=False)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    s_own = predict(model, Xva, device); s_oth = predict(model, Xva_other, device)
    own, oth = args.variant, ("raw" if args.variant == "mr" else "mr")
    res = dict(stem=stem, run_name=run_name, variant=args.variant, seed=args.seed, best_epoch=best_ep, epochs_run=len(log),
               epochs_planned=args.epochs, lr=args.lr, wd=args.wd, batch=args.batch, patience=args.patience,
               n_train_patches=int(len(Xtr)), n_train_images=int(idx_tr.new_id.nunique()), n_val_patches=int(len(Xva)),
               train_time_s=train_time, peak_gpu_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
               final_train_loss=log[-1]["train_loss"], best_val_loss=log[best_ep - 1]["val_loss"],
               torch=torch.__version__, script_sha=P.file_sha(__file__), common_sha=P.file_sha(P.__file__),
               ckpt=str(ckpt), ckpt_sha=P.file_sha(ckpt))
    res[f"val_{own}"] = patch_metrics(yva, s_own, pt); res[f"val_{oth}"] = patch_metrics(yva, s_oth, pt)
    json.dump(res, open(P.RES / f"train_{stem}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    pred = idx_va[["new_id", "patch_type", "tool", "area_bin", "label"]].copy()
    pred[f"score_val_{own}"] = s_own; pred[f"score_val_{oth}"] = s_oth
    pred.to_csv(P.RES / f"val_patch_preds_{stem}.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, ensure_ascii=False, default=float), flush=True)
    print("val_raw", json.dumps(res["val_raw"], default=float)); print("val_mr", json.dumps(res["val_mr"], default=float), flush=True)


if __name__ == "__main__":
    main()
