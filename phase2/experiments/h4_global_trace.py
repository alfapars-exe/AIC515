"""
h4_global_trace.py — H4: forged görüntülerde bbox DIŞINDA global yeniden-kodlama izi var mı? (TR)

Sınıf 1: forged görüntüden MASKEYE (bbox) DEĞMEYEN rastgele 256×256 crop; sınıf 0: eşleşen authentic görüntüde AYNI koordinat.
Train: train split (dışlananlar hariç, pair_accepted & aynı boyut) görüntü başına 2 çift; val: val split 1 çift.
Model: ResNet-18 (ImageNet) fine-tune, 224 merkez kesit (eğitimde rastgele 224 kesit, flip yok), 10 epoch, early-stop (val AUROC, patience 3).
Sonra aynı model, negatifleri fullimage_resample (val) ile değiştirilmiş simetrik koşulda test edilir (aynı koordinatlar).
Çıktı: results/h4_results.csv, results/h4_per_image_val.csv, logs/h4_train.csv, models/phase2/h4_resnet18.pt
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score
import common.p2data as D

PATCH, OUT = 256, 224


def sample_clean_window(bbox, W, H, size, rng, max_try=300):
    x0b, y0b, x1b, y1b = bbox
    if W < size or H < size:
        return None
    for _ in range(max_try):
        x0 = int(rng.integers(0, W - size + 1)); y0 = int(rng.integers(0, H - size + 1))
        if x0 + size <= x0b or x0 >= x1b or y0 + size <= y0b or y0 >= y1b:
            return x0, y0
    return None


def sample_bbox_window(bbox, W, H, size, rng, jitter=32):
    x0b, y0b, x1b, y1b = bbox
    if W < size or H < size or x1b <= x0b:
        return None
    cx = (x0b + x1b) // 2 + int(rng.integers(-jitter, jitter + 1)); cy = (y0b + y1b) // 2 + int(rng.integers(-jitter, jitter + 1))
    return int(np.clip(cx - size // 2, 0, W - size)), int(np.clip(cy - size // 2, 0, H - size))


def build_pairs(split: str, n_per_image: int, seed: int, tab: pd.DataFrame, positive_control: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for new_id, r in tab.iterrows():
        if not (bool(r["pair_accepted"]) and bool(r["pair_same_size"])):
            continue
        bm = D.read_mask(D.bbox_mask_path(new_id, split)); H, W = bm.shape
        bbox = D.mask_bbox(bm)
        for k in range(n_per_image):
            xy = sample_bbox_window(bbox, W, H, PATCH, rng) if positive_control else sample_clean_window(bbox, W, H, PATCH, rng)
            if xy is None:
                break
            rows.append(dict(new_id=new_id, x0=xy[0], y0=xy[1], rep=k, tool=r["tool"], source=r["source"], area_bin=r["area_bin"],
                             group=r["group"], pair_name=r["pair_name"]))
    return pd.DataFrame(rows)


def extract(pairs: pd.DataFrame, split: str, neg_source: str = "authentic", threads: int = 8):
    """(N,2,256,256,3) uint8: [forged crop, negatif crop]. neg_source: authentic | fullimage_resample"""
    def work(r):
        img = D.read_rgb(D.img_path(r.new_id, split))
        if neg_source == "authentic":
            neg = D.read_rgb(D.auth_path(r.pair_name, split))
        else:
            neg = D.read_rgb(D.control_img_path(neg_source, r.new_id))
        a = img[r.y0:r.y0 + PATCH, r.x0:r.x0 + PATCH]; b = neg[r.y0:r.y0 + PATCH, r.x0:r.x0 + PATCH]
        if a.shape != (PATCH, PATCH, 3) or b.shape != (PATCH, PATCH, 3):
            return None
        return np.stack([a, b])
    out = np.zeros((len(pairs), 2, PATCH, PATCH, 3), np.uint8); ok = np.zeros(len(pairs), bool)
    with ThreadPoolExecutor(threads) as ex:
        for i, res in enumerate(ex.map(work, list(pairs.itertuples(index=False)))):
            if res is not None:
                out[i] = res; ok[i] = True
    return out[ok], pairs[ok].reset_index(drop=True)


def batches(arr: np.ndarray, labels: np.ndarray, bs: int, train: bool, rng: np.random.Generator, device):
    idx = rng.permutation(len(arr)) if train else np.arange(len(arr))
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        x = torch.from_numpy(arr[sel]).to(device)  # (B,256,256,3) uint8
        if train:
            ox, oy = int(rng.integers(0, PATCH - OUT + 1)), int(rng.integers(0, PATCH - OUT + 1))
        else:
            ox = oy = (PATCH - OUT) // 2
        x = x[:, oy:oy + OUT, ox:ox + OUT]
        x = D.gpu_normalize(x)
        yield x, torch.from_numpy(labels[sel]).float().to(device)


@torch.no_grad()
def predict(model, arr, device, bs=128):
    model.eval(); out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(device)
        o = (PATCH - OUT) // 2
        x = D.gpu_normalize(x[:, o:o + OUT, o:o + OUT])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(x).float()[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--positive-control", action="store_true", help="bbox-merkezli (tampered) crop'lar: pipeline'ın öğrenebildiğinin kanıtı")
    args = ap.parse_args()
    PC = args.positive_control; sfx = "_poscontrol" if PC else ""
    D.ensure_dirs(); D.set_seed(args.seed)
    device = torch.device("cuda")
    tr_tab = D.split_table("train"); tr_tab = tr_tab[~tr_tab.index.isin(set(D.load_excluded()))]
    va_tab = D.split_table("val")
    t0 = time.time()
    tr_pairs = build_pairs("train", 2, args.seed, tr_tab, PC); va_pairs = build_pairs("val", 1, args.seed + 1, va_tab, PC)
    Xtr, tr_pairs = extract(tr_pairs, "train"); Xva, va_pairs = extract(va_pairs, "val")
    Xva_rs, va_pairs_rs = extract(va_pairs, "val", neg_source="fullimage_resample")
    print(f"pairs: train {len(tr_pairs)} val {len(va_pairs)} val-resample {len(va_pairs_rs)} | extract {time.time() - t0:.0f}s | RAM {Xtr.nbytes / 2 ** 30:.1f} GiB", flush=True)
    # (N,2,...) → (2N,...) etiketler [1,0]
    def flat(X):
        return X.reshape(-1, PATCH, PATCH, 3), np.tile(np.array([1, 0], np.float32), len(X))
    Atr, ytr = flat(Xtr); Ava, yva = flat(Xva); Ars, yrs = flat(Xva_rs)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 1); model = model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_b = int(np.ceil(len(Atr) / args.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_b)
    bce = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(args.seed)
    best, best_ep, bad, log = -1, -1, 0, []
    ckpt = D.MODELS_DIR / f"h4_resnet18{sfx}.pt"
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0; nb = 0; te = time.time()
        for x, y in batches(Atr, ytr, args.batch, True, rng, device):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x.contiguous(memory_format=torch.channels_last))[:, 0]
            loss = bce(logit.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
        pv = predict(model, Ava, device); auc = roc_auc_score(yva, pv)
        log.append(dict(epoch=ep, train_loss=tl / nb, val_auroc=auc, epoch_s=time.time() - te))
        print(f"[h4] ep {ep:02d} loss {tl / nb:.4f} val AUROC {auc:.4f} ({time.time() - te:.0f}s)", flush=True)
        if auc > best:
            best, best_ep, bad = auc, ep, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= 3:
                print("early stop", flush=True); break
    pd.DataFrame(log).to_csv(D.LOG_DIR / f"h4_train{sfx}.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    pv = predict(model, Ava, device); prs = predict(model, Ars, device)
    # per-pair tablo
    rows = []
    for i, r in va_pairs.iterrows():
        rows.append(dict(new_id=r.new_id, tool=r.tool, source=r.source, area_bin=r.area_bin, group=r.group, x0=r.x0, y0=r.y0,
                         p_forged=pv[2 * i], p_authentic=pv[2 * i + 1]))
    pi = pd.DataFrame(rows)
    rs_map = {r.new_id: (prs[2 * i], prs[2 * i + 1]) for i, r in va_pairs_rs.iterrows()}
    pi["p_forged_rs"] = pi.new_id.map(lambda k: rs_map.get(k, (np.nan, np.nan))[0]); pi["p_resample"] = pi.new_id.map(lambda k: rs_map.get(k, (np.nan, np.nan))[1])
    pi.to_csv(D.RESULTS_DIR / f"h4_per_image_val{sfx}.csv", index=False)

    def auc_of(d, col_neg):
        d = d.dropna(subset=["p_forged", col_neg])
        if len(d) < 5:
            return np.nan, len(d)
        y = np.r_[np.ones(len(d)), np.zeros(len(d))]; s = np.r_[d.p_forged.values, d[col_neg].values]
        return float(roc_auc_score(y, s)), len(d)
    res = []
    for cond, col in [("forged_vs_authentic", "p_authentic"), ("forged_vs_fullimage_resample", "p_resample")]:
        for sub, d in [("all", pi), ("gemini", pi[pi.tool == "gemini"]), ("ideogram", pi[pi.tool == "ideogram"])] + [(f"source={s}", pi[pi.source == s]) for s in sorted(pi.source.unique())]:
            a, n = auc_of(d, col)
            res.append(dict(condition=cond, subset=sub, n_pairs=n, auroc=a, acc_pairwise=float((d.p_forged > d[col]).mean()) if n else np.nan))
    res = pd.DataFrame(res)
    res["best_epoch"] = best_ep; res["n_train_pairs"] = len(tr_pairs); res["n_val_pairs"] = len(va_pairs)
    res["probe"] = "positive_control_bbox_crops" if PC else "outside_bbox_crops"
    res.to_csv(D.RESULTS_DIR / f"h4_results{sfx}.csv", index=False)
    print(res.round(4).to_string(), flush=True)
    json.dump(dict(best_epoch=best_ep, best_val_auroc=best, n_train_pairs=len(tr_pairs), n_val_pairs=len(va_pairs),
                   train_time_s=time.time() - t0, epochs_run=len(log)), open(D.RESULTS_DIR / f"h4_train{sfx}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
