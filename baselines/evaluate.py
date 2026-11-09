"""
evaluate.py — Sonuçları birleştirir, E (dayanıklılık) probunu çalıştırır ve figürleri üretir (TR).

Adımlar:
  1. resnet_*_results.json → results/resnet_results.json/csv; unet_*_results.json → results/unet_results.json/csv
  2. E) U-Net rgb / rgbfreq checkpoint'leri val crop'larının JPEG Q75 ve 0.75× ölçek-geri sürümlerinde
     yeniden değerlendirilir (eğitim yok) → results/robustness_results.csv
  3. F) resource_usage.csv (süre, peak GPU bellek, parametre, çıkarım süresi, checkpoint boyutu)
  4. Figürler: training_curves_resnet.png, training_curves_unet.png, confusion_resnet.png,
     rf_feature_importance.png, unet_predictions.png (figure-style kurallarına uygun, sade)
Kullanım: python evaluate.py [--skip-robustness]
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common.data as D
from common.metrics import segmentation_summary, grouped_segmentation

RES, FIG, LOG = D.RESULTS_DIR, D.FIG_DIR, D.LOG_DIR
COL = {"probe": "#1b6ca8", "ft": "#d1495b", "scratch": "#6c757d", "rgb": "#1b6ca8", "rgbfreq": "#d1495b"}
NAME = {"probe": "pretrained, frozen backbone (linear probe)", "ft": "pretrained, full fine-tune", "scratch": "random init (scratch)",
        "rgb": "U-Net RGB (3 ch)", "rgbfreq": "U-Net RGB + FFT high-pass + Laplacian (5 ch)"}


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 1. birleştirme
def aggregate_resnet():
    rows, full = [], {}
    for s in ("probe", "ft", "scratch"):
        p = RES / f"resnet_{s}_results.json"
        if p.exists():
            r = load_json(p); full[s] = r
            rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    if rows:
        pd.DataFrame(rows).to_csv(RES / "resnet_results.csv", index=False)
        with open(RES / "resnet_results.json", "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2, ensure_ascii=False)
    return full


def aggregate_unet():
    rows, full = [], {}
    for v in ("rgb", "rgbfreq"):
        p = RES / f"unet_{v}_results.json"
        if p.exists():
            r = load_json(p); full[v] = r
            row = dict(variant=v, **r["overall"])
            for k, g in r["by_tool"].items():
                row[f"f1_tool_{k}"] = g.get("pixel_f1_at_0p5"); row[f"auprc_tool_{k}"] = g.get("pixel_auprc_pooled_subsampled")
            for k, g in r["by_area_bin"].items():
                row[f"f1_area_{k}"] = g.get("pixel_f1_at_0p5"); row[f"auprc_area_{k}"] = g.get("pixel_auprc_pooled_subsampled")
            row.update({f"res_{k}": v2 for k, v2 in r["resources"].items()})
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(RES / "unet_results.csv", index=False)
        with open(RES / "unet_results.json", "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2, ensure_ascii=False)
    return full


def resource_table(resnet, unet):
    rows = []
    cj = RES / "classical_results.json"
    if cj.exists():
        c = load_json(cj)
        for m in ("svm_rbf", "random_forest"):
            if m in c:
                rows.append(dict(model=m, train_time_s=c[m].get("train_time_s"), peak_gpu_mem_gib=0.0, n_params=None,
                                 infer_ms_per_sample=c["meta"]["feature_extract_time_s"] / (c["meta"]["n_train"] + c["meta"]["n_val"]) * 1000,
                                 ckpt_mb=None, note="CPU; çıkarım süresi = öznitelik çıkarımı/patch"))
    for s, r in resnet.items():
        rows.append(dict(model=f"resnet18_{s}", train_time_s=r["train_time_s"], peak_gpu_mem_gib=r["peak_gpu_mem_gib"], n_params=r["n_params"],
                         infer_ms_per_sample=r["infer_ms_per_patch"], ckpt_mb=r["ckpt_mb"], note=f"{r['epochs_run']}/{r['epochs_planned']} epoch, 224 patch, batch 1 bf16"))
    for v, r in unet.items():
        q = r["resources"]
        rows.append(dict(model=f"unet_{v}", train_time_s=q["train_time_s"], peak_gpu_mem_gib=q["peak_gpu_mem_gib"], n_params=q["n_params"],
                         infer_ms_per_sample=q["infer_ms_per_crop"], ckpt_mb=q["ckpt_mb"], note=f"{q['epochs_run']}/{q['epochs_planned']} epoch, 512 crop, batch 1 bf16"))
    df = pd.DataFrame(rows); df.to_csv(RES / "resource_usage.csv", index=False)
    return df


# ------------------------------------------------------------------ 2. dayanıklılık (E)
def robustness(device):
    import torch
    from torch.utils.data import DataLoader
    from train_unet import UNetWrapper, predict_records
    splits = D.load_splits(); cache = D.ImageCache()
    val_idx = pd.read_csv(RES / "seg_val_crop_index.csv", dtype={"new_id": str})
    ds = D.SegCropDataset(splits["smoke_val"], cache, 512, train=False, val_index=val_idx)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=D.seg_collate)
    conds = {"clean (PNG)": None, "JPEG Q75": lambda im: D.degrade_jpeg(im, 75),
             "rescale 0.75x→1x": lambda im: D.degrade_rescale(im, 0.75)}
    rows, full = [], {}
    for v in ("rgb", "rgbfreq"):
        ck = D.MODELS_DIR / f"unet_{v}_best.pt"
        if not ck.exists():
            print("checkpoint yok:", ck); continue
        model = UNetWrapper(v).to(device); model.load_state_dict(torch.load(ck, map_location=device)); model.eval()
        for cname, fn in conds.items():
            recs = predict_records(model, dl, device, degrade=fn)
            s = segmentation_summary(recs)
            bt = grouped_segmentation([r for r in recs if r["meta"]["crop_type"] == "bbox"], "tool")
            row = dict(variant=v, condition=cname, **{k: s.get(k) for k in ("pixel_f1_at_0p5", "iou_at_0p5", "oracle_f1",
                       "global_best_thr_f1", "pixel_auprc_pooled_subsampled", "pixel_auroc_pooled_subsampled", "clean_crop_fp_pixel_rate_at_0p5")})
            for t, g in bt.items():
                row[f"f1_tool_{t}"] = g.get("pixel_f1_at_0p5")
            rows.append(row); full[f"{v}|{cname}"] = s
            print(v, cname, {k: round(val, 4) for k, val in row.items() if isinstance(val, float)}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "robustness_results.csv", index=False)
    with open(RES / "robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    return df


# ------------------------------------------------------------------ 4. figürler
def fig_training_curves_resnet():
    logs = {s: pd.read_csv(LOG / f"resnet_{s}.csv") for s in ("probe", "ft", "scratch") if (LOG / f"resnet_{s}.csv").exists()}
    if not logs: return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for s, df in logs.items():
        axes[0].plot(df.epoch, df.train_loss, color=COL[s], marker="o", ms=2.5, lw=1.2, label=NAME[s])
        axes[1].plot(df.epoch, df.val_loss, color=COL[s], marker="o", ms=2.5, lw=1.2)
        axes[2].plot(df.epoch, df.val_auroc, color=COL[s], marker="o", ms=2.5, lw=1.2)
    axes[0].set_ylabel("train BCE loss"); axes[1].set_ylabel("val BCE loss"); axes[2].set_ylabel("val AUROC")
    for ax in axes: ax.set_xlabel("epoch"); ax.margins(0.04)
    axes[2].set_ylim(0.45, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ResNet18 patch classification on smoke_train→smoke_val (n=1587 train / 296 val patches)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "training_curves_resnet.png"); plt.close(fig)


def fig_training_curves_unet():
    logs = {v: pd.read_csv(LOG / f"unet_{v}.csv") for v in ("rgb", "rgbfreq") if (LOG / f"unet_{v}.csv").exists()}
    if not logs: return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for v, df in logs.items():
        axes[0].plot(df.epoch, df.train_loss, color=COL[v], marker="o", ms=2.5, lw=1.2, label=NAME[v])
        axes[1].plot(df.epoch, df.val_f1, color=COL[v], marker="o", ms=2.5, lw=1.2)
        axes[2].plot(df.epoch, df.val_auprc, color=COL[v], marker="o", ms=2.5, lw=1.2)
    axes[0].set_ylabel("train BCE + Dice loss"); axes[1].set_ylabel("val Pixel-F1 @0.5"); axes[2].set_ylabel("val Pixel-AUPRC (pooled)")
    for ax in axes: ax.set_xlabel("epoch"); ax.margins(0.04)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("U-Net (ResNet18 encoder) localization, 512 px native crops (400 train / 100 val images)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "training_curves_unet.png"); plt.close(fig)


def fig_confusion_resnet(resnet):
    if not resnet: return
    fig, axes = plt.subplots(1, len(resnet), figsize=(2.4 * len(resnet), 2.5))
    axes = np.atleast_1d(axes)
    for ax, (s, r) in zip(axes, resnet.items()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["clean", "tampered"]); ax.set_yticklabels(["clean", "tampered"])
        ax.set_xlabel("predicted (thr 0.5)"); ax.set_title(f"{NAME[s]}\nAUROC {r['auroc']:.3f}, bal. acc {r['balanced_accuracy']:.3f}")
        for sp in ax.spines.values(): sp.set_visible(False)
    axes[0].set_ylabel("ground truth")
    fig.tight_layout(); fig.savefig(FIG / "confusion_resnet.png"); plt.close(fig)


def fig_rf_importance():
    p = RES / "rf_feature_importance.csv"
    if not p.exists(): return
    imp = pd.read_csv(p).head(15).iloc[::-1]
    fam = imp.feature.str.split("_").str[0]
    fcol = {"dct": "#1b6ca8", "lbp": "#d1495b", "glcm": "#edae49", "lap": "#00798c", "fft": "#6c757d"}
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.barh(imp.feature, imp.importance, color=[fcol[f] for f in fam])
    ax.set_xlabel("RandomForest Gini importance"); ax.margins(0.04)
    ax.set_title("Top-15 handcrafted features (DCT=blue, LBP=red, GLCM=yellow, Laplacian=teal, FFT=grey)")
    fig.tight_layout(); fig.savefig(FIG / "rf_feature_importance.png"); plt.close(fig)


def fig_unet_predictions(n=6):
    probs = {v: np.load(D.CACHE_DIR / f"unet_{v}_val_probs.npz") for v in ("rgb", "rgbfreq") if (D.CACHE_DIR / f"unet_{v}_val_probs.npz").exists()}
    if not probs: return
    val_idx = pd.read_csv(RES / "seg_val_crop_index.csv", dtype={"new_id": str})
    cache = D.ImageCache()
    bbox_rows = val_idx[val_idx.crop_type == "bbox"].reset_index()
    rng = np.random.default_rng(3)
    # tool ve kaynak çeşitliliği: her tool'dan 3
    pick = []
    for t in sorted(bbox_rows.tool.unique()):
        sub = bbox_rows[bbox_rows.tool == t]
        pick += list(rng.choice(sub["index"].values, min(3, len(sub)), replace=False))
    pick = pick[:n]
    cols = 2 + len(probs)
    fig, axes = plt.subplots(len(pick), cols, figsize=(1.7 * cols, 1.75 * len(pick)))
    for r_i, idx in enumerate(pick):
        r = val_idx.iloc[idx]
        img = D.crop(D.pad_to_min(cache.image(r.new_id), 512), int(r.x0), int(r.y0), 512)
        gt = list(probs.values())[0]["gt"][idx]
        axes[r_i, 0].imshow(img); axes[r_i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        for c_i, (v, z) in enumerate(probs.items()):
            axes[r_i, 2 + c_i].imshow(z["prob"][idx] / 255.0, cmap="magma", vmin=0, vmax=1)
        axes[r_i, 0].set_ylabel(f"{r.new_id}\n{r.tool}, {r.source}", fontsize=6)
        for ax in axes[r_i]: ax.set_xticks([]); ax.set_yticks([])
    titles = ["forged crop (512 px)", "GT mask (bbox)"] + [NAME[v].replace("U-Net ", "") for v in probs]
    for ax, t in zip(axes[0], titles): ax.set_title(t, fontsize=7)
    fig.suptitle("U-Net predicted tamper probability on smoke_val bbox-containing crops", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "unet_predictions.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()
    D.ensure_dirs(); style()
    resnet = aggregate_resnet(); unet = aggregate_unet()
    resource_table(resnet, unet)
    if not args.skip_robustness and unet:
        import torch
        robustness(torch.device("cuda"))
    fig_training_curves_resnet(); fig_training_curves_unet(); fig_confusion_resnet(resnet); fig_rf_importance(); fig_unet_predictions()
    print("evaluate done")


if __name__ == "__main__":
    main()
