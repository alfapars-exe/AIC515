"""
probes_p3.py — Faz 3 sondaları (TR), val eşleşmiş çiftleri (forged ve authentic aynı boyut), görüntü başına tek decode.

Koşullar (model girdisi):
  forged                : ham sahte görüntü
  forged_repaired       : işaretçi kalıntısı authentic'ten geri yazılmış (Faz 2 Ek-C ile aynı)
  render_only           : sahte görüntüde piksel-değişim bölgesi (pixel_eps8_morph ∪ işaretçi, 2 px genişletilmiş) authentic'e döndürülmüş
                          → içerik değişimi YOK; yalnız üreticinin ε≤8 altı yeniden-render izi ve bbox-içi doku kalır   (PHASE3-d)
  content_only          : authentic görüntüye yalnız piksel-değişim bölgesi (işaretçisiz, pixel_eps8_morph_nm, 2 px genişletilmiş) yapıştırılmış
                          → içerik değişimi VAR; çevredeki yeniden-render izi ve işaretçi YOK                                       (PHASE3-d)
  authentic, authentic_plus_marker
  clahe, gray           : sahte görüntüye test-zamanı CLAHE (L kanalı) / gri seviye (Faz 2 §6.4 'test edilmedi')
Ölçümler (her model × görüntü × koşul): bbox-GT ve piksel-GT AP (görüntü içi), F1@τ*(bbox), bbox içi ort. olasılık,
bbox kapsama@τ*, görüntü skoru top-1 % (99. persentil) ve max.  → phase3/results/per_image_probes_p3.csv + özet.
Kullanım: python probes_p3.py --unet unet_bbox_s42,... [--yolo aiforge-tiles-yolo11s-img1024-conserv-s42,...] [--limit N]
"""
from __future__ import annotations
import sys, json, time, argparse, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3ext as X  # noqa: E402
import numpy as np, pandas as pd, cv2, torch
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score
D, I = X.D, X.I
OUT = X.OUT / "results"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def ap_img(prob, gt, nneg=300000):
    pos = prob[gt]; neg = prob[~gt]
    if pos.size == 0:
        return np.nan
    if neg.size > nneg:
        neg = neg[rng.choice(neg.size, nneg, replace=False)]
    return float(average_precision_score(np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg]))


def f1(pred, gt):
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="")
    ap.add_argument("--yolo", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    unets = [m for m in args.unet.split(",") if m]
    nets = [I.load_model(m, dev) for m in unets]
    yolos = [m for m in args.yolo.split(",") if m]
    if yolos:
        sys.path.insert(0, str(X.HERE.parent / "phase3_yolo"))
        import infer_to_maps as Y
        from ultralytics import YOLO
        ymods = [YOLO(str(X.YOLO_OUT / "runs" / r / "weights" / "best.pt")) for r in yolos]
        yargs = types.SimpleNamespace(tile=1024, stride=768, full=0, conf=0.001, iou=0.6, batch=8)
    tau = json.load(open(D.RESULTS_DIR / "tau.json"))
    tab = D.split_table("val")
    ids = list(tab.index)[: args.limit or None]
    rows, t0 = [], time.time()
    for k, nid in enumerate(ids):
        pn = tab.loc[nid, "pair_name"]
        if not isinstance(pn, str):
            continue
        f = D.read_rgb(D.img_path(nid, "val")); a = D.read_rgb(D.auth_path(pn, "val"))
        if f.shape != a.shape:
            continue
        mb = D.read_mask(D.bbox_mask_path(nid, "val")) > 0
        pg = D.read_gt(nid, "pixel_eps8_morph", "val", shape=mb.shape)
        pg_nm = D.read_gt(nid, "pixel_eps8_morph_nm", "val", shape=mb.shape)
        if pg is None or pg.sum() == 0:
            continue
        pg = pg > 0; pg_nm = (pg_nm > 0) if pg_nm is not None else pg
        ring = ndi.binary_dilation(mb, iterations=6) & ~ndi.binary_erosion(mb, iterations=6)
        resd = ndi.binary_dilation(X.gmask(f) & ~X.gmask(a) & ring, iterations=1)
        rep = f.copy(); rep[resd] = a[resd]
        sel_r = ndi.binary_dilation(pg | resd, iterations=2)
        render_only = f.copy(); render_only[sel_r] = a[sel_r]
        sel_c = ndi.binary_dilation(pg_nm, iterations=2) & ~resd
        content_only = a.copy(); content_only[sel_c] = f[sel_c]
        am = a.copy(); am[resd] = f[resd]
        conds = {"forged": f, "forged_repaired": rep, "render_only": render_only, "content_only": content_only,
                 "authentic": a, "authentic_plus_marker": am, "clahe": X.degrade_clahe(f), "gray": X.degrade_gray(f)}
        meta = dict(new_id=nid, tool=tab.loc[nid, "tool"], source=tab.loc[nid, "source"], group=tab.loc[nid, "group"],
                    area_bin=tab.loc[nid, "area_bin"], has_residue=int(resd.sum() >= 20),
                    render_px_in_bbox_gt0=int(((np.abs(render_only.astype(int) - a.astype(int)).max(2) > 0) & mb).sum()))
        for cname, img in conds.items():
            outs = []
            if nets:
                probs, _ = I.predict_full_multi(nets, img, dev)
                outs += list(zip(unets, probs, ["unet"] * len(unets)))
            for r, ym in zip(yolos, ymods if yolos else []):
                b, c = Y.detect_full(ym, img, yargs)
                outs.append((r, Y.boxes_to_map(b, c, *img.shape[:2]).astype(np.float32) / 255.0, "yolo"))
            for m, p, fam in outs:
                tb = float(tau.get(m, {}).get("bbox", 0.5))
                pb = p >= tb
                rows.append(dict(**meta, cond=cname, model=m, family=fam, ap_bbox=ap_img(p, mb), ap_pixel=ap_img(p, pg),
                                 f1_bbox_tau=f1(pb, mb), mean_in_bbox=float(p[mb].mean()), bbox_cover_tau=float((pb & mb).sum() / max(mb.sum(), 1)),
                                 pos_frac_tau=float(pb.mean()), top1=float(np.percentile(p, 99)), score_max=float(p.max())))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    sfx = f"_{args.tag}" if args.tag else ""
    df.to_csv(OUT / f"per_image_probes_p3{sfx}.csv", index=False)
    # özet: koşul × model ortalamaları + görüntü düzeyi AUROC (koşul vs authentic, top-1 skor)
    S = df.groupby(["model", "family", "cond"]).agg(n=("new_id", "size"), ap_bbox=("ap_bbox", "mean"), ap_pixel=("ap_pixel", "mean"),
                                                   f1_bbox_tau=("f1_bbox_tau", "mean"), mean_in_bbox=("mean_in_bbox", "mean"),
                                                   bbox_cover_tau=("bbox_cover_tau", "mean"), pos_frac_tau=("pos_frac_tau", "mean"),
                                                   top1=("top1", "mean")).reset_index()
    au = []
    for (m, c), d in df.groupby(["model", "cond"]):
        if c == "authentic":
            continue
        a_ = df[(df.model == m) & (df.cond == "authentic")].set_index("new_id").score_max
        x_ = d.set_index("new_id").score_max
        common = x_.index.intersection(a_.index)
        if len(common) > 10:
            y = np.r_[np.ones(len(common)), np.zeros(len(common))]
            au.append(dict(model=m, cond=c, auroc_vs_authentic_max=roc_auc_score(y, np.r_[x_[common], a_[common]])))
    S = S.merge(pd.DataFrame(au), on=["model", "cond"], how="left")
    S.to_csv(OUT / f"probes_p3_summary{sfx}.csv", index=False)
    print(S.round(3).to_string(index=False))
    print(f"done {len(df)} rows {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
