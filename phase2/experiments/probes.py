"""
probes.py — H2 shortcut probeları (TR).

1) Sınır ablasyonu dönüşümleri (forged val görüntüsü + bbox maskesi):
     boundary_blur : halka = dilate(bbox,3) − erode(bbox,2) (≈5 px, bbox kenarını saran) → Gaussian σ=1.5 ile yumuşatılmış kopya
     placebo_ring  : aynı halka şekli, bbox ile kesişmeyen rastgele konumda (bulunamazsa görüntü atlanır)
     inner_blur    : erode(bbox,3) içi blur (glif dokusu bozulur, sınır korunur)
   → run_eval.py bu dönüşümleri 'ablation' aşamasında uygular; sonuç probes_boundary.csv
2) Rectangle-fit son-işleme (kaydedilmiş val tahminleri): τ* eşik → en büyük bileşen → bbox doldur → F1/IoU (3 GT)
     python probes.py rectfit --models m1,m2
3) Benign kontroller image-level analizi (run_eval.py 'controls' aşamasının per_image_controls.csv çıktısından):
     python probes.py controls --models m1,m2
"""
from __future__ import annotations
import os, sys, json, argparse, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.p2data as D
import common.p2metrics as M

K3 = np.ones((3, 3), np.uint8)
ABLATIONS = ["boundary_blur", "placebo_ring", "inner_blur"]


def ring_mask(bbox_mask: np.ndarray) -> np.ndarray:
    d = cv2.dilate(bbox_mask, K3, iterations=3)
    e = cv2.erode(bbox_mask, K3, iterations=2)
    return (d > 0) & (e == 0)


def place_placebo(bbox_mask: np.ndarray, rng: np.random.Generator, max_try: int = 100):
    """Halka şeklini (bbox ile aynı boyutta) bbox'ın 3 px genişletilmişiyle kesişmeyen rastgele konuma taşır."""
    x0, y0, x1, y1 = D.mask_bbox(bbox_mask)
    H, W = bbox_mask.shape
    bw, bh = x1 - x0, y1 - y0
    if bw + 6 >= W or bh + 6 >= H:
        return None
    for _ in range(max_try):
        nx = int(rng.integers(3, W - bw - 3)); ny = int(rng.integers(3, H - bh - 3))
        if nx + bw + 3 <= x0 - 3 or nx - 3 >= x1 + 3 or ny + bh + 3 <= y0 - 3 or ny - 3 >= y1 + 3:
            m = np.zeros_like(bbox_mask); m[ny:ny + bh, nx:nx + bw] = 1
            return ring_mask(m)
    return None


def apply_ablation(img: np.ndarray, bbox_mask: np.ndarray, kind: str, rng: np.random.Generator):
    """Dönüştürülmüş görüntü (ya da None: uygulanamadı) ve etkilenen piksel sayısı."""
    blurred = cv2.GaussianBlur(img, (0, 0), 1.5)
    if kind == "boundary_blur":
        region = ring_mask(bbox_mask)
    elif kind == "placebo_ring":
        region = place_placebo(bbox_mask, rng)
        if region is None:
            return None, 0
    elif kind == "inner_blur":
        region = cv2.erode(bbox_mask, K3, iterations=3) > 0
    else:
        raise ValueError(kind)
    out = img.copy(); out[region] = blurred[region]
    return out, int(region.sum())


# ------------------------------------------------------------------ rect-fit
def rectfit_models(models, gts=("bbox", "pixel_eps8_morph", "glyph_v2"), threads=8):
    tau = json.load(open(D.RESULTS_DIR / "tau.json", "r", encoding="utf-8"))
    tab = D.split_table("val"); ids = list(tab.index)
    rows = []
    for model in models:
        pdir = D.PRED_DIR / model / "val"
        t_bbox = tau[model]["bbox"]

        def work(i):
            p = cv2.imread(str(pdir / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            out = []
            if p is None:
                return out
            H, W = p.shape
            for g in gts:
                gt = D.read_gt(i, g, "val", shape=(H, W))
                if gt is None or not gt.any():
                    continue
                tg = tau[model][g]
                raw = (p >= int(round(tg * 255))).astype(np.uint8)
                f_raw, i_raw = M.binary_f1_iou(raw, gt)
                rf = M.rect_fit(p, tg); f_rf, i_rf = M.binary_f1_iou(rf, gt)
                # tek eşik: bbox-GT τ*'ı ile üretilen dikdörtgen, tüm GT'lere uygulanır (uygulamada tek eşik kullanılır)
                rf_b = M.rect_fit(p, t_bbox); f_rfb, i_rfb = M.binary_f1_iou(rf_b, gt)
                out.append(dict(new_id=i, model=model, gt=g, tau=tg, f1_raw=f_raw, iou_raw=i_raw, f1_rectfit=f_rf, iou_rectfit=i_rf,
                                f1_rectfit_taubbox=f_rfb, iou_rectfit_taubbox=i_rfb, rect_area=int(rf.sum()), raw_area=int(raw.sum()),
                                group=tab.loc[i, "group"], tool=tab.loc[i, "tool"], source=tab.loc[i, "source"], area_bin=tab.loc[i, "area_bin"]))
            return out
        with ThreadPoolExecutor(threads) as ex:
            for o in ex.map(work, ids):
                rows.extend(o)
        print(f"[rectfit] {model} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D.RESULTS_DIR / "per_image_rectfit.csv", index=False)
    summ = df.groupby(["model", "gt"]).agg(n=("new_id", "count"), tau=("tau", "first"), f1_raw=("f1_raw", "mean"), f1_rectfit=("f1_rectfit", "mean"),
                                            iou_raw=("iou_raw", "mean"), iou_rectfit=("iou_rectfit", "mean"),
                                            f1_rectfit_taubbox=("f1_rectfit_taubbox", "mean")).reset_index()
    summ["delta_f1"] = summ.f1_rectfit - summ.f1_raw
    summ["delta_iou"] = summ.iou_rectfit - summ.iou_raw
    summ["train_gt"] = summ.model.map(model_train_gt)
    summ.to_csv(D.RESULTS_DIR / "probes_rectfit.csv", index=False)
    print(summ.round(3).to_string())
    return summ


def model_train_gt(model: str) -> str:
    for g in ["pixel_eps8_morph", "glyph_v2", "bbox"]:
        if f"unet_{g}_" in model:
            return g
    return "?"


# ------------------------------------------------------------------ controls (image-level)
def controls_analysis(models):
    """per_image_controls.csv (run_eval 'controls') + per_image_val.csv (forged) → image-level AUROC ve FP tabloları."""
    pc = pd.read_csv(D.RESULTS_DIR / "per_image_controls.csv", dtype={"new_id": str})
    pv = pd.read_csv(D.RESULTS_DIR / "per_image_val.csv", dtype={"new_id": str})
    pv = pv[pv['gt'] == "bbox"]  # image-level skorlar GT'den bağımsız
    rows = []
    for model in models:
        f = pv[pv.model == model]
        for src, c in pc[pc.model == model].groupby("source"):
            r = dict(model=model, train_gt=model_train_gt(model), source=src, n_control=len(c), n_forged=len(f),
                     fp_pixel_rate_0p5=float(c.pos_frac_0p5.mean()), fp_pixel_rate_tau=float(c.pos_frac_tau.mean()) if "pos_frac_tau" in c else np.nan,
                     frac_imgs_gt0p1pct_0p5=float((c.pos_frac_0p5 > 0.001).mean()),
                     frac_imgs_gt0p1pct_tau=float((c.pos_frac_tau > 0.001).mean()) if "pos_frac_tau" in c else np.nan,
                     score_max_median=float(c.score_max.median()), score_top1_median=float(c.score_top1pct.median()),
                     forged_score_max_median=float(f.score_max.median()), forged_score_top1_median=float(f.score_top1pct.median()))
            for sc in ["score_max", "score_top1pct", "pos_frac_0p5"]:
                y = np.r_[np.ones(len(f)), np.zeros(len(c))]; s = np.r_[f[sc].values, c[sc].values]
                r[f"auroc_{sc}"] = float(roc_auc_score(y, s))
            # tool kırılımı (forged tool'u; kontroller aynı val id'lerinden türediği için eşleşir)
            for tool in ["gemini", "ideogram"]:
                ft = f[f.tool == tool]; ct = c[c.new_id.isin(ft.new_id)]
                if len(ft) and len(ct):
                    y = np.r_[np.ones(len(ft)), np.zeros(len(ct))]; s = np.r_[ft.score_top1pct.values, ct.score_top1pct.values]
                    r[f"auroc_top1_{tool}"] = float(roc_auc_score(y, s))
            rows.append(r)
        # forged içi referans: pozitif piksel oranı
        rows.append(dict(model=model, train_gt=model_train_gt(model), source="forged_val", n_control=len(f), n_forged=len(f),
                         fp_pixel_rate_0p5=float(f.pos_frac_0p5.mean()), frac_imgs_gt0p1pct_0p5=float((f.pos_frac_0p5 > 0.001).mean()),
                         score_max_median=float(f.score_max.median()), score_top1_median=float(f.score_top1pct.median())))
    out = pd.DataFrame(rows)
    out.to_csv(D.RESULTS_DIR / "probes_controls.csv", index=False)
    print(out.round(4).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rectfit", "controls"])
    ap.add_argument("--models", required=True)
    a = ap.parse_args()
    ms = a.models.split(",")
    if a.cmd == "rectfit":
        rectfit_models(ms)
    else:
        controls_analysis(ms)
