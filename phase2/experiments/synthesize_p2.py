"""
synthesize_p2.py — Sonuç sentezi (TR): seed ort ± std özet tabloları, held-out generator, kaynak kullanımı, figürler, README.

Girdi: results/{summary_val,summary_test,per_image_val,per_image_test,robustness,probes_boundary,probes_rectfit,
       probes_controls,h4_results}.csv, results/train_*.json, results/hists/*.npz, logs/train_*.csv
Çıktı: results/{summary_val_agg,summary_test_agg,heldout_generator,robustness_agg,probes_boundary_agg,
       probes_rectfit_agg,probes_controls_agg,resource_usage}.csv, figures/*.png, README_tables.md
"""
from __future__ import annotations
import os, sys, json, glob, time
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common.p2data as D
import common.p2metrics as M

R, F, L = D.RESULTS_DIR, D.FIG_DIR, D.LOG_DIR
GTS = ["bbox", "pixel_eps8_morph", "glyph_v2"]
GT_SHORT = {"bbox": "bbox", "pixel_eps8_morph": "pixel", "glyph_v2": "glyph"}
COL = {"bbox": "#4C72B0", "pixel_eps8_morph": "#DD8452", "glyph_v2": "#8172B3"}  # mavi / turuncu / mor (CVD-güvenli; kırmızı-yeşil yok)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                     "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": False})


def parse_model(m: str):
    gt = next(g for g in ["pixel_eps8_morph", "glyph_v2", "bbox"] if f"unet_{g}_" in m)
    seed = int(m.split("_s")[-1].split("_")[0])
    return gt, seed, ("geminionly" in m)


def add_model_cols(df):
    p = df.model.map(parse_model)
    df["train_gt"] = p.map(lambda t: t[0]); df["seed"] = p.map(lambda t: t[1]); df["gemini_only"] = p.map(lambda t: t[2])
    return df


def agg(df, keys, metrics):
    g = df.groupby(keys)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g["model"].nunique()
    return out.reset_index()


def fmt(m, s, nd=3):
    if pd.isna(m):
        return "—"
    return f"{m:.{nd}f}" if pd.isna(s) or s == 0 else f"{m:.{nd}f} ± {s:.{nd}f}"


def md_table(df, cols, headers=None, nd=3):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pooled_subset(model, suffix, gt, ids):
    """suffix: 'val' | 'test' | 'val+test' — ilgili split(ler)in histogramları toplanarak havuzlanmış AUPRC."""
    hp, ha = [], []
    for sfx in suffix.split("+"):
        p = R / "hists" / f"{model}_{sfx}_{gt}.npz"
        if not p.exists():
            continue
        z = np.load(p); sel = np.isin(z["ids"], np.array(list(ids), dtype=z["ids"].dtype))
        if sel.sum():
            hp.append(z["h_pos"][sel]); ha.append(z["h_all"][sel])
    if not hp:
        return np.nan
    return M.pooled_from_hists(np.concatenate(hp), np.concatenate(ha))["auprc_pooled"]


def main():
    F.mkdir(exist_ok=True)
    md = []
    # ------------------------------------------------------------ 1. val özet (seed ort ± std)
    sv = add_model_cols(pd.read_csv(R / "summary_val.csv"))
    B = sv[~sv.gemini_only]
    metrics = ["auprc_pooled", "auroc_pooled", "ap_img", "f1_0p5", "iou_0p5", "oracle_f1", "f1_tau", "iou_tau", "rect_0p5", "ncc_0p5", "rect_tau", "tau"]
    metrics = [m for m in metrics if m in B]
    va = agg(B[B.subset == "all"], ["train_gt", "gt"], metrics)
    va.to_csv(R / "summary_val_agg.csv", index=False)
    vsub = agg(B[B.subset != "all"], ["train_gt", "gt", "subset"], [m for m in ["auprc_pooled", "f1_0p5", "f1_tau", "ap_img"] if m in B])
    vsub.to_csv(R / "summary_val_subsets_agg.csv", index=False)
    rows = []
    for _, r in va.iterrows():
        rows.append(dict(train_gt=GT_SHORT[r.train_gt], eval_gt=GT_SHORT[r['gt']], n_seeds=r.n_seeds,
                         AUPRC=fmt(r.auprc_pooled_mean, r.auprc_pooled_std), AUROC=fmt(r.auroc_pooled_mean, r.auroc_pooled_std),
                         AP_img=fmt(r.ap_img_mean, r.ap_img_std), F1_05=fmt(r.f1_0p5_mean, r.f1_0p5_std), IoU_05=fmt(r.iou_0p5_mean, r.iou_0p5_std),
                         OracleF1=fmt(r.oracle_f1_mean, r.oracle_f1_std), tau=fmt(r.tau_mean, r.tau_std, 2), F1_tau=fmt(r.f1_tau_mean, r.f1_tau_std),
                         IoU_tau=fmt(r.iou_tau_mean, r.iou_tau_std) if "iou_tau_mean" in r else "—"))
    T1 = pd.DataFrame(rows)
    md.append("## T1. Val (487, tam görüntü) — eğitim GT × değerlendirme GT (seed ort ± std, n_seeds=3)\n\n" +
              md_table(T1, list(T1.columns)))
    # sıralama değişimi: her eval GT'de en iyi train_gt
    rank = va.pivot(index="train_gt", columns="gt", values="auprc_pooled_mean")
    md.append("\nAUPRC sıralaması (eval GT sütunlarına göre en iyi eğitim GT'si): " +
              "; ".join(f"{GT_SHORT[c]}-GT → {GT_SHORT[rank[c].idxmax()]}-model ({rank[c].max():.3f})" for c in rank.columns))
    # ------------------------------------------------------------ 2. test özet
    st = add_model_cols(pd.read_csv(R / "summary_test.csv"))
    Bt = st[~st.gemini_only]
    tm = [m for m in ["auprc_pooled", "auroc_pooled", "ap_img", "f1_0p5", "iou_0p5", "oracle_f1", "f1_tau", "iou_tau"] if m in Bt]
    ta = agg(Bt[Bt.subset.isin(["all", "clean_test", "paired_89"])], ["train_gt", "gt", "subset"], tm)
    ta["n_images"] = [int(Bt[(Bt.train_gt == r.train_gt) & (Bt['gt'] == getattr(r, 'gt')) & (Bt.subset == r.subset)].n_with_pos.iloc[0]) for r in ta.itertuples()]
    ta.to_csv(R / "summary_test_agg.csv", index=False)
    rows = []
    for _, r in ta.iterrows():
        if r['gt'] != "bbox" and r.subset != "paired_89":
            continue
        rows.append(dict(train_gt=GT_SHORT[r.train_gt], eval_gt=GT_SHORT[r['gt']], subset=r.subset, n_img=r.n_images,
                         AUPRC=fmt(r.auprc_pooled_mean, r.auprc_pooled_std), AP_img=fmt(r.ap_img_mean, r.ap_img_std),
                         F1_05=fmt(r.f1_0p5_mean, r.f1_0p5_std), F1_tau=fmt(r.f1_tau_mean, r.f1_tau_std), IoU_tau=fmt(r.iou_tau_mean, r.iou_tau_std),
                         OracleF1=fmt(r.oracle_f1_mean, r.oracle_f1_std)))
    T2 = pd.DataFrame(rows)
    md.append("\n## T2. Resmî test (812) — bbox-GT: tam 812 ve temiz 740; 89 çiftte piksel/glif GT (τ* val'den)\n\n" + md_table(T2, list(T2.columns)))
    # ------------------------------------------------------------ 3. held-out generator
    pv = pd.read_csv(R / "per_image_val.csv", dtype={"new_id": str}); pt = pd.read_csv(R / "per_image_test.csv", dtype={"new_id": str})
    pv["split"] = "val"; pt["split"] = "test"
    pi = pd.concat([pv, pt], ignore_index=True); pi = add_model_cols(pi)
    pi = pi[pi.n_pos > 0]
    ideo = pi[pi.tool == "ideogram"]
    gem_matched = pi[(pi.tool == "gemini") & (pi.source.isin(["cord", "sroie", "xfund"]))]
    hrows = []
    for model, d in pi.groupby("model"):
        gt_tr, seed, go = parse_model(model)
        for g in GTS:
            di = ideo[(ideo.model == model) & (ideo['gt'] == g)]; dg = gem_matched[(gem_matched.model == model) & (gem_matched['gt'] == g)]
            if len(di) == 0:
                continue
            # alan dilimine göre yeniden ağırlıklandırma: Gemini alt kümesini Ideogram alan-dilimi dağılımına ağırlıkla
            w_bins = di.area_bin.value_counts(normalize=True)
            dg_w = dg.area_bin.map(lambda b: w_bins.get(b, 0.0) / max((dg.area_bin == b).mean(), 1e-9))
            def pooled_mix(dsub):
                # val ve test id uzayları çakışır → split başına ayrı seçim, histogramlar toplanır
                hp, ha = [], []
                for sfx in ["val", "test"]:
                    ids_s = dsub[dsub.split == sfx].new_id
                    pth = R / "hists" / f"{model}_{sfx}_{g}.npz"
                    if len(ids_s) and pth.exists():
                        z = np.load(pth); sel = np.isin(z["ids"], np.array(list(ids_s), dtype=z["ids"].dtype))
                        if sel.sum():
                            hp.append(z["h_pos"][sel]); ha.append(z["h_all"][sel])
                return M.pooled_from_hists(np.concatenate(hp), np.concatenate(ha))["auprc_pooled"] if hp else np.nan
            hrows.append(dict(model=model, train_gt=gt_tr, seed=seed, gemini_only=go, gt=g,
                              n_ideogram=len(di), n_gemini_matched=len(dg),
                              ideogram_f1_0p5=di.f1_0p5.mean(), ideogram_f1_tau=di.f1_tau.mean(), ideogram_ap_img=di.ap_img.mean(),
                              ideogram_auprc_pooled=pooled_mix(di),
                              gemini_matched_f1_0p5=dg.f1_0p5.mean(), gemini_matched_f1_tau=dg.f1_tau.mean(), gemini_matched_ap_img=dg.ap_img.mean(),
                              gemini_matched_auprc_pooled=pooled_mix(dg),
                              gemini_matched_f1_0p5_areaweighted=np.average(dg.f1_0p5, weights=dg_w) if dg_w.sum() > 0 else np.nan,
                              gemini_matched_ap_img_areaweighted=np.average(dg.ap_img, weights=dg_w) if dg_w.sum() > 0 else np.nan,
                              ideogram_val_n=int((di.split == "val").sum()), ideogram_test_n=int((di.split == "test").sum())))
    H = pd.DataFrame(hrows)
    H["gap_f1_0p5"] = H.gemini_matched_f1_0p5 - H.ideogram_f1_0p5
    H["gap_ap_img"] = H.gemini_matched_ap_img - H.ideogram_ap_img
    H.to_csv(R / "heldout_generator.csv", index=False)
    rows = []
    for (go, gt_tr, g), d in H.groupby(["gemini_only", "train_gt", "gt"]):
        rows.append(dict(model=("C: gemini-only" if go else "B: tüm araçlar"), train_gt=GT_SHORT[gt_tr], eval_gt=GT_SHORT[g], n_seeds=len(d),
                         n_ideo=int(d.n_ideogram.iloc[0]), ideo_AP=fmt(d.ideogram_ap_img.mean(), d.ideogram_ap_img.std() if len(d) > 1 else np.nan),
                         ideo_F1_05=fmt(d.ideogram_f1_0p5.mean(), d.ideogram_f1_0p5.std() if len(d) > 1 else np.nan),
                         ideo_AUPRC=fmt(d.ideogram_auprc_pooled.mean(), d.ideogram_auprc_pooled.std() if len(d) > 1 else np.nan),
                         gem_AP=fmt(d.gemini_matched_ap_img.mean(), d.gemini_matched_ap_img.std() if len(d) > 1 else np.nan),
                         gem_F1_05=fmt(d.gemini_matched_f1_0p5.mean(), d.gemini_matched_f1_0p5.std() if len(d) > 1 else np.nan),
                         gem_F1_05_areaw=fmt(d.gemini_matched_f1_0p5_areaweighted.mean(), np.nan),
                         gap_AP=fmt(d.gap_ap_img.mean(), np.nan)))
    T3 = pd.DataFrame(rows)
    md.append("\n## T3. Held-out generator — Ideogram (val 41 + test 85) vs kaynak-eşlenmiş Gemini (cord/sroie/xfund; val+test), görüntü-içi ortalama\n\n" + md_table(T3, list(T3.columns)))
    # ------------------------------------------------------------ 4. dayanıklılık
    if (R / "robustness.csv").exists():
        rb = add_model_cols(pd.read_csv(R / "robustness.csv"))
        rm = [m for m in ["auprc_pooled", "f1_0p5", "f1_tau", "ap_img", "delta_auprc_pooled", "delta_f1_0p5", "delta_f1_tau", "delta_ap_img", "pos_frac_0p5"] if m in rb]
        ra = agg(rb, ["train_gt", "gt", "condition"], rm); ra.to_csv(R / "robustness_agg.csv", index=False)
        rows = []
        for _, r in ra.iterrows():
            if r['gt'] not in ("bbox", "pixel_eps8_morph"):
                continue
            rows.append(dict(train_gt=GT_SHORT[r.train_gt], eval_gt=GT_SHORT[r['gt']], condition=r.condition,
                             AUPRC=fmt(r.auprc_pooled_mean, r.auprc_pooled_std), dAUPRC=fmt(r.delta_auprc_pooled_mean, r.delta_auprc_pooled_std),
                             F1_tau=fmt(r.f1_tau_mean, r.f1_tau_std), dF1_tau=fmt(r.delta_f1_tau_mean, r.delta_f1_tau_std),
                             AP_img=fmt(r.ap_img_mean, r.ap_img_std), dAP=fmt(r.delta_ap_img_mean, r.delta_ap_img_std)))
        T4 = pd.DataFrame(rows)
        md.append("\n## T4. Dayanıklılık (val 487; eğitim yok): Δ = bozulmuş − temiz (seed ort ± std)\n\n" + md_table(T4, list(T4.columns)))
    # ------------------------------------------------------------ 5. sınır ablasyonu
    if (R / "probes_boundary.csv").exists():
        pb = add_model_cols(pd.read_csv(R / "probes_boundary.csv"))
        bm = [m for m in ["auprc_pooled", "auprc_pooled_clean", "delta_auprc_pooled", "rel_delta_auprc_pooled", "f1_tau", "f1_tau_clean", "delta_f1_tau", "rel_delta_f1_tau",
                          "ap_img", "ap_img_clean", "delta_ap_img", "rel_delta_ap_img", "n_affected_px_mean"] if m in pb]
        ba = agg(pb, ["train_gt", "gt", "condition"], bm); ba.to_csv(R / "probes_boundary_agg.csv", index=False)
        rows = []
        for _, r in ba.iterrows():
            rows.append(dict(train_gt=GT_SHORT[r.train_gt], eval_gt=GT_SHORT[r['gt']], condition=r.condition, n=int(pb[(pb.train_gt == r.train_gt) & (pb['gt'] == r['gt']) & (pb.condition == r.condition)].n.iloc[0]),
                             AUPRC_clean=fmt(r.auprc_pooled_clean_mean, r.auprc_pooled_clean_std), AUPRC=fmt(r.auprc_pooled_mean, r.auprc_pooled_std),
                             relD_AUPRC=fmt(r.rel_delta_auprc_pooled_mean, r.rel_delta_auprc_pooled_std),
                             AP_clean=fmt(r.ap_img_clean_mean, r.ap_img_clean_std), AP=fmt(r.ap_img_mean, r.ap_img_std), relD_AP=fmt(r.rel_delta_ap_img_mean, r.rel_delta_ap_img_std),
                             F1tau_clean=fmt(r.f1_tau_clean_mean, r.f1_tau_clean_std), F1tau=fmt(r.f1_tau_mean, r.f1_tau_std), relD_F1tau=fmt(r.rel_delta_f1_tau_mean, r.rel_delta_f1_tau_std)))
        T5 = pd.DataFrame(rows)
        md.append("\n## T5. Sınır ablasyonu (H2; val 487): boundary_blur = bbox kenarı 5 px halka blur σ=1.5; placebo_ring = aynı halka bbox dışında; inner_blur = erode(bbox,3) içi blur. relΔ = (ablasyon − temiz)/temiz, aynı görüntü kümesi\n\n" + md_table(T5, list(T5.columns)))
    # ------------------------------------------------------------ 6. rect-fit
    if (R / "probes_rectfit.csv").exists():
        rf = add_model_cols(pd.read_csv(R / "probes_rectfit.csv"))
        rfa = agg(rf, ["train_gt", "gt"], ["f1_raw", "f1_rectfit", "delta_f1", "iou_raw", "iou_rectfit", "delta_iou", "f1_rectfit_taubbox"]); rfa.to_csv(R / "probes_rectfit_agg.csv", index=False)
        rows = [dict(train_gt=GT_SHORT[r.train_gt], eval_gt=GT_SHORT[r['gt']], F1_raw_tau=fmt(r.f1_raw_mean, r.f1_raw_std), F1_rectfit=fmt(r.f1_rectfit_mean, r.f1_rectfit_std),
                     dF1=fmt(r.delta_f1_mean, r.delta_f1_std), IoU_raw=fmt(r.iou_raw_mean, r.iou_raw_std), IoU_rectfit=fmt(r.iou_rectfit_mean, r.iou_rectfit_std), dIoU=fmt(r.delta_iou_mean, r.delta_iou_std))
                for _, r in rfa.iterrows()]
        T6 = pd.DataFrame(rows)
        # rectangularity dağılımı (per_image_val, eval gt=bbox satırları; şekil metrikleri GT'den bağımsız)
        pr = pv[pv['gt'] == "bbox"].copy(); pr = add_model_cols(pr)
        rq = pr.groupby("train_gt")[["rect_0p5", "rect_tau", "ncc_0p5", "ncc_tau"]].agg(["median", "mean"])
        rq.columns = [f"{a}_{b}" for a, b in rq.columns]; rq = rq.reset_index()
        rq.to_csv(R / "pred_rectangularity_by_traingt.csv", index=False)
        md.append("\n## T6. Rectangle-fit son-işleme (val; τ* eşik → en büyük bileşen → bbox doldur)\n\n" + md_table(T6, list(T6.columns)) +
                  "\n\nTahmin dikdörtgenselliği (val, görüntü başına alan/bbox-alanı; medyan / ortalama) ve bileşen sayısı, eğitim GT'sine göre:\n\n" +
                  md_table(rq.assign(train_gt=rq.train_gt.map(GT_SHORT)), list(rq.columns)))
    # ------------------------------------------------------------ 7. kontroller
    if (R / "probes_controls.csv").exists():
        pc = add_model_cols(pd.read_csv(R / "probes_controls.csv"))
        cm = [m for m in ["fp_pixel_rate_0p5", "fp_pixel_rate_tau", "frac_imgs_gt0p1pct_0p5", "frac_imgs_gt0p1pct_tau", "score_max_median", "score_top1_median",
                          "auroc_score_max", "auroc_score_top1pct", "auroc_pos_frac_0p5", "auroc_top1_gemini", "auroc_top1_ideogram"] if m in pc]
        ca = agg(pc, ["train_gt", "source"], cm); ca.to_csv(R / "probes_controls_agg.csv", index=False)
        rows = [dict(train_gt=GT_SHORT[r.train_gt], source=r.source, FPpx_05=fmt(r.fp_pixel_rate_0p5_mean, r.fp_pixel_rate_0p5_std, 4),
                     FPpx_tau=fmt(r.fp_pixel_rate_tau_mean, r.fp_pixel_rate_tau_std, 4), imgs_gt0p1pct_05=fmt(r.frac_imgs_gt0p1pct_0p5_mean, r.frac_imgs_gt0p1pct_0p5_std),
                     imgs_gt0p1pct_tau=fmt(r.frac_imgs_gt0p1pct_tau_mean, r.frac_imgs_gt0p1pct_tau_std), top1_median=fmt(r.score_top1_median_mean, r.score_top1_median_std),
                     AUROC_max=fmt(r.auroc_score_max_mean, r.auroc_score_max_std), AUROC_top1=fmt(r.auroc_score_top1pct_mean, r.auroc_score_top1pct_std),
                     AUROC_top1_gem=fmt(r.auroc_top1_gemini_mean, r.auroc_top1_gemini_std), AUROC_top1_ideo=fmt(r.auroc_top1_ideogram_mean, r.auroc_top1_ideogram_std))
                for _, r in ca.iterrows()]
        T7 = pd.DataFrame(rows)
        md.append("\n## T7. Benign kontroller (val 487 kaynaklı): FP piksel oranı, >%0.1 pozitif içeren görüntü payı, image-level AUROC (forged val vs kontrol)\n\n" + md_table(T7, list(T7.columns)))
    # ------------------------------------------------------------ 8. H4
    if (R / "h4_results.csv").exists():
        h4 = pd.read_csv(R / "h4_results.csv"); h4["probe"] = "outside_bbox (H4)"
        if (R / "h4_results_poscontrol.csv").exists():
            h4 = pd.concat([h4, pd.read_csv(R / "h4_results_poscontrol.csv").assign(probe="positive_control (bbox-centred crops)")], ignore_index=True)
        md.append("\n## T8. H4 global-iz probu (ResNet-18 ft, 224 crop): forged vs eşleşen authentic / fullimage_resample — 'outside_bbox' = maskeye değmeyen crop (H4), 'positive_control' = bbox-merkezli crop (pipeline doğrulaması). acc_pairwise: p(forged) > p(negatif) olan çift payı (eşitlik = başarısız; cord/xfund'da bbox-dışı pikseller özdeş → 0)\n\n" +
                  md_table(h4, ["probe", "condition", "subset", "n_pairs", "auroc", "acc_pairwise", "n_train_pairs"]))
    # ------------------------------------------------------------ 9. kaynak kullanımı
    rows = []
    for f in sorted(glob.glob(str(R / "train_*.json"))):
        j = json.load(open(f)); rows.append(dict(run=j["run"], n_train=j["n_train"], epochs_run=j["epochs_run"], best_epoch=j["best_epoch"], best_val_crop_auprc=j["best_val_auprc"],
                                                 train_time_min=j["train_time_s"] / 60, peak_gpu_mem_gib=j["peak_gpu_mem_gib"], n_params=j["n_params"], ckpt_mb=j.get("ckpt_mb")))
    ru = pd.DataFrame(rows)
    if (L / "infer_log.csv").exists():
        il = pd.read_csv(L / "infer_log.csv")
        inf = il.groupby("stage").agg(n_images=("new_id", "count"), gpu_s_total=("gpu_s", "sum"), gpu_s_per_img=("gpu_s", "mean"), n_resized=("resized", "sum"), windows_mean=("n_windows", "mean")).reset_index()
        inf.to_csv(R / "resource_usage_inference.csv", index=False)
        md.append("\n## Çıkarım kaynak kullanımı (aşama başına; 3 model aynı anda, GPU süresi görüntü başına)\n\n" + md_table(inf, list(inf.columns)))
    ru.to_csv(R / "resource_usage.csv", index=False)
    md.append("\n## Eğitim kaynak kullanımı\n\n" + md_table(ru, list(ru.columns), nd=2))
    md.append(f"\nToplam eğitim GPU süresi: {ru.train_time_min.sum():.0f} dk ({len(ru)} run).")
    # ------------------------------------------------------------ figürler
    make_figures(sv, B, va, H, pv)
    (D.EXP_DIR / "README_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def make_figures(sv, B, va, H, pv):
    # F1: eğitim eğrileri
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for f in sorted(glob.glob(str(L / "train_unet_*.csv"))):
        name = os.path.basename(f)[6:-4]
        if "dryrun" in name:
            continue
        gt, seed, go = parse_model(name); d = pd.read_csv(f)
        ls = "--" if go else "-"; lab = f"{GT_SHORT[gt]} s{seed}" + (" (gem-only)" if go else "")
        axes[0].plot(d.epoch, d.train_loss, ls, color=COL[gt], alpha=0.8, label=lab)
        axes[1].plot(d.epoch, d.val_auprc, ls, color=COL[gt], alpha=0.8)
        axes[2].plot(d.epoch, d.val_f1, ls, color=COL[gt], alpha=0.8)
    axes[0].set_title("Train loss (BCE+Dice)"); axes[1].set_title("Val crop AUPRC (eğitim GT'si)"); axes[2].set_title("Val crop F1@0.5 (eğitim GT'si)")
    for a in axes: a.set_xlabel("epoch"); a.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(F / "fig_training_curves.png", dpi=150); plt.close(fig)
    # F2: çift-GT metrik karşılaştırması (train_gt × eval_gt)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, met, title in zip(axes, ["auprc_pooled", "f1_tau", "ap_img"], ["Pixel-AUPRC (havuz)", "F1@τ*", "AP (görüntü-içi ort.)"]):
        x = np.arange(3); w = 0.25
        for k, tg in enumerate(GTS):
            sub = va[va.train_gt == tg].set_index("gt").reindex(GTS)
            ax.bar(x + (k - 1) * w, sub[f"{met}_mean"], w, yerr=sub[f"{met}_std"], color=COL[tg], label=f"model: {GT_SHORT[tg]}-GT", capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([f"eval: {GT_SHORT[g]}" for g in GTS]); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Val 487 — aynı tahminler, üç GT ile değerlendirme (seed ort ± std)")
    fig.tight_layout(); fig.savefig(F / "fig_dual_gt_metrics.png", dpi=150); plt.close(fig)
    # F3: ablasyon Δ
    if (R / "probes_boundary_agg.csv").exists():
        ba = pd.read_csv(R / "probes_boundary_agg.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, met, title in zip(axes, ["rel_delta_auprc_pooled", "rel_delta_ap_img"], ["relΔ Pixel-AUPRC", "relΔ AP (görüntü-içi)"]):
            conds = ["boundary_blur", "placebo_ring", "inner_blur"]; x = np.arange(len(conds)); w = 0.13; i = 0
            for tg in GTS:
                for eg in ["bbox", "pixel_eps8_morph"]:
                    sub = ba[(ba.train_gt == tg) & (ba['gt'] == eg)].set_index("condition").reindex(conds)
                    ax.bar(x + (i - 2.5) * w, sub[f"{met}_mean"], w, yerr=sub[f"{met}_std"], color=COL[tg], alpha=1.0 if eg == "bbox" else 0.5,
                           label=f"{GT_SHORT[tg]}-model / eval {GT_SHORT[eg]}", capsize=2); i += 1
            ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(F / "fig_ablation_delta.png", dpi=150); plt.close(fig)
    # F4: held-out generator
    if len(H):
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(3); w = 0.2; i = 0
        for go, lab in [(False, "B (tüm araçlar)"), (True, "C (gemini-only)")]:
            for tg in GTS:
                sub = H[(H.gemini_only == go) & (H.train_gt == tg) & (H['gt'] == "pixel_eps8_morph")]
                vals = [sub.ideogram_ap_img.mean(), sub.gemini_matched_ap_img.mean(), sub.gemini_matched_ap_img_areaweighted.mean()]
                ax.bar(x + (i - 2.5) * w * 0.65, vals, w * 0.65, color=COL[tg], alpha=0.5 if go else 1.0, hatch="//" if go else None, label=f"{lab} — {GT_SHORT[tg]}-model"); i += 1
        ax.set_xticks(x); ax.set_xticklabels(["Ideogram (held-out)\npiksel GT'li n=41 (val)", "Gemini, kaynak-eşlenmiş\n(cord/sroie/xfund; val + 89 test çifti)", "Gemini, kaynak-eşlenmiş\nIdeogram alan-dilimine ağırlıklı"])
        ax.set_ylabel("AP (görüntü-içi ort., eval GT: pixel)"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(F / "fig_heldout_generator.png", dpi=150); plt.close(fig)
    # F5: dayanıklılık
    if (R / "robustness_agg.csv").exists():
        ra = pd.read_csv(R / "robustness_agg.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, eg in zip(axes, ["bbox", "pixel_eps8_morph"]):
            conds = ["jpeg75", "rescale075", "blur1"]; x = np.arange(3); w = 0.25
            for k, tg in enumerate(GTS):
                sub = ra[(ra.train_gt == tg) & (ra['gt'] == eg)].set_index("condition").reindex(conds)
                ax.bar(x + (k - 1) * w, sub["delta_auprc_pooled_mean"], w, yerr=sub["delta_auprc_pooled_std"], color=COL[tg], label=f"{GT_SHORT[tg]}-model", capsize=2)
            ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_title(f"Δ AUPRC vs temiz — eval GT: {GT_SHORT[eg]}"); ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(F / "fig_robustness.png", dpi=150); plt.close(fig)
    # F6: kontrol skor dağılımları
    if (R / "per_image_controls.csv").exists():
        pc = add_model_cols(pd.read_csv(R / "per_image_controls.csv", dtype={"new_id": str}))
        pvv = add_model_cols(pv[pv['gt'] == "bbox"].copy())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, tg in zip(axes, GTS):
            data = [pvv[(pvv.train_gt == tg) & (pvv.seed == 42)].score_top1pct.values]
            labels = ["forged val"]
            for src in ["authentic", "benign_pasteback", "fullimage_resample"]:
                data.append(pc[(pc.train_gt == tg) & (pc.seed == 42) & (pc.source == src)].score_top1pct.values); labels.append(src)
            ax.boxplot(data, tick_labels=labels, showfliers=False); ax.set_title(f"{GT_SHORT[tg]}-model (s42): top-1% ort. skor"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(F / "fig_controls_scores.png", dpi=150); plt.close(fig)
    # F7: örnek tahminler (8 val görüntüsü; s42 modelleri)
    tab = D.split_table("val"); tau = json.load(open(R / "tau.json"))
    rng = np.random.default_rng(3)
    cand = pv[(pv.model == "unet_bbox_s42") & (pv['gt'] == "bbox") & (pv.n_pos > 0)]
    picks = []
    for tool in ["gemini", "ideogram"]:
        for src in ["cord", "sroie", "wildreceipt", "xfund"]:
            c = cand[(cand.tool == tool) & (cand.source == src)]
            if len(c):
                picks.append(c.sample(1, random_state=int(rng.integers(1e6))).new_id.iloc[0])
    picks = picks[:8]
    if len(picks) < 8:
        picks += list(cand[~cand.new_id.isin(picks)].sample(8 - len(picks), random_state=1).new_id)
    models = ["unet_bbox_s42", "unet_pixel_eps8_morph_s42", "unet_glyph_v2_s42"]
    fig, axes = plt.subplots(len(picks), 7, figsize=(17, 2.4 * len(picks)))
    for r, i in enumerate(picks):
        img = D.read_rgb(D.img_path(i, "val")); bm = D.read_mask(D.bbox_mask_path(i, "val"))
        x0, y0, x1, y1 = D.mask_bbox(bm); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        S = max(256, int(max(x1 - x0, y1 - y0) * 2.5)); H_, W_ = bm.shape
        xa, ya = max(0, min(cx - S // 2, W_ - S)), max(0, min(cy - S // 2, H_ - S))
        sl = (slice(ya, ya + S), slice(xa, xa + S))
        panels = [("forged", img[sl]), ("bbox GT", bm[sl] * 255), ("pixel GT", (D.read_gt(i, "pixel_eps8_morph", "val", bm.shape) if D.read_gt(i, "pixel_eps8_morph", "val", bm.shape) is not None else np.zeros_like(bm))[sl] * 255),
                  ("glyph GT", (D.read_gt(i, "glyph_v2", "val", bm.shape) if D.read_gt(i, "glyph_v2", "val", bm.shape) is not None else np.zeros_like(bm))[sl] * 255)]
        for m in models:
            p = cv2.imread(str(D.PRED_DIR / m / "val" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            panels.append((f"pred {GT_SHORT[parse_model(m)[0]]}-model", p[sl] if p is not None else np.zeros_like(bm[sl])))
        for c, (t, a) in enumerate(panels):
            ax = axes[r, c]; ax.imshow(a, cmap=None if a.ndim == 3 else "gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(t, fontsize=9)
            if c == 0: ax.set_ylabel(f"{i}\n{tab.loc[i, 'tool']}/{tab.loc[i, 'source']}", fontsize=7)
    fig.tight_layout(); fig.savefig(F / "fig_examples_predictions.png", dpi=110); plt.close(fig)
    # F8: H4
    if (R / "h4_per_image_val.csv").exists():
        h = pd.read_csv(R / "h4_per_image_val.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(h.p_forged, bins=30, alpha=0.6, label="forged (bbox dışı crop)"); axes[0].hist(h.p_authentic, bins=30, alpha=0.6, label="authentic (aynı koordinat)")
        axes[0].set_title("H4 — val skor dağılımı"); axes[0].legend(fontsize=8)
        axes[1].hist(h.p_forged_rs.dropna(), bins=30, alpha=0.6, label="forged"); axes[1].hist(h.p_resample.dropna(), bins=30, alpha=0.6, label="fullimage_resample (simetrik)")
        axes[1].set_title("H4 — negatif = yeniden-örneklenmiş authentic"); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(F / "fig_h4_scores.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
