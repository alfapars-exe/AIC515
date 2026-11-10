"""
run_eval.py — Değerlendirme orkestratörü (TR): bir grup model için tüm aşamalar, görüntü başına tek decode.

  python run_eval.py --models m1,m2,m3 --stages val,test,robust,controls,ablation

Aşamalar:
  val       : val 487 tam-görüntü çıkarım → preds/<m>/val/<id>.png → evaluate_p2 (τ* kalibrasyonu + 3 GT metrikleri)
  test      : test 812 → preds/<m>/test/<id>.png → evaluate_p2 (τ* val'den; bbox-GT tam 812 / temiz 740; 89 çiftte piksel-GT)
  robust    : val'de JPEG Q75, 0.75× yeniden ölçek, Gaussian blur σ=1 → inline metrikler (3 GT) → results/robustness.csv
  controls  : benign_pasteback / fullimage_resample / authentic (val 487) → image-level + FP metrikleri → per_image_controls.csv
  ablation  : sınır-blur / plasebo / iç-blur (val 487) → 3 GT metrikleri → probes_boundary.csv (+ per_image_boundary.csv)
Tüm aşamalar inline hesaplanır; yalnız val/test olasılık haritaları diske yazılır.
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("USERNAME", "user")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import torch
import common.p2data as D
import common.p2metrics as M
import infer_fullimage as I
import evaluate_p2 as E
import probes as P

GTS = D.EVAL_GTS


def iterate_images(ids, loader, lookahead=6, threads=6):
    """Thread prefetch ile (id, payload) üretir; loader(id) → payload."""
    pool = ThreadPoolExecutor(threads)
    futs = {k: pool.submit(loader, ids[k]) for k in range(min(lookahead, len(ids)))}
    for k, i in enumerate(ids):
        pay = futs.pop(k).result()
        if k + lookahead < len(ids):
            futs[k + lookahead] = pool.submit(loader, ids[k + lookahead])
        yield i, pay
    pool.shutdown(wait=True)


def stage_predict_save(names, models, split, device, log_rows, tab=None):
    tab = D.split_table(split) if tab is None else tab
    ids = list(tab.index)
    out_dirs = {n: D.PRED_DIR / n / split for n in names}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    writer = ThreadPoolExecutor(4)
    t0 = time.time()
    for k, (i, img) in enumerate(iterate_images(ids, lambda i: D.read_rgb(D.img_path(i, split)))):
        t = time.time(); probs, info = I.predict_full_multi(models, img, device); tg = time.time() - t
        for n, p in zip(names, probs):
            writer.submit(I.write_png, out_dirs[n] / f"{i}.png", I.to_u8(p))
        log_rows.append(dict(stage=split, new_id=i, gpu_s=tg, **info))
        if (k + 1) % 100 == 0:
            print(f"  [{split}] {k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    writer.shutdown(wait=True)
    print(f"[{split}] predicted {len(ids)} images × {len(names)} models in {time.time() - t0:.0f}s", flush=True)


def stage_inline(names, models, split, device, condition_fn, cond_name, gts, taus, tab, ids, source_loader,
                 with_gt=True, log_rows=None, stage="robust"):
    """Her görüntü için: yükle → condition_fn(img, meta) → tüm modeller → inline metrikler. Dönüş: per-image DataFrame."""
    rows = []
    rng = np.random.default_rng(12345)
    t0 = time.time()
    gt_pool = ThreadPoolExecutor(4)

    def load(i):
        img = source_loader(i)
        bm = D.read_mask(D.bbox_mask_path(i, split))
        gts_i = {g: D.read_gt(i, g, split, shape=bm.shape) for g in gts} if with_gt else {}
        return img, bm, gts_i
    def metrics_job(i, n, pu8, gts_i, n_aff):
        out = []
        if with_gt:
            r, hs = E.eval_one(i, pu8, split, gts, taus[n], gt_cache=gts_i)
            for g in gts:
                m = dict(r[g]); m.update(new_id=i, model=n, split=split, condition=cond_name, gt=g, n_affected_px=n_aff,
                                         tool=tab.loc[i, "tool"], source=tab.loc[i, "source"], area_bin=tab.loc[i, "area_bin"], group=tab.loc[i, "group"])
                m["h_pos"] = hs[g][0] if hs[g] is not None else None; m["h_all"] = hs[g][1] if hs[g] is not None else None
                out.append(m)
        else:
            m = M.image_level_scores(pu8)
            tb = taus[n].get("bbox", 0.5)
            m["pos_frac_tau"] = float((pu8 >= int(round(tb * 255))).mean())
            m["rect_0p5"], m["ncc_0p5"] = M.rectangularity(pu8 >= 128)
            m.update(new_id=i, model=n, split=split, condition=cond_name, tool=tab.loc[i, "tool"], source_doc=tab.loc[i, "source"],
                     area_bin=tab.loc[i, "area_bin"], group=tab.loc[i, "group"], tau_bbox=tb)
            out.append(m)
        return out
    # metrikler GPU ile örtüşsün: thread havuzu (bincount/np ops GIL'i büyük ölçüde bırakır), sınırlı kuyruk
    mpool = ThreadPoolExecutor(6); futs = []
    for k, (i, (img, bm, gts_i)) in enumerate(iterate_images(ids, load)):
        img2, n_aff = condition_fn(img, bm, rng)
        if img2 is None:
            continue
        t = time.time(); probs, info = I.predict_full_multi(models, img2, device); tg = time.time() - t
        if log_rows is not None:
            log_rows.append(dict(stage=f"{stage}:{cond_name}", new_id=i, gpu_s=tg, **info))
        for n, p in zip(names, probs):
            futs.append(mpool.submit(metrics_job, i, n, I.to_u8(p), gts_i, n_aff))
        while len(futs) > 60:  # bellek: en fazla ~20 görüntülük tahmin kuyrukta
            rows.extend(futs.pop(0).result())
        if (k + 1) % 100 == 0:
            print(f"  [{stage}:{cond_name}] {k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    for f in futs:
        rows.extend(f.result())
    mpool.shutdown(wait=True)
    print(f"[{stage}:{cond_name}] {len(ids)} images × {len(names)} models in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarize_inline(df: pd.DataFrame, extra_keys=("condition",)) -> pd.DataFrame:
    out = []
    keys = ["model", "gt", *extra_keys]
    for key, d in df.groupby(keys):
        pos = d[d.n_pos > 0]
        hp = np.stack([h for h in d.h_pos if h is not None]); ha = np.stack([h for h in d.h_all if h is not None])
        pooled = M.pooled_from_hists(hp, ha)
        row = dict(zip(keys, key)); row.update(n=len(d), n_with_pos=len(pos), **pooled)
        for c in ["ap_img", "auroc_img", "f1_0p5", "iou_0p5", "oracle_f1", "f1_tau", "iou_tau"]:
            row[c] = float(pos[c].mean()) if c in pos and len(pos) else np.nan
        for c in ["pos_frac_0p5", "score_max", "score_top1pct"]:
            row[c] = float(d[c].mean())
        for tool in ["gemini", "ideogram"]:
            dt = pos[pos.tool == tool]
            if len(dt):
                row[f"f1_0p5_{tool}"] = float(dt.f1_0p5.mean()); row[f"ap_img_{tool}"] = float(dt.ap_img.mean())
                hp_t = np.stack(dt.h_pos.tolist()); ha_t = np.stack(dt.h_all.tolist())
                row[f"auprc_pooled_{tool}"] = M.pooled_from_hists(hp_t, ha_t)["auprc_pooled"]
        out.append(row)
    return pd.DataFrame(out)


def merge_csv(path, df, key_cols=("model",)):
    if path.exists():
        old = pd.read_csv(path, dtype={"new_id": str})
        mask = np.ones(len(old), bool)
        for c in key_cols:
            mask &= old[c].isin(df[c].unique())
        df = pd.concat([old[~mask], df], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--stages", default="val,test,robust,controls,ablation")
    ap.add_argument("--limit", type=int, default=0, help="debug: ilk N görüntü")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    names = args.models.split(",")
    models = [I.load_model(n, device) for n in names]
    stages = args.stages.split(",")
    D.ensure_dirs()
    log_rows = []
    T0 = time.time()
    val_tab = D.split_table("val")
    if args.limit:
        val_tab = val_tab.iloc[: args.limit]
    val_ids = list(val_tab.index)

    if "val" in stages:
        stage_predict_save(names, models, "val", device, log_rows, val_tab)
        for n in names:
            pdir = D.PRED_DIR / n / "val"
            E.evaluate_preds(n, "val", "clean", GTS, lambda i, pdir=pdir: E.read_pred(pdir / f"{i}.png"), "val", val_tab)
    if "test" in stages:
        test_tab = D.split_table("test")
        if args.limit:
            test_tab = test_tab.iloc[: args.limit]
        stage_predict_save(names, models, "test", device, log_rows, test_tab)
        for n in names:
            pdir = D.PRED_DIR / n / "test"
            E.evaluate_preds(n, "test", "clean", GTS, lambda i, pdir=pdir: E.read_pred(pdir / f"{i}.png"), "test", test_tab)
    taus = E.load_tau()
    taus = {n: taus.get(n, {}) for n in names}
    if "robust" in stages:
        dfs = []
        for cond, fn in D.DEGRADATIONS.items():
            df = stage_inline(names, models, "val", device, lambda img, bm, rng, fn=fn: (fn(img), 0), cond, GTS, taus, val_tab, val_ids,
                              lambda i: D.read_rgb(D.img_path(i, "val")), True, log_rows, "robust")
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        summ = summarize_inline(df)
        # temiz referans: summary_val.csv (subset=all)
        sv = pd.read_csv(D.RESULTS_DIR / "summary_val.csv")
        sv = sv[(sv.subset == "all") & (sv.model.isin(names))][["model", "gt", "auprc_pooled", "f1_0p5", "f1_tau", "ap_img", "oracle_f1"]]
        summ = summ.merge(sv.rename(columns={c: f"{c}_clean" for c in ["auprc_pooled", "f1_0p5", "f1_tau", "ap_img", "oracle_f1"]}), on=["model", "gt"], how="left")
        for c in ["auprc_pooled", "f1_0p5", "f1_tau", "ap_img"]:
            summ[f"delta_{c}"] = summ[c] - summ[f"{c}_clean"]
        merge_csv(D.RESULTS_DIR / "robustness.csv", summ)
        merge_csv(D.RESULTS_DIR / "per_image_robustness.csv", df.drop(columns=["h_pos", "h_all", "f1_curve"], errors="ignore"))
    if "controls" in stages:
        dfs = []
        for src in ["benign_pasteback", "fullimage_resample", "authentic"]:
            def loader(i, src=src):
                if src == "authentic":
                    return D.read_rgb(D.auth_path(val_tab.loc[i, "pair_name"], "val"))
                return D.read_rgb(D.control_img_path(src, i))
            df = stage_inline(names, models, "val", device, lambda img, bm, rng: (img, 0), src, GTS, taus, val_tab, val_ids, loader, False, log_rows, "controls")
            df["source"] = src
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        merge_csv(D.RESULTS_DIR / "per_image_controls.csv", df)
        P.controls_analysis(names) if (D.RESULTS_DIR / "per_image_val.csv").exists() else None
    if "ablation" in stages:
        dfs = []
        for kind in P.ABLATIONS:
            df = stage_inline(names, models, "val", device, lambda img, bm, rng, kind=kind: P.apply_ablation(img, bm, kind, rng), kind, GTS, taus,
                              val_tab, val_ids, lambda i: D.read_rgb(D.img_path(i, "val")), True, log_rows, "ablation")
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        summ = summarize_inline(df)
        summ["n_affected_px_mean"] = [float(df[(df.model == r.model) & (df['gt'] == getattr(r, 'gt')) & (df.condition == r.condition)].n_affected_px.mean()) for r in summ.itertuples()]
        # temiz referans: AYNI görüntü kümesi üzerinde (plasebo bazı görüntüleri atlayabilir) per_image_val + hists
        pv = pd.read_csv(D.RESULTS_DIR / "per_image_val.csv", dtype={"new_id": str})
        clean_cols = {c: [] for c in ["auprc_pooled_clean", "f1_0p5_clean", "f1_tau_clean", "ap_img_clean"]}
        for r in summ.itertuples():
            ids_c = df[(df.model == r.model) & (df['gt'] == getattr(r, 'gt')) & (df.condition == r.condition)].new_id.unique()
            pvs = pv[(pv.model == r.model) & (pv['gt'] == getattr(r, 'gt')) & (pv.new_id.isin(ids_c)) & (pv.n_pos > 0)]
            z = np.load(D.RESULTS_DIR / "hists" / f"{r.model}_val_{getattr(r, 'gt')}.npz")
            sel = np.isin(z["ids"], ids_c)
            clean_cols["auprc_pooled_clean"].append(M.pooled_from_hists(z["h_pos"][sel], z["h_all"][sel])["auprc_pooled"])
            clean_cols["f1_0p5_clean"].append(float(pvs.f1_0p5.mean())); clean_cols["f1_tau_clean"].append(float(pvs.f1_tau.mean()))
            clean_cols["ap_img_clean"].append(float(pvs.ap_img.mean()))
        for c, v in clean_cols.items():
            summ[c] = v
        for c in ["auprc_pooled", "f1_0p5", "f1_tau", "ap_img"]:
            summ[f"delta_{c}"] = summ[c] - summ[f"{c}_clean"]
            summ[f"rel_delta_{c}"] = summ[f"delta_{c}"] / summ[f"{c}_clean"]
        summ["train_gt"] = summ.model.map(P.model_train_gt)
        merge_csv(D.RESULTS_DIR / "probes_boundary.csv", summ)
        merge_csv(D.RESULTS_DIR / "per_image_boundary.csv", df.drop(columns=["h_pos", "h_all", "f1_curve"], errors="ignore"))
    if "rectfit" in stages:
        P.rectfit_models(names)
    if log_rows:
        lg = pd.DataFrame(log_rows); lg["models"] = "+".join(names)
        merge_csv(D.LOG_DIR / "infer_log.csv", lg, key_cols=("models",))
    print(f"run_eval done {names} stages={stages} in {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
