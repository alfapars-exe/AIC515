"""
evaluate_p2.py — Kaydedilmiş olasılık haritalarını (preds/<model>/<suffix>/<id>.png) ÜÇ GT ile değerlendirir (TR).

  python evaluate_p2.py --models m1,m2 --split val [--condition clean|jpeg75|rescale075|blur1] [--gts bbox,pixel_eps8_morph,glyph_v2]

* τ* (val-kalibre global eşik): her (model, GT) için val temiz koşulda ortalama F1 eğrisinin argmax'ı → results/tau.json.
  Test ve diğer koşullarda bu τ* kullanılır (--split test ya da condition≠clean iken tau.json'dan okunur).
* Çıktı: results/per_image_<split>[_<condition>].csv'ye satır ekler (model, gt, id, group, tool, source, area_bin, metrikler),
  results/summary_<split>[_<condition>].csv (overall + tool/source/area_bin/clean_test alt kümeleri),
  results/hists/<model>_<suffix>_<gt>.npz (görüntü başına h_pos/h_all → bootstrap için).
Kütüphane: evaluate_preds(model, split, condition, gts, pred_loader) — probes.py yeniden kullanır.
"""
from __future__ import annotations
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.p2data as D
import common.p2metrics as M

TAU_PATH = D.RESULTS_DIR / "tau.json"
META_COLS = ["tool", "source", "area_bin", "group", "pair_name", "W", "H"]


def load_tau() -> Dict[str, Dict[str, float]]:
    if TAU_PATH.exists():
        return json.load(open(TAU_PATH, "r", encoding="utf-8"))
    return {}


def save_tau(tau):
    json.dump(tau, open(TAU_PATH, "w", encoding="utf-8"), indent=1)


def read_pred(path) -> np.ndarray:
    p = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if p is None:
        raise FileNotFoundError(path)
    return p


def eval_one(new_id: str, prob_u8: np.ndarray, split: str, gts: List[str], taus: Dict[str, Optional[float]],
             gt_cache: Optional[Dict[str, np.ndarray]] = None):
    """Bir görüntü, tüm GT'ler → (rows dict per gt, hists per gt)."""
    H, W = prob_u8.shape
    rows, hs = {}, {}
    for g in gts:
        gt = gt_cache[g] if gt_cache is not None and g in gt_cache else D.read_gt(new_id, g, split, shape=(H, W))
        m, h = M.per_image_metrics(prob_u8, gt, tau=taus.get(g), with_shape=(g == gts[0]))
        rows[g] = m; hs[g] = h
    return rows, hs


def summarize(df: pd.DataFrame, H: Dict[str, Dict[str, np.ndarray]], model: str, split: str, condition: str,
              subsets: Dict[str, pd.Series]) -> List[Dict]:
    """df: per-image satırları (model, gt sabit değil). H[gt] = {id: (h_pos,h_all)}. subsets: ad → id maskesi."""
    out = []
    for g, dg in df.groupby("gt"):
        for sname, smask in subsets.items():
            ids = [i for i in dg.new_id if smask.get(i, False)]
            d = dg[dg.new_id.isin(ids)]
            if len(d) == 0:
                continue
            pos = d[d.n_pos > 0]
            hl = [H[g][i] for i in d.new_id if H[g].get(i) is not None]
            if hl:
                pooled = M.pooled_from_hists(np.stack([h[0] for h in hl]), np.stack([h[1] for h in hl]))
            else:
                pooled = dict(auprc_pooled=np.nan, auroc_pooled=np.nan)
            row = dict(model=model, split=split, condition=condition, gt=g, subset=sname, n=len(d), n_with_pos=len(pos), **pooled)
            for k in ["ap_img", "auroc_img", "f1_0p5", "iou_0p5", "oracle_f1", "f1_tau", "iou_tau"]:
                if k in pos:
                    row[k] = float(pos[k].mean()) if len(pos) else np.nan
            for k in ["rect_0p5", "ncc_0p5", "rect_tau", "ncc_tau", "pos_frac_0p5", "score_max", "score_top1pct"]:
                if k in d:
                    row[k] = float(d[k].mean()) if len(d) else np.nan
            row["tau"] = float(d["tau"].iloc[0]) if "tau" in d else np.nan
            out.append(row)
    return out


def evaluate_preds(model: str, split: str, condition: str, gts: List[str], pred_loader: Callable[[str], np.ndarray],
                   suffix: str, tab: pd.DataFrame, threads: int = 8, write: bool = True, tag: Optional[str] = None):
    """pred_loader(new_id) → uint8 prob. τ*: val/clean'de hesaplanır ve kaydedilir; aksi hâlde tau.json'dan okunur."""
    taus_all = load_tau()
    calibrate = (split == "val" and condition == "clean" and tag is None)
    ids = list(tab.index)
    t0 = time.time()
    if calibrate:
        # 1. geçiş: yalnız F1 eğrileri → τ* (GT başına, ortalama eğri argmax)
        def work0(i):
            p = pred_loader(i); H_, W_ = p.shape
            out = {}
            for g in gts:
                gt = D.read_gt(i, g, split, shape=(H_, W_))
                if gt is None or not gt.any():
                    continue
                hp, ha = M.hists(p, gt); out[g] = M.curves_from_hist(hp, ha)[0]
            return out
        curves = {g: [] for g in gts}
        with ThreadPoolExecutor(threads) as ex:
            for out in ex.map(work0, ids):
                for g, c in out.items():
                    curves[g].append(c)
        taus_all.setdefault(model, {})
        for g in gts:
            taus_all[model][g] = M.choose_tau(np.stack(curves[g]))[0] if curves[g] else 0.5
        save_tau(taus_all)
        print(f"[tau* {model}] " + ", ".join(f"{g}={taus_all[model][g]:.2f}" for g in gts), flush=True)
    taus = {g: taus_all.get(model, {}).get(g) for g in gts}

    def work(i):
        p = pred_loader(i)
        return i, eval_one(i, p, split, gts, taus)
    rows, H = [], {g: {} for g in gts}
    with ThreadPoolExecutor(threads) as ex:
        for i, (r, hs) in ex.map(work, ids):
            for g in gts:
                m = dict(r[g]); m.update(new_id=i, model=model, split=split, condition=condition, gt=g)
                for c in META_COLS:
                    m[c] = tab.loc[i, c]
                if "clean_test" in tab:
                    m["clean_test"] = bool(tab.loc[i, "clean_test"])
                rows.append(m); H[g][i] = hs[g]
    df = pd.DataFrame(rows)
    df["tau"] = df["gt"].map(lambda g: taus_all.get(model, {}).get(g, np.nan))
    df = df.drop(columns=["f1_curve"])
    # alt kümeler
    subsets = {"all": pd.Series(True, index=ids)}
    for col in ["tool", "source", "area_bin"]:
        for v in sorted(tab[col].dropna().unique()):
            subsets[f"{col}={v}"] = (tab[col] == v)
    if "clean_test" in tab:
        subsets["clean_test"] = tab["clean_test"].astype(bool)
        subsets["paired_89"] = pd.Series([D.gt_path(i, "pixel_eps8_morph", "test") is not None for i in ids], index=ids)
    summ = summarize(df, H, model, split, condition, subsets)
    sdf = pd.DataFrame(summ)
    if write:
        D.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        base = f"{split}" + ("" if condition == "clean" else f"_{condition}") + (f"_{tag}" if tag else "")
        pi_path = D.RESULTS_DIR / f"per_image_{base}.csv"
        su_path = D.RESULTS_DIR / f"summary_{base}.csv"
        for path, d in ((pi_path, df), (su_path, sdf)):
            if path.exists():
                old = pd.read_csv(path, dtype={"new_id": str})
                old = old[old.model != model]
                d = pd.concat([old, d], ignore_index=True)
            d.to_csv(path, index=False)
        hd = D.RESULTS_DIR / "hists"; hd.mkdir(exist_ok=True)
        for g in gts:
            keep = [i for i in ids if H[g].get(i) is not None]
            np.savez_compressed(hd / f"{model}_{suffix}{('_' + tag) if tag else ''}_{g}.npz", ids=np.array(keep),
                                h_pos=np.stack([H[g][i][0] for i in keep]) if keep else np.zeros((0, 256)),
                                h_all=np.stack([H[g][i][1] for i in keep]) if keep else np.zeros((0, 256)))
    print(f"[eval {model} {suffix}] {len(ids)} images, {time.time() - t0:.0f}s; "
          + "; ".join(f"{r['gt']}: AUPRC {r['auprc_pooled']:.3f} F1@.5 {r.get('f1_0p5', np.nan):.3f}" for r in summ if r["subset"] == "all"), flush=True)
    return df, sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--gts", default=",".join(D.EVAL_GTS))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    gts = args.gts.split(",")
    tab = D.split_table(args.split)
    suffix = args.split + ("" if args.condition == "clean" else f"_{args.condition}")
    for model in args.models.split(","):
        pdir = D.PRED_DIR / model / suffix
        evaluate_preds(model, args.split, args.condition, gts, lambda i, pdir=pdir: read_pred(pdir / f"{i}.png"), suffix, tab, args.threads)


if __name__ == "__main__":
    main()
