# -*- coding: utf-8 -*-
"""Zero-shot evaluation of saved probability maps against AIForge-Doc v1 bbox masks.

All pixel metrics are computed from the uint8 maps via 256-bin histograms of (positive, negative) pixels, which
gives EXACT pooled/per-image AUROC, AP, F1(t), IoU(t) for every threshold t in 0..255 (scores are quantised
to 1/255 anyway). Threshold "0.5" == u8 value >= 128.  Additionally the task protocol variant (all positives +
<=50k sampled negatives per image, pooled) is reported as *_sub50k.
Per-image rows -> zero_shot_per_image.csv ; aggregated -> zero_shot_results.csv (long format).
"""
import os, sys, json, pathlib, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, cv2

LB = pathlib.Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")) / "phase2" / "literature_baselines"
DATA = pathlib.Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")) / "data" / "AIForge-Doc-v1"
AUD = pathlib.Path(os.environ.get("VIZE_ROOT", r"C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje")) / "dataset_audit"
T05 = 128  # u8 threshold equivalent to prob >= 0.5
CONDS = [c for c in sys.argv[1].split(',')] if len(sys.argv) > 1 else \
    ['trufor', 'adcd_q100_ocrproxy', 'adcd_q95_ocrproxy', 'adcd_q100_noaln', 'adcd_q100_emptyocr']

splits = json.load(open(AUD / "splits.json"))
sf = pd.read_csv(AUD / "samples_forged.csv")
sf['id'] = sf['new_id'].astype(int).astype(str).str.zfill(9)
meta = {}
for sp, sub in [('val', 'TrainingSet'), ('test', 'TestingSet')]:
    m = sf[(sf.split == ('training' if sp == 'val' else 'testing')) & sf.id.isin(splits[sp])].set_index('id')
    for i in splits[sp]:
        r = m.loc[i]
        meta[(sp, i)] = dict(split=sp, sub=sub, tool=r.assigned_tool, area_bin=r.area_bin, tamper_ratio=float(r.tamper_ratio),
                             source=r.source_dataset, W=int(r.W), H=int(r.H),
                             clean=bool(pd.isna(r.nn_train_any_dist) or r.nn_train_any_dist > 20))

def curves_from_hist(hp, hn):
    """hp, hn: 256 counts. Returns dict of threshold-indexed arrays (t=0..255 => pred = v>=t) and AUROC/AP."""
    P = hp.sum(); N = hn.sum()
    tp = np.cumsum(hp[::-1])[::-1].astype(np.float64)   # tp[t] = sum_{v>=t} hp
    fp = np.cumsum(hn[::-1])[::-1].astype(np.float64)
    fn = P - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    iou = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), 0.0)
    # ROC: points from t=256 (0,0) down to t=0 (1,1)
    tpr = np.concatenate([[0.0], (tp / max(P, 1))[::-1]]); fpr = np.concatenate([[0.0], (fp / max(N, 1))[::-1]])
    auroc = float(np.trapezoid(tpr, fpr)) if (P > 0 and N > 0) else float('nan')
    # AP (sklearn step definition): sum over decreasing thresholds of (R_k - R_{k-1}) * P_k
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)[::-1]; rec = (tp / max(P, 1))[::-1]
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    ap = float(np.sum((rec - rec_prev) * prec)) if P > 0 else float('nan')
    return dict(f1=f1, iou=iou, auroc=auroc, ap=ap, tp=tp, fp=fp, P=P, N=N)

def process(args):
    cond, sp, i = args
    mt = meta[(sp, i)]
    pred = cv2.imread(str(LB / "preds" / cond / sp / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(DATA / mt['sub'] / "masks" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None: return None
    if pred.shape != gt.shape: pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    g = gt > 127
    pv = pred[g]; nv = pred[~g]
    hp = np.bincount(pv, minlength=256); hn = np.bincount(nv, minlength=256)
    rng = np.random.default_rng(int(i))
    take = min(len(nv), 50000)
    nv_sub = rng.choice(nv, take, replace=False) if take < len(nv) else nv
    hn_sub = np.bincount(nv_sub, minlength=256)
    c = curves_from_hist(hp, hn)
    best = int(c['f1'].argmax())
    row = dict(cond=cond, split=sp, id=i, **{k: v for k, v in mt.items() if k not in ('sub', 'split')},
               pos_px=int(c['P']), neg_px=int(c['N']),
               f1_05=float(c['f1'][T05]), iou_05=float(c['iou'][T05]),
               oracle_f1=float(c['f1'][best]), oracle_thr=best / 255.0, oracle_iou=float(c['iou'].max()),
               auroc_img=c['auroc'], ap_img=c['ap'],
               pred_mean=float(pred.mean() / 255.0), pred_pos_frac_05=float((pred >= T05).mean()),
               mean_prob_in_gt=float(pv.mean() / 255.0), mean_prob_out_gt=float(nv.mean() / 255.0))
    return row, c['f1'], c['iou'], hp, hn, hn_sub

def aggregate(rows, f1c, iouc, hp, hn, hn_sub, thr_global=None):
    """rows: list of per-image dicts; f1c/iouc: (n,256) curves; hp/hn/hn_sub: (n,256) hists."""
    df = pd.DataFrame(rows)
    out = dict(n_images=len(df))
    if len(df) == 0: return out, None
    out.update(pixel_f1_05=df.f1_05.mean(), iou_05=df.iou_05.mean(), oracle_f1=df.oracle_f1.mean(), oracle_iou=df.oracle_iou.mean(),
               auroc_img_mean=df.auroc_img.mean(), ap_img_mean=df.ap_img.mean(),
               median_f1_05=df.f1_05.median(), frac_f1_05_gt_0p5=float((df.f1_05 > 0.5).mean()),
               pred_pos_frac_05=df.pred_pos_frac_05.mean(), gt_pos_frac=float((df.pos_px / (df.pos_px + df.neg_px)).mean()))
    mean_f1_curve = f1c.mean(0); gbest = int(mean_f1_curve.argmax())
    out.update(global_best_thr_self=gbest / 255.0, global_best_thr_self_f1=float(mean_f1_curve[gbest]))
    if thr_global is not None:
        out.update(f1_at_val_thr=float(f1c[:, thr_global].mean()), iou_at_val_thr=float(iouc[:, thr_global].mean()), val_thr=thr_global / 255.0)
    pooled = curves_from_hist(hp.sum(0), hn.sum(0)); pooled_sub = curves_from_hist(hp.sum(0), hn_sub.sum(0))
    out.update(pixel_auroc_pooled=pooled['auroc'], pixel_auprc_pooled=pooled['ap'],
               pixel_auroc_sub50k=pooled_sub['auroc'], pixel_auprc_sub50k=pooled_sub['ap'],
               pooled_f1_05=float(pooled['f1'][T05]), pooled_iou_05=float(pooled['iou'][T05]))
    return out, gbest

if __name__ == '__main__':
    t0 = time.time()
    all_rows = []; results = []
    for cond in CONDS:
        store = {}
        for sp in ['val', 'test']:
            jobs = [(cond, sp, i) for i in splits[sp] if (LB / "preds" / cond / sp / f"{i}.png").exists()]
            with ThreadPoolExecutor(8) as ex:
                outs = [o for o in ex.map(process, jobs) if o is not None]
            if not outs: continue
            rows = [o[0] for o in outs]
            store[sp] = dict(rows=rows, f1c=np.stack([o[1] for o in outs]), iouc=np.stack([o[2] for o in outs]),
                             hp=np.stack([o[3] for o in outs]), hn=np.stack([o[4] for o in outs]), hn_sub=np.stack([o[5] for o in outs]))
            all_rows += rows
            print(f"{cond} {sp}: {len(rows)} images, {time.time()-t0:.0f}s", flush=True)
        if 'val' not in store: continue
        agg_val, gbest = aggregate(**store['val'])
        results.append(dict(cond=cond, split='val', subset='all', **agg_val))
        def subset_agg(sp, mask, name):
            s = store[sp]; idx = np.flatnonzero(mask)
            if len(idx) == 0: return
            a, _ = aggregate([s['rows'][k] for k in idx], s['f1c'][idx], s['iouc'][idx], s['hp'][idx], s['hn'][idx], s['hn_sub'][idx], thr_global=gbest)
            results.append(dict(cond=cond, split=sp, subset=name, **a))
        vdf = pd.DataFrame(store['val']['rows'])
        for tool in sorted(vdf.tool.unique()): subset_agg('val', (vdf.tool == tool).values, f'tool={tool}')
        for ab in sorted(vdf.area_bin.unique()): subset_agg('val', (vdf.area_bin == ab).values, f'area={ab}')
        if 'test' in store:
            tdf = pd.DataFrame(store['test']['rows'])
            subset_agg('test', np.ones(len(tdf), bool), 'full')
            subset_agg('test', tdf.clean.values, 'clean')
            for tool in sorted(tdf.tool.unique()): subset_agg('test', (tdf.tool == tool).values, f'tool={tool}')
            for ab in sorted(tdf.area_bin.unique()): subset_agg('test', (tdf.area_bin == ab).values, f'area={ab}')
            for tool in sorted(tdf.tool.unique()): subset_agg('test', ((tdf.tool == tool) & tdf.clean).values, f'clean&tool={tool}')
            for ab in sorted(tdf.area_bin.unique()): subset_agg('test', ((tdf.area_bin == ab) & tdf.clean).values, f'clean&area={ab}')
    pd.DataFrame(all_rows).to_csv(LB / "zero_shot_per_image.csv", index=False)
    res = pd.DataFrame(results); res.to_csv(LB / "zero_shot_results.csv", index=False)
    print(res[['cond', 'split', 'subset', 'n_images', 'pixel_auroc_pooled', 'pixel_auprc_pooled', 'pixel_f1_05', 'iou_05', 'oracle_f1', 'f1_at_val_thr']].to_string())
    print("total", round(time.time() - t0), "s")
